#!/usr/bin/env python3
"""Full-parameter FSDP SFT on complete MathQA reasoning trajectories."""

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def remove_answer_disclosure(rationale):
    rationale = rationale.strip().strip('"')
    match = re.search(
        r"\b(?:answer\s*:|the\s+answer\s+is|answer\s+is|option\s+[a-e]\b)",
        rationale,
        re.IGNORECASE,
    )
    if match:
        rationale = rationale[:match.start()].rstrip(" .,:;\n")
    return rationale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, default=Path("data/jev-mathqa/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/jev-full-trajectory-sft"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Stop and save after this epoch while keeping the full --epochs LR schedule")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        parser.error("--stop-after-epoch must be between 0 and --epochs")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("Full-parameter training requires torchrun with multiple GPUs")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {args.output}")

    import functools
    import os

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    class TrajectoryDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.truncated = 0

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            labels = [label_chars[i] for i in range(len(row["options"]))]
            if row["gold_label"] not in labels:
                raise ValueError(f"Invalid answer label in {row['id']}: {row['gold_label']}")
            options = "\n".join(
                f"({label_chars[i]}) {option}" for i, option in enumerate(row["options"])
            )
            user_content = f"Question: {row['question']}\nOptions:\n{options}"
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rationale = remove_answer_disclosure(row.get("rationale", ""))
            rationale_ids = tokenizer.encode(rationale, add_special_tokens=False)
            answer = f"\nAnswer: ({row['gold_label']})"

            def make_full(reason_ids):
                reason = tokenizer.decode(reason_ids, skip_special_tokens=True).strip()
                response = f"Reasoning: {reason}{answer}" if reason else answer.lstrip()
                return tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )

            full_ids = make_full(rationale_ids)
            was_truncated = len(full_ids) > args.max_length
            while len(full_ids) > args.max_length and rationale_ids:
                overflow = len(full_ids) - args.max_length
                del rationale_ids[max(0, len(rationale_ids) - max(overflow, 1)):]
                full_ids = make_full(rationale_ids)
            if len(full_ids) > args.max_length:
                raise ValueError(f"Prompt plus answer exceeds max length for {row['id']}")
            if not full_ids[:len(prompt_ids)] == prompt_ids:
                raise ValueError(f"Chat template prompt mismatch for {row['id']}")
            if was_truncated:
                self.truncated += 1
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            return {"input_ids": full_ids, "labels": labels, "sample_id": row["id"]}

    def collate(examples):
        width = max(len(item["input_ids"]) for item in examples)
        input_ids, attention_mask, labels = [], [], []
        for item in examples:
            padding = width - len(item["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * padding + item["input_ids"])
            attention_mask.append([0] * padding + [1] * len(item["input_ids"]))
            labels.append([-100] * padding + item["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_ids": [item["sample_id"] for item in examples],
        }

    train_rows = read_rows(args.train)
    if args.max_train_examples and len(train_rows) > args.max_train_examples:
        selection = random.Random(args.seed).sample(range(len(train_rows)), args.max_train_examples)
        train_rows = [train_rows[index] for index in selection]
    train_data = TrajectoryDataset(train_rows)
    # Materialize preprocessing once so every epoch sees identical tokenized examples.
    encoded = [train_data[index] for index in range(len(train_data))]
    truncated_count = train_data.truncated

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    transformer_classes = {
        module.__class__ for module in model.modules()
        if "DecoderLayer" in module.__class__.__name__
    }
    if not transformer_classes:
        raise ValueError("Could not identify transformer decoder layers for FSDP auto-wrap")
    auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_classes)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    sampler_size = len(encoded) // world_size
    batches_per_rank = math.ceil(sampler_size / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_rank / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, round(total_steps * 0.03))
    update_step = 0
    completed_epochs = 0
    optimizer.zero_grad(set_to_none=True)

    class EncodedDataset(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    for epoch in range(args.epochs):
        sampler = DistributedSampler(
            EncodedDataset(), num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        loader = DataLoader(
            EncodedDataset(), batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
        )
        sampler.set_epoch(epoch)
        epoch_loss = torch.zeros((), device=local_rank)
        update_sample_ids = []
        for batch_index, batch in enumerate(loader):
            remainder = len(loader) % args.gradient_accumulation_steps
            group_size = (
                remainder if remainder and batch_index >= len(loader) - remainder
                else args.gradient_accumulation_steps
            )
            update_sample_ids.extend(batch.pop("sample_ids"))
            batch = {key: value.to(local_rank) for key, value in batch.items()}
            is_update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            with model.no_sync() if not is_update else torch.enable_grad():
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite SFT loss at epoch={epoch + 1}, batch={batch_index + 1}"
                    )
                loss = loss / group_size
                loss.backward()
            epoch_loss += loss.detach() * group_size
            if is_update:
                update_step += 1
                grad_norm = model.clip_grad_norm_(1.0)
                if not torch.isfinite(grad_norm):
                    bad_grad = next(
                        (name for name, param in model.named_parameters()
                         if param.grad is not None and not torch.isfinite(param.grad).all()),
                        "unknown",
                    )
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch + 1}, batch={batch_index + 1}, "
                        f"samples={update_sample_ids}, first_bad_grad={bad_grad}"
                    )
                lr_scale = (
                    update_step / warmup_steps if update_step <= warmup_steps
                    else max(0.0, (total_steps - update_step) / max(1, total_steps - warmup_steps))
                )
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * lr_scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_sample_ids = []
                if rank == 0 and update_step % 10 == 0:
                    print(
                        f"epoch={epoch + 1} step={update_step}/{total_steps} "
                        f"loss={loss.item() * group_size:.5f} lr={optimizer.param_groups[0]['lr']:.3g}",
                        flush=True,
                    )
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        completed_epochs = epoch + 1
        if rank == 0:
            print(
                f"epoch={epoch + 1} mean_train_loss="
                f"{epoch_loss.item() / (len(loader) * world_size):.6f}",
                flush=True,
            )
        if args.stop_after_epoch and completed_epochs >= args.stop_after_epoch:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
        state_dict = model.state_dict()
    if rank == 0:
        model.module.save_pretrained(
            args.output, state_dict=state_dict, safe_serialization=True, max_shard_size="5GB",
        )
        tokenizer.save_pretrained(args.output)
        metadata = {
            "model": args.model,
            "train": str(args.train),
            "epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "optimizer_steps": update_step,
            "lr_schedule_steps": total_steps,
            "warmup_steps": warmup_steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "world_size": world_size,
            "global_batch_size": world_size * args.batch_size * args.gradient_accumulation_steps,
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "examples": len(train_rows),
            "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "max_train_examples": args.max_train_examples,
            "max_length": args.max_length,
            "parameter_dtype": "float32",
            "compute_dtype": "float32",
            "truncated_rationales": truncated_count,
            "max_grad_norm": 1.0,
            "objective": "causal language modeling over the complete assistant rationale and answer; user prompt masked",
            "status": "complete" if completed_epochs == args.epochs else "stopped_after_requested_epoch",
        }
        (args.output / "experiment.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

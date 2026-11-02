#!/usr/bin/env python3
"""Full-parameter FSDP training: one full-trajectory SFT epoch, then Jev pruning."""

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, default=Path("data/jev-mathqa/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/jev-answer-token-full"))
    parser.add_argument("--epochs", type=int, default=4, help="One full-trajectory SFT epoch followed by three pruning stages")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs != 4:
        parser.error("The schedule requires exactly 4 epochs: one full-trajectory SFT epoch and three pruning stages")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("Full-parameter training requires torchrun with multiple GPUs")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {args.output}")

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        StateDictType,
        FullStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from functools import partial
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import functools

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
    candidate_ids = []
    for label in label_chars:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Answer label {label!r} is not one token: {token_ids}")
        candidate_ids.append(token_ids[0])

    def remove_answer_disclosure(rationale):
        rationale = rationale.strip().strip('"')
        match = re.search(r"\b(?:answer\s*:|the\s+answer\s+is|answer\s+is|option\s+[a-e]\b)", rationale, re.IGNORECASE)
        if match:
            rationale = rationale[:match.start()].rstrip(" .,:;\n")
        return rationale

    class AnswerDataset(Dataset):
        def __init__(self, rows, stage):
            self.rows = rows
            self.stage = stage

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            options = row["options"]
            option_text = "\n".join(f"({label_chars[i]}) {option}" for i, option in enumerate(options))
            content = f"Question: {row['question']}\nOptions:\n{option_text}"
            rationale = remove_answer_disclosure(row.get("rationale", ""))
            rationale_ids = tokenizer.encode(rationale, add_special_tokens=False) if rationale else []
            first_cut = len(rationale_ids) // 3
            second_cut = (2 * len(rationale_ids)) // 3
            if self.stage == 0:
                retained_rationale = rationale_ids[first_cut:]
            elif self.stage == 1:
                retained_rationale = rationale_ids[second_cut:]
            else:
                retained_rationale = []
            if retained_rationale:
                retained_text = tokenizer.decode(retained_rationale, skip_special_tokens=True).strip()
                if retained_text:
                    content += f"\nReasoning: {retained_text}"
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False,
            )
            prompt += tokenizer.encode("Answer: (", add_special_tokens=False)
            if len(prompt) > args.max_length:
                answer_prefix = tokenizer.encode("Answer: (", add_special_tokens=False)
                while len(prompt) > args.max_length and retained_rationale:
                    retained_rationale = retained_rationale[1:]
                    content = f"Question: {row['question']}\nOptions:\n{option_text}"
                    if retained_rationale:
                        retained_text = tokenizer.decode(retained_rationale, skip_special_tokens=True).strip()
                        if retained_text:
                            content += f"\nReasoning: {retained_text}"
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}], tokenize=True,
                        return_dict=False, add_generation_prompt=True, enable_thinking=False,
                    ) + answer_prefix
                if len(prompt) > args.max_length:
                    raise ValueError(f"Question and options exceed max length for {row['id']}")
            return {"input_ids": prompt, "target": row["gold_index"], "option_count": len(options)}

    class TrajectoryDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.truncated = 0

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            options = "\n".join(f"({label_chars[i]}) {option}" for i, option in enumerate(row["options"]))
            user_content = f"Question: {row['question']}\nOptions:\n{options}"
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False,
            )
            rationale = remove_answer_disclosure(row.get("rationale", ""))
            rationale_ids = tokenizer.encode(rationale, add_special_tokens=False)
            answer = f"\nAnswer: ({label_chars[row['gold_index']]})"

            def make_full(reason_ids):
                reason = tokenizer.decode(reason_ids, skip_special_tokens=True).strip()
                response = f"Reasoning: {reason}{answer}" if reason else answer.lstrip()
                return tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response},
                    ], tokenize=True, return_dict=False, add_generation_prompt=False,
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
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(f"Chat template prompt mismatch for {row['id']}")
            if was_truncated:
                self.truncated += 1
            return {
                "input_ids": full_ids,
                "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
            }

    def collate(examples):
        width = max(len(item["input_ids"]) for item in examples)
        ids = [[tokenizer.pad_token_id] * (width - len(item["input_ids"])) + item["input_ids"] for item in examples]
        masks = [[0] * (width - len(item["input_ids"])) + [1] * len(item["input_ids"]) for item in examples]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "target": torch.tensor([item["target"] for item in examples], dtype=torch.long),
            "option_count": torch.tensor([item["option_count"] for item in examples], dtype=torch.long),
        }

    def collate_trajectories(examples):
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
        }

    train_rows = read_rows(args.train)
    if args.max_train_examples and len(train_rows) > args.max_train_examples:
        selection = random.Random(args.seed).sample(range(len(train_rows)), args.max_train_examples)
        train_rows = [train_rows[index] for index in selection]
    trajectory_data = TrajectoryDataset(train_rows)
    encoded_trajectories = [trajectory_data[index] for index in range(len(trajectory_data))]
    truncated_trajectories = trajectory_data.truncated
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    transformer_classes = {module.__class__ for module in model.modules() if "DecoderLayer" in module.__class__.__name__}
    if not transformer_classes:
        raise ValueError("Could not identify transformer decoder layers for FSDP auto-wrap")
    auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_classes)
    model = FSDP(
        model, auto_wrap_policy=auto_wrap, sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(), use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    batches_per_rank = math.ceil(math.ceil(len(train_rows) / world_size) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_rank / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, round(total_steps * 0.03))
    update_step = 0
    optimizer.zero_grad(set_to_none=True)

    stage_descriptions = [
        "full_trajectory_sft",
        "retain_final_two_thirds",
        "retain_final_one_third",
        "answer_token_only",
    ]
    for epoch in range(args.epochs):
        is_trajectory_epoch = epoch == 0
        train_data = encoded_trajectories if is_trajectory_epoch else AnswerDataset(train_rows, stage=epoch - 1)
        sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        loader = DataLoader(
            train_data, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate_trajectories if is_trajectory_epoch else collate,
        )
        sampler.set_epoch(epoch)
        epoch_loss = torch.zeros((), device=local_rank)
        for batch_index, batch in enumerate(loader):
            remainder = len(loader) % args.gradient_accumulation_steps
            group_size = remainder if remainder and batch_index >= len(loader) - remainder else args.gradient_accumulation_steps
            if not is_trajectory_epoch:
                targets = batch.pop("target").to(local_rank)
                option_counts = batch.pop("option_count").to(local_rank)
            batch = {key: value.to(local_rank) for key, value in batch.items()}
            with model.no_sync() if (batch_index + 1) % args.gradient_accumulation_steps and batch_index + 1 != len(loader) else torch.enable_grad():
                if is_trajectory_epoch:
                    loss = model(**batch).loss / group_size
                else:
                    outputs = model(**batch, logits_to_keep=1)
                    logits = outputs.logits[:, -1, :][:, candidate_ids].float()
                    valid = torch.arange(len(label_chars), device=local_rank)[None, :] < option_counts[:, None]
                    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
                    loss = F.cross_entropy(logits, targets) / group_size
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at stage={stage_descriptions[epoch]}, batch={batch_index + 1}"
                    )
                loss.backward()
            epoch_loss += loss.detach() * group_size
            should_update = (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
            if should_update:
                update_step += 1
                grad_norm = model.clip_grad_norm_(1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at stage={stage_descriptions[epoch]}, "
                        f"batch={batch_index + 1}"
                    )
                if update_step <= warmup_steps:
                    lr_scale = update_step / warmup_steps
                else:
                    lr_scale = max(0.0, (total_steps - update_step) / max(1, total_steps - warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * lr_scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if rank == 0 and update_step % 10 == 0:
                    print(f"epoch={epoch + 1} step={update_step}/{total_steps} loss={loss.item() * group_size:.5f} lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"stage={stage_descriptions[epoch]} epoch={epoch + 1} mean_train_loss={epoch_loss.item() / (len(loader) * world_size):.6f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
        state_dict = model.state_dict()
    if rank == 0:
        model.module.save_pretrained(args.output, state_dict=state_dict, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(args.output)
        metadata = {
            "model": args.model, "train": str(args.train), "epochs": args.epochs,
            "learning_rate": args.learning_rate, "world_size": world_size,
            "global_batch_size": world_size * args.batch_size * args.gradient_accumulation_steps,
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "examples": len(train_rows), "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "max_train_examples": args.max_train_examples,
            "max_length": args.max_length,
            "parameter_dtype": "float32", "compute_dtype": "float32",
            "truncated_trajectories": truncated_trajectories,
            "max_grad_norm": 1.0,
            "stages": stage_descriptions,
            "rationale": "Epoch 1 uses full assistant trajectory SFT; subsequent epochs retain the final 2/3, final 1/3, and no rationale, respectively",
            "objective": "causal language modeling over the complete assistant response in epoch 1, then cross-entropy over valid answer-letter logits at the Jev answer slot",
            "status": "complete",
        }
        (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

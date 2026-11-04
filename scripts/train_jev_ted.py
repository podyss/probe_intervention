#!/usr/bin/env python3
"""Task-aware layer-wise distillation inspired by TED for GSM-MC answer slots."""

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


def clean_rationale(text):
    text = text.strip().strip('"')
    match = re.search(r"\b(?:answer\s*:|the\s+answer\s+is|answer\s+is|option\s+[a-e]\b)", text, re.I)
    return text[:match.start()].rstrip(" .,:;\n") if match else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, default=Path("data/gsm-mc/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1, help="Stage II epochs")
    parser.add_argument("--stage1-epochs", type=int, default=1,
                        help="number of epochs used to fit the task-aware layer filters")
    parser.add_argument("--max-train-examples", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("Run with torchrun on at least two GPUs")
    if args.stage1_epochs < 1:
        parser.error("--stage1-epochs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {args.output}")

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    import functools
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, ShardingStrategy, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    label_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    rows = read_rows(args.train)
    if args.max_train_examples and len(rows) > args.max_train_examples:
        rows = [rows[i] for i in random.Random(args.seed).sample(range(len(rows)), args.max_train_examples)]

    def encode_row(row):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "\n".join(f"({labels[i]}) {v}" for i, v in enumerate(row["options"]))
        text = f"Question: {row['question']}\nOptions:\n{options}"
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False)
        student = prompt + tokenizer.encode("Answer: (", add_special_tokens=False)
        rationale = tokenizer.encode(clean_rationale(row.get("rationale", "")), add_special_tokens=False)
        def teacher_prefix():
            body = tokenizer.decode(rationale, skip_special_tokens=True).strip()
            return prompt + tokenizer.encode(f"Reasoning: {body}\nAnswer: (" if body else "Answer: (", add_special_tokens=False)
        teacher = teacher_prefix()
        while len(teacher) > args.max_length and rationale:
            del rationale[max(0, len(rationale) - max(1, len(teacher) - args.max_length)):]
            teacher = teacher_prefix()
        if len(student) > args.max_length or len(teacher) > args.max_length:
            raise ValueError(f"Prompt too long: {row['id']}")
        return {"s": student, "t": teacher, "gold": row["gold_index"], "n": len(row["options"])}

    encoded = [encode_row(row) for row in rows]

    def collate(items):
        out = {}
        for name in ("s", "t"):
            width = max(len(x[name]) for x in items)
            ids = [[pad_id] * (width-len(x[name])) + x[name] for x in items]
            mask = [[0] * (width-len(x[name])) + [1] * len(x[name]) for x in items]
            out[name+"_ids"] = torch.tensor(ids, dtype=torch.long)
            out[name+"_mask"] = torch.tensor(mask, dtype=torch.long)
        out["gold"] = torch.tensor([x["gold"] for x in items])
        out["n"] = torch.tensor([x["n"] for x in items])
        return out

    class Data(Dataset):
        def __len__(self): return len(encoded)
        def __getitem__(self, i): return encoded[i]

    class FilterBank(nn.Module):
        def __init__(self, include_heads):
            super().__init__()
            self.filters = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layer_count)])
            self.heads = nn.ModuleList([nn.Linear(dim, len(label_ids)) for _ in range(layer_count)]) if include_heads else None

        def forward(self, states, include_heads=False):
            filtered = [self.filters[i](state) for i, state in enumerate(states)]
            if include_heads:
                return filtered, [self.heads[i](state) for i, state in enumerate(filtered)]
            return filtered

    def load_model(dtype):
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
            attn_implementation="sdpa", trust_remote_code=False)
        model.config.use_cache = False
        return model

    def wrap_fsdp(model):
        classes = {m.__class__ for m in model.modules() if "DecoderLayer" in m.__class__.__name__}
        if not classes:
            raise ValueError("Could not identify decoder layers for FSDP wrapping")
        return FSDP(model,
            auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=classes),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(), use_orig_params=True)

    # Train task-aware filters on the answer-only context used by the student.
    teacher = wrap_fsdp(load_model(torch.bfloat16)).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    stage1_student = wrap_fsdp(load_model(torch.bfloat16)).eval()
    for p in stage1_student.parameters(): p.requires_grad_(False)
    layer_count = teacher.config.num_hidden_layers
    dim = teacher.config.hidden_size
    student_bank = DDP(FilterBank(include_heads=True).to(local_rank), device_ids=[local_rank], find_unused_parameters=False)
    filter_opt = torch.optim.AdamW(student_bank.parameters(), lr=2.5e-4, weight_decay=0.01)
    data = Data()

    def batch_loader(epoch):
        sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        sampler.set_epoch(epoch)
        return DataLoader(data, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)

    steps1 = 0
    for epoch in range(args.stage1_epochs):
        loader = batch_loader(epoch)
        filter_opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(loader):
            student_ids, student_mask = batch["s_ids"].to(local_rank), batch["s_mask"].to(local_rank)
            gold, count = batch["gold"].to(local_rank), batch["n"].to(local_rank)
            with torch.no_grad():
                student_hs = stage1_student(input_ids=student_ids, attention_mask=student_mask,
                                            logits_to_keep=1, output_hidden_states=True).hidden_states
                student_states = [student_hs[i+1][:, -1, :].float() for i in range(layer_count)]
            valid = torch.arange(len(label_ids), device=local_rank)[None, :] < count[:, None]
            _, student_predictions = student_bank(student_states, include_heads=True)
            task_loss = 0
            for pred in student_predictions:
                task_loss = task_loss + F.cross_entropy(
                    pred.masked_fill(~valid, torch.finfo(pred.dtype).min), gold) / layer_count
            (task_loss / args.gradient_accumulation_steps).backward()
            if (bi+1) % args.gradient_accumulation_steps == 0 or bi+1 == len(loader):
                filter_opt.step(); filter_opt.zero_grad(set_to_none=True); steps1 += 1
            if rank == 0 and (bi+1) % 20 == 0:
                print(f"stage1 epoch={epoch+1} batch={bi+1}/{len(loader)} student_context_task_ce={task_loss.item():.5f}", flush=True)
            del student_hs, student_states
    student_filter_state = {k: v.detach().cpu() for k, v in student_bank.module.filters.state_dict().items()}
    teacher_filter_state = student_filter_state
    dist.barrier()

    # Stage II freezes the teacher path and jointly optimizes the student model
    # with its own Stage I filters.
    del student_bank, filter_opt, stage1_student
    student_base = load_model(torch.float32)
    student_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student = wrap_fsdp(student_base)
    student_filters = FilterBank(include_heads=False).to(local_rank)
    student_filters.filters.load_state_dict(student_filter_state)
    student_filters = DDP(student_filters, device_ids=[local_rank], find_unused_parameters=False)
    teacher_filters = FilterBank(include_heads=False).to(local_rank)
    teacher_filters.filters.load_state_dict(teacher_filter_state)
    teacher_filters.eval()
    for p in teacher_filters.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(student_filters.parameters()), lr=args.learning_rate, weight_decay=0.01)
    steps_epoch = math.ceil(math.ceil((len(data)//world) / args.batch_size) / args.gradient_accumulation_steps)
    update = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loader = batch_loader(epoch)
        for bi, batch in enumerate(loader):
            gold, count = batch["gold"].to(local_rank), batch["n"].to(local_rank)
            sb = {"input_ids": batch["s_ids"].to(local_rank), "attention_mask": batch["s_mask"].to(local_rank)}
            tb = {"input_ids": batch["t_ids"].to(local_rank), "attention_mask": batch["t_mask"].to(local_rank)}
            with torch.no_grad():
                tout = teacher(**tb, logits_to_keep=1)
                tlogits = tout.logits[:, -1, :].float()
                del tout
                hidden_target = teacher(**sb, logits_to_keep=1, output_hidden_states=True)
                th = [hidden_target.hidden_states[i+1][:, -1, :].float() for i in range(layer_count)]
                tfiltered = teacher_filters(th)
                del hidden_target
            sout = student(**sb, logits_to_keep=1, output_hidden_states=True)
            slogits = sout.logits[:, -1, :].float()
            valid = torch.arange(len(label_ids), device=local_rank)[None, :] < count[:, None]
            sopts = slogits[:, label_ids].masked_fill(~valid, torch.finfo(slogits.dtype).min)
            ce = F.cross_entropy(sopts, gold)
            kl = F.kl_div(F.log_softmax(slogits / args.temperature, -1),
                          F.softmax(tlogits / args.temperature, -1), reduction="batchmean") * args.temperature**2
            sh = [sout.hidden_states[i+1][:, -1, :].float() for i in range(layer_count)]
            sfiltered = student_filters(sh)
            mse = torch.stack([F.mse_loss(sv, tv) for sv, tv in zip(sfiltered, tfiltered)]).mean()
            loss = ce + args.kl_weight * kl + 0.1 * mse
            (loss / args.gradient_accumulation_steps).backward()
            if (bi+1) % args.gradient_accumulation_steps == 0 or bi+1 == len(loader):
                student.clip_grad_norm_(1.0)
                torch.nn.utils.clip_grad_norm_(student_filters.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1
            if rank == 0 and (bi+1) % 20 == 0:
                print(f"stage2 epoch={epoch+1} batch={bi+1}/{len(loader)} ce={ce.item():.5f} kl={kl.item():.5f} ted_mse={mse.item():.5f}", flush=True)
            del sout, th, sh, tfiltered, sfiltered

    dist.barrier()
    args.output.mkdir(parents=True, exist_ok=True)
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, state_config):
        student_state = student.state_dict()
    if rank == 0:
        student.module.save_pretrained(args.output, state_dict=student_state, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(args.output)
        metadata = {"method": "TED-inspired task-aware layer filters", "model": args.model,
            "train": str(args.train), "examples": len(rows), "seed": args.seed,
            "stage1_epochs": args.stage1_epochs, "stage1_objective": "per-layer valid-option CE filter/head bank trained on frozen student question-options states",
            "stage2_epochs": args.epochs, "stage2_objective": "answer CE + full-vocabulary logits KL + 0.1 filtered-hidden MSE",
            "layers": layer_count, "filter": "per-layer learned square linear projection bank shared by teacher and student; hidden-state targets use matched question-options context while logits KL retains teacher rationale context",
            "learning_rate": args.learning_rate, "stage1_learning_rate": 2.5e-4,
            "temperature": args.temperature, "kl_weight": args.kl_weight,
            "world_size": world, "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": world*args.batch_size*args.gradient_accumulation_steps,
            "max_length": args.max_length, "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "stage1_optimizer_steps": steps1, "stage2_optimizer_steps": update, "status": "complete"}
        (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(metadata), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()

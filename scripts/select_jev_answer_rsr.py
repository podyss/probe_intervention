#!/usr/bin/env python3
"""Rank GSM-MC rationales by an answer-slot adaptation/information score."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/gsm-mc/train.jsonl"))
    parser.add_argument("--output-scores", type=Path, required=True)
    parser.add_argument("--output-selected", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--_worker-rank", type=int)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_length < 1 or args.temperature <= 0:
        parser.error("top-k, max-length, and temperature must be positive")
    rows = read_rows(args.input)
    if not rows:
        raise ValueError("Input dataset is empty")

    if args._worker_rank is None:
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            parser.error("--gpus must contain unique GPU IDs")
        args.output_scores.parent.mkdir(parents=True, exist_ok=True)
        args.output_selected.parent.mkdir(parents=True, exist_ok=True)
        for path in (args.output_scores, args.output_selected):
            if path.exists():
                raise FileExistsError(path)
        with tempfile.TemporaryDirectory(prefix="jev-answer-rsr-") as temp_dir:
            shards = [Path(temp_dir) / f"rank-{rank}.jsonl" for rank in range(len(gpu_ids))]
            processes = []
            for rank, (gpu, shard) in enumerate(zip(gpu_ids, shards)):
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--model", args.model, "--input", str(args.input),
                    "--output-scores", str(shard), "--output-selected", str(Path(temp_dir) / "unused.jsonl"),
                    "--top-k", str(args.top_k), "--batch-size", str(args.batch_size),
                    "--max-length", str(args.max_length), "--temperature", str(args.temperature),
                    "--gpus", args.gpus, "--_worker-rank", str(rank),
                ]
                worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                                  HF_ENDPOINT="https://hf-mirror.com", PYTHONUNBUFFERED="1")
                processes.append(subprocess.Popen(command, env=worker_env))
            statuses = [process.wait() for process in processes]
            if any(status != 0 for status in statuses):
                raise subprocess.CalledProcessError(next(status for status in statuses if status), "RSR worker")
            scored_by_id = {}
            for shard in shards:
                for row in read_rows(shard):
                    if row["id"] in scored_by_id:
                        raise RuntimeError(f"Duplicate scored row {row['id']}")
                    scored_by_id[row["id"]] = row
        if set(scored_by_id) != {row["id"] for row in rows}:
            raise RuntimeError(f"Scored ID mismatch: {len(scored_by_id)} != {len(rows)}")
        scored = [scored_by_id[row["id"]] for row in rows]
        ranked = sorted(scored, key=lambda row: (row["answer_rsr"], row["id"]))
        selected_ids = {row["id"] for row in ranked[:min(args.top_k, len(ranked))]}
        selected = [row for row in rows if row["id"] in selected_ids]
        args.output_scores.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ranked), encoding="utf-8")
        args.output_selected.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
        print(json.dumps({
            "input_examples": len(rows), "selected_examples": len(selected),
            "score_definition": "teacher answer distribution expected baseline option rank / teacher-to-baseline answer cross-entropy",
            "temperature": args.temperature,
            "score_min": ranked[0]["answer_rsr"], "score_median": ranked[len(ranked) // 2]["answer_rsr"],
            "score_selected_max": ranked[min(args.top_k, len(ranked)) - 1]["answer_rsr"],
            "score_selected_mean": sum(row["answer_rsr"] for row in ranked[:args.top_k]) / min(args.top_k, len(ranked)),
            "output_scores": str(args.output_scores), "output_selected": str(args.output_selected),
        }, ensure_ascii=False, indent=2), flush=True)
        return

    rank = args._worker_rank
    gpu_count = len([item for item in args.gpus.split(",") if item.strip()])
    rows = rows[rank::gpu_count]
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    answer_prefix = tokenizer.encode("Answer: (", add_special_tokens=False)
    label_ids = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer label {label!r} is not one token: {ids}")
        label_ids.append(ids[0])
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": 0}, trust_remote_code=False,
    ).eval()
    device = model.get_input_embeddings().weight.device

    def make_ids(row, include_rationale):
        options = "\n".join(f"({LABELS[i]}) {option}" for i, option in enumerate(row["options"]))
        content = f"Question: {row['question']}\nOptions:\n{options}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False,
        )
        if include_rationale:
            rationale = row.get("rationale", "").strip()
            reason_ids = tokenizer.encode(f"\nReasoning: {rationale}\n", add_special_tokens=False)
            room = args.max_length - len(prompt) - len(answer_prefix) - len(reason_ids)
            if room < 0:
                reason_ids = reason_ids[:max(0, len(reason_ids) + room)]
            prompt = prompt + reason_ids
        return (prompt + answer_prefix)[-args.max_length:]

    scored = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        base_ids = [make_ids(row, False) for row in batch_rows]
        trace_ids = [make_ids(row, True) for row in batch_rows]

        def logits_for(sequences):
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, sequence in enumerate(sequences):
                ids[index, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long)
                mask[index, width - len(sequence):] = 1
            with torch.inference_mode():
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=1).logits[:, -1, :]
            return logits[:, label_ids[:max(len(row["options"]) for row in batch_rows)]].float()

        base_logits = logits_for(base_ids)
        trace_logits = logits_for(trace_ids)
        for row, baseline_logits, teacher_logits in zip(batch_rows, base_logits, trace_logits):
            count = len(row["options"])
            p = F.softmax(baseline_logits[:count] / args.temperature, dim=-1)
            q = F.softmax(teacher_logits[:count] / args.temperature, dim=-1)
            ranks = torch.argsort(torch.argsort(baseline_logits[:count], descending=True)) + 1
            expected_rank = float((q * ranks).sum().item())
            cross_entropy = float((-(q * p.clamp_min(1e-12).log()).sum()).item())
            kl = float((q * (q.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum().item())
            scored.append({
                **row,
                "answer_rsr": expected_rank / max(cross_entropy, 1e-8),
                "baseline_expected_teacher_rank": expected_rank,
                "teacher_to_baseline_cross_entropy": cross_entropy,
                "teacher_to_baseline_kl": kl,
                "baseline_option_probabilities": p.cpu().tolist(),
                "trace_option_probabilities": q.cpu().tolist(),
                "rationale_source": "gsm-mc-dataset",
            })
        print(f"GPU worker {rank}: scored {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    args.output_scores.parent.mkdir(parents=True, exist_ok=True)
    args.output_scores.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8")


if __name__ == "__main__":
    main()

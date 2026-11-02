#!/usr/bin/env python3
"""Score multiple-choice answers from one-token candidate probabilities."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_rows(path, limit=None):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def render_prompt(tokenizer, row):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"({labels[i]}) {option}" for i, option in enumerate(row["options"]))
    text = f"Question: {row['question']}\nOptions:\n{options}"
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    answer_slot = tokenizer.encode("Answer: (", add_special_tokens=False)
    return prompt_ids + answer_slot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--_worker-rank", type=int)
    args = parser.parse_args()

    rows = read_rows(args.input, args.limit)
    if not rows:
        raise ValueError("Input dataset is empty")
    if args.data_parallel_size < 1:
        raise ValueError("--data-parallel-size must be positive")
    if args._worker_rank is None and args.data_parallel_size > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = ([item.strip() for item in visible.split(",") if item.strip()]
                   if visible is not None else [str(i) for i in range(torch.cuda.device_count())])
        if len(gpu_ids) < args.data_parallel_size:
            raise ValueError(f"Requested {args.data_parallel_size} DP workers, found GPUs {gpu_ids}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="jev-dp-") as tmp:
            shard_paths = [Path(tmp) / f"rank-{rank}.jsonl" for rank in range(args.data_parallel_size)]
            processes = []
            for rank, (gpu_id, shard_path) in enumerate(zip(gpu_ids, shard_paths)):
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--model", args.model, "--input", str(args.input),
                    "--output", str(shard_path), "--batch-size", str(args.batch_size),
                    "--dtype", args.dtype, "--data-parallel-size", str(args.data_parallel_size),
                    "--_worker-rank", str(rank),
                ]
                if args.limit is not None:
                    command.extend(["--limit", str(args.limit)])
                worker_env = dict(os.environ)
                worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id
                worker_env["HF_ENDPOINT"] = "https://hf-mirror.com"
                processes.append(subprocess.Popen(command, env=worker_env))
            statuses = [process.wait() for process in processes]
            failures = [status for status in statuses if status != 0]
            if failures:
                raise subprocess.CalledProcessError(failures[0], "DP evaluation worker")

            by_id = {}
            for shard_path in shard_paths:
                for row in read_rows(shard_path):
                    by_id[row["id"]] = row
            if len(by_id) != len(rows):
                raise RuntimeError(f"DP merge mismatch: expected {len(rows)} rows, got {len(by_id)}")
            results = [by_id[row["id"]] for row in rows]
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
            encoding="utf-8",
        )
        groups = defaultdict(lambda: [0, 0])
        for row in results:
            group = f"{row.get('dataset', 'unknown')}/{row.get('split', 'unknown')}"
            groups[group][0] += int(row["correct_prediction"])
            groups[group][1] += 1
        summary = {name: {"correct": correct, "total": total, "accuracy": correct / total}
                   for name, (correct, total) in groups.items()}
        print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
        return

    if args._worker_rank is not None:
        rows = rows[args._worker_rank::args.data_parallel_size]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    candidate_ids = []
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Candidate label {label!r} is not a single token: {ids}")
        candidate_ids.append(ids[0])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": 0}, trust_remote_code=False
    ).eval()
    device = model.get_input_embeddings().weight.device
    results = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        prompts = [render_prompt(tokenizer, row) for row in batch_rows]
        max_length = max(map(len, prompts))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        input_ids = torch.full((len(prompts), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, prompt in enumerate(prompts):
            offset = max_length - len(prompt)
            input_ids[index, offset:] = torch.tensor(prompt, dtype=torch.long)
            attention_mask[index, offset:] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
            ).logits
        last_logits = logits[:, -1, :]
        for row, scores in zip(batch_rows, last_logits[:, candidate_ids].float().cpu()):
            valid_scores = scores[:len(row["options"])]
            probs = torch.softmax(valid_scores, dim=-1).tolist()
            pred = int(torch.argmax(valid_scores).item())
            results.append({
                **row,
                "model": args.model,
                "prediction_index": pred,
                "prediction_label": chr(ord("A") + pred),
                "correct_prediction": pred == row["gold_index"],
                "option_probabilities": probs,
            })
        print(f"Scored {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    groups = defaultdict(lambda: [0, 0])
    for row in results:
        group = f"{row.get('dataset', 'unknown')}/{row.get('split', 'unknown')}"
        groups[group][0] += int(row["correct_prediction"])
        groups[group][1] += 1
    summary = {name: {"correct": correct, "total": total, "accuracy": correct / total}
               for name, (correct, total) in groups.items()}
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

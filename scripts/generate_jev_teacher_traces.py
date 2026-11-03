#!/usr/bin/env python3
"""Generate answer-blind teacher rationales for a seeded MathQA train subset."""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_teacher_response(response):
    matches = list(re.finditer(
        r"(?i)\b(?:answer\s*(?:is|:)?\s*\\?\(?\s*([A-E])\s*\\?\)?|option\s+([A-E])\b)",
        response,
    ))
    if not matches:
        return None, response.strip()
    match = matches[-1]
    answer = next(group.upper() for group in match.groups() if group)
    return answer, response[:match.start()].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/jev-mathqa/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=9216)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--_worker-rank", type=int)
    args = parser.parse_args()
    if args.max_examples < 1 or args.max_new_tokens < 1:
        parser.error("example count and max new tokens must be positive")
    if args.max_model_len < args.max_new_tokens:
        parser.error("--max-model-len must be at least --max-new-tokens")

    rows = read_rows(args.input)
    if args.max_examples and len(rows) > args.max_examples:
        chosen = random.Random(args.seed).sample(range(len(rows)), args.max_examples)
        rows = [rows[index] for index in chosen]

    if args._worker_rank is None:
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if len(gpu_ids) < 1 or len(set(gpu_ids)) != len(gpu_ids):
            parser.error("--gpus must contain unique GPU IDs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import tempfile

        with tempfile.TemporaryDirectory(prefix="jev-traces-") as temp_dir:
            shards = [Path(temp_dir) / f"rank-{rank}.jsonl" for rank in range(len(gpu_ids))]
            processes = []
            for rank, (gpu, shard) in enumerate(zip(gpu_ids, shards)):
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--model", args.model, "--input", str(args.input),
                           "--output", str(shard), "--seed", str(args.seed),
                           "--max-examples", str(args.max_examples),
                           "--max-new-tokens", str(args.max_new_tokens),
                           "--max-model-len", str(args.max_model_len), "--gpus", args.gpus,
                           "--_worker-rank", str(rank)]
                worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                                  HF_ENDPOINT="https://hf-mirror.com", PYTHONUNBUFFERED="1",
                                  VLLM_USE_FLASHINFER_SAMPLER="0")
                processes.append(subprocess.Popen(command, env=worker_env))
            statuses = [process.wait() for process in processes]
            if any(status != 0 for status in statuses):
                raise subprocess.CalledProcessError(next(status for status in statuses if status), "trace worker")
            by_id = {}
            for shard in shards:
                for row in read_rows(shard):
                    if row["id"] in by_id:
                        raise RuntimeError(f"Duplicate generated row: {row['id']}")
                    by_id[row["id"]] = row
        if set(by_id) != {row["id"] for row in rows}:
            raise RuntimeError(f"Generated ID set mismatch: {len(by_id)} != {len(rows)}")
        ordered = [by_id[row["id"]] for row in rows]
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        parsed = [row["teacher_generated_answer"] for row in ordered]
        parsed_rows = [(answer, row) for answer, row in zip(parsed, ordered) if answer is not None]
        correct = sum(answer == row["gold_label"] for answer, row in parsed_rows)
        summary = {
            "model": args.model, "input": str(args.input), "output": str(args.output),
            "seed": args.seed, "examples": len(ordered), "gpus": gpu_ids,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "parsed_answers": sum(answer is not None for answer in parsed),
            "teacher_correct_among_parsed": correct,
            "teacher_accuracy_among_parsed": correct / len(parsed_rows) if parsed_rows else None,
            "finish_reasons": {
                reason: sum(row["teacher_finish_reason"] == reason for row in ordered)
                for reason in sorted({row["teacher_finish_reason"] for row in ordered})
            },
        }
        args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    rank = args._worker_rank
    rows = rows[rank::len([item for item in args.gpus.split(",") if item.strip()])]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    generated = []
    prompts = []
    for row in rows:
        options = "\n".join(f"({LABELS[i]}) {option}" for i, option in enumerate(row["options"]))
        user_content = (
            f"Question: {row['question']}\nOptions:\n{options}\n\n"
            "Solve the problem carefully with clear step-by-step reasoning. Do not use an answer key. "
            "End with exactly one line `Answer: (A)` using the option letter."
        )
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=True,
        ))
    requests = [{"prompt_token_ids": prompt} for prompt in prompts]
    outputs = engine.generate(requests, sampling, use_tqdm=False)
    for row, result in zip(rows, outputs):
        response = result.outputs[0].text.strip()
        answer, rationale = parse_teacher_response(response)
        generated.append({
            **row,
            "dataset_rationale": row.get("rationale", ""),
            "rationale": rationale,
            "teacher_generated_response": response,
            "teacher_generated_answer": answer,
            "teacher_generated_correct": answer == row["gold_label"],
            "teacher_finish_reason": result.outputs[0].finish_reason,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")


if __name__ == "__main__":
    main()

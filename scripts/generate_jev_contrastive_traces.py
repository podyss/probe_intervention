#!/usr/bin/env python3
"""Generate privileged, option-by-option GSM-MC rationales and keep correct traces."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


LABELS = "ABCD"


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_response(response, option_count):
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    answer = None
    if lines:
        match = re.fullmatch(r"(?i)final\s+answer\s*:\s*\(?\s*([A-D])\s*\)?\s*[.!。]?", lines[-1])
        if match:
            answer = match.group(1).upper()
    labels_covered = [
        bool(re.search(rf"(?im)^\s*(?:[-*]\s*)?Option\s+{label}\b", response))
        for label in LABELS[:option_count]
    ]
    return answer, all(labels_covered), labels_covered


def prompt_for(row):
    options = "\n".join(f"Option {LABELS[i]}: {value}" for i, value in enumerate(row["options"]))
    gold = LABELS[row["gold_index"]]
    gold_value = row["options"][row["gold_index"]]
    return (
        f"Question: {row['question']}\nOptions:\n{options}\n\n"
        f"Teacher-only answer key: the correct option is {gold}, whose value is {gold_value}.\n\n"
        "Write a concise but complete mathematical analysis using this answer key only as privileged guidance. "
        "First solve the problem and justify the correct result from the question. Then examine every option, "
        "including the correct one, exactly once. For each option use a separate heading exactly in the form "
        "`Option A: <value>` (replace the label and value). Explain why a wrong option is wrong by identifying "
        "the specific arithmetic or interpretation error that could produce it; do not merely say it is incorrect. "
        "For the correct option, briefly connect its value to the valid calculation. Do not reveal or mention that "
        "you were given an answer key. Finish with exactly one final line in this format: `Final answer: (C)`."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/gsm-mc/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for all generated rows; the correct-only file gets a .correct.jsonl suffix")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Generate all rows when 0, otherwise deterministically take this many")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--_worker-rank", type=int)
    args = parser.parse_args()
    if args.max_examples < 0 or args.max_new_tokens < 1 or args.max_model_len < args.max_new_tokens:
        parser.error("example count must be nonnegative, token limits positive, and model length >= new-token limit")
    rows = read_rows(args.input)
    if args.max_examples and len(rows) > args.max_examples:
        import random
        indexes = random.Random(args.seed).sample(range(len(rows)), args.max_examples)
        rows = [rows[index] for index in indexes]

    if args._worker_rank is None:
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            parser.error("--gpus must contain one or more unique GPU IDs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        correct_path = args.output.with_name(args.output.stem + ".correct" + args.output.suffix)
        for path in (args.output, correct_path, args.output.with_suffix(args.output.suffix + ".manifest.json")):
            if path.exists():
                raise FileExistsError(path)
        with tempfile.TemporaryDirectory(prefix="jev-contrastive-") as temp_dir:
            shards = [Path(temp_dir) / f"rank-{rank}.jsonl" for rank in range(len(gpu_ids))]
            processes = []
            for rank, (gpu, shard) in enumerate(zip(gpu_ids, shards)):
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--model", args.model, "--input", str(args.input), "--output", str(shard),
                           "--max-examples", str(args.max_examples), "--seed", str(args.seed),
                           "--max-new-tokens", str(args.max_new_tokens), "--max-model-len", str(args.max_model_len),
                           "--gpus", args.gpus, "--_worker-rank", str(rank)]
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
                        raise RuntimeError(f"Duplicate generated row {row['id']}")
                    by_id[row["id"]] = row
        expected_ids = {row["id"] for row in rows}
        if set(by_id) != expected_ids:
            raise RuntimeError(f"Generated ID mismatch: {len(by_id)} != {len(expected_ids)}")
        ordered = [by_id[row["id"]] for row in rows]
        kept = [row for row in ordered if row["teacher_generated_correct"] and row["all_options_explained"]]
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        correct_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
        summary = {
            "model": args.model, "input": str(args.input), "output_all": str(args.output),
            "output_correct_only": str(correct_path), "seed": args.seed, "gpus": gpu_ids,
            "examples": len(ordered), "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "parsed_final_answers": sum(row["teacher_generated_answer"] is not None for row in ordered),
            "all_options_explained": sum(row["all_options_explained"] for row in ordered),
            "correct_and_all_options_explained": len(kept),
            "correct_and_all_options_explained_rate": len(kept) / len(ordered) if ordered else 0.0,
            "mean_generated_tokens": sum(row["teacher_generated_tokens"] for row in ordered) / len(ordered) if ordered else 0.0,
            "max_generated_tokens_observed": max((row["teacher_generated_tokens"] for row in ordered), default=0),
            "finish_reasons": {reason: sum(row["teacher_finish_reason"] == reason for row in ordered)
                               for reason in sorted({row["teacher_finish_reason"] for row in ordered})},
        }
        args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    gpu_count = len([item for item in args.gpus.split(",") if item.strip()])
    rows = rows[args._worker_rank::gpu_count]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    engine = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16", trust_remote_code=False,
                 max_model_len=args.max_model_len, gpu_memory_utilization=0.88,
                 enable_prefix_caching=True, disable_log_stats=True)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_for(row)}], tokenize=True,
        return_dict=False, add_generation_prompt=True, enable_thinking=True,
    ) for row in rows]
    requests = [{"prompt_token_ids": tokens} for tokens in prompts]
    outputs = engine.generate(requests, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens), use_tqdm=False)
    generated = []
    for row, result in zip(rows, outputs):
        response = result.outputs[0].text.strip()
        answer, all_options_explained, labels_covered = parse_response(response, len(row["options"]))
        generated.append({
            **row, "dataset_rationale": row.get("rationale", ""), "rationale": response,
            "teacher_generated_response": response, "teacher_generated_answer": answer,
            "teacher_generated_correct": answer == row["gold_label"],
            "all_options_explained": all_options_explained, "option_explanations_present": labels_covered,
            "teacher_generated_tokens": len(result.outputs[0].token_ids),
            "teacher_finish_reason": result.outputs[0].finish_reason,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Use dataset solutions for the gold choice and generate distractor rejections independently."""

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


def parse_verdict(response):
    if re.search(r"</think>", response, flags=re.IGNORECASE):
        response = re.split(r"</think>", response, flags=re.IGNORECASE)[-1]
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.IGNORECASE | re.DOTALL)
    response = response.strip()
    match = re.search(r"(?im)^\s*verdict\s*:\s*(CORRECT|INCORRECT)[.!]?\s*$", response)
    if not match:
        return None, response.strip()
    reasoning = (response[:match.start()] + response[match.end():]).strip()
    reasoning = re.sub(r"(?i)^reasoning\s*:\s*", "", reasoning, count=1)
    reasoning = re.sub(r"</?think>", "", reasoning).strip()
    return match.group(1).upper(), reasoning


def prompt_for(row, index):
    label = LABELS[index]
    return (
        f"Question: {row['question']}\n"
        f"Candidate choice: ({label}) {row['options'][index]}\n\n"
        "Solve the problem independently, then decide whether this candidate value correctly answers the question. "
        "Do not assume any other choices exist and do not guess an overall multiple-choice answer. "
        "Start with exactly one line: `Verdict: CORRECT` or `Verdict: INCORRECT`. "
        "On the following lines give at most two short sentences. State the computed result and whether it matches "
        "this candidate; for a mismatch, name only the relevant arithmetic or interpretation difference. Do not "
        "invent a specific mistake that is not supported by the question. Do not mention an answer key, the phrase "
        "`the answer is`, or any other choice label."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/gsm-mc/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--_worker-rank", type=int)
    args = parser.parse_args()
    if args.max_examples < 0 or args.max_new_tokens < 1 or args.max_model_len < args.max_new_tokens:
        parser.error("example count must be nonnegative and token limits must be positive and consistent")

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
        accepted_path = args.output.with_name(args.output.stem + ".accepted" + args.output.suffix)
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        for path in (args.output, accepted_path, manifest_path):
            if path.exists():
                raise FileExistsError(path)
        with tempfile.TemporaryDirectory(prefix="jev-mixed-contrastive-") as temp_dir:
            shards = [Path(temp_dir) / f"rank-{rank}.jsonl" for rank in range(len(gpu_ids))]
            processes = []
            for rank, (gpu, shard) in enumerate(zip(gpu_ids, shards)):
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--model", args.model, "--input", str(args.input), "--output", str(shard),
                           "--max-examples", str(args.max_examples), "--seed", str(args.seed),
                           "--max-new-tokens", str(args.max_new_tokens), "--max-model-len", str(args.max_model_len),
                           "--max-num-seqs", str(args.max_num_seqs),
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
        if set(by_id) != {row["id"] for row in rows}:
            raise RuntimeError(f"Generated ID mismatch: {len(by_id)} != {len(rows)}")
        ordered = [by_id[row["id"]] for row in rows]
        accepted = [row for row in ordered if row["final_answer_correct"]]
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        accepted_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
        distractor_count = sum(len(row["distractor_judgments"]) for row in ordered)
        retained_count = sum(item["retained"] for row in ordered for item in row["distractor_judgments"])
        summary = {
            "model": args.model, "input": str(args.input), "output_all": str(args.output),
            "output_final_answer_correct": str(accepted_path), "seed": args.seed, "gpus": gpu_ids,
            "examples": len(ordered), "max_new_tokens_per_distractor": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "final_answer_correct": len(accepted),
            "final_answer_correct_rate": len(accepted) / len(ordered) if ordered else 0.0,
            "distractor_judgments_parsed": sum(item["predicted_verdict"] is not None
                                                for row in ordered for item in row["distractor_judgments"]),
            "distractor_rejections_retained": retained_count,
            "distractor_positions_left_blank": distractor_count - retained_count,
            "mean_tokens_per_distractor": sum(item["tokens"] for row in ordered
                                               for item in row["distractor_judgments"]) / distractor_count
            if distractor_count else 0.0,
            "finish_reasons": {reason: sum(item["finish_reason"] == reason for row in ordered
                                           for item in row["distractor_judgments"])
                               for reason in sorted({item["finish_reason"] for row in ordered
                                                     for item in row["distractor_judgments"]})},
        }
        manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return

    gpu_count = len([item for item in args.gpus.split(",") if item.strip()])
    rows = rows[args._worker_rank::gpu_count]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    engine = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16", trust_remote_code=False,
                 max_model_len=args.max_model_len, gpu_memory_utilization=0.88,
                 max_num_seqs=args.max_num_seqs,
                 enable_prefix_caching=True, disable_log_stats=True)
    requests = []
    jobs = []
    for row in rows:
        for index in range(len(row["options"])):
            if index == row["gold_index"]:
                continue
            prompt_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_for(row, index)}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=True,
            )
            requests.append({"prompt_token_ids": prompt_tokens})
            jobs.append((row["id"], index))
    outputs = engine.generate(
        requests, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens), use_tqdm=False,
    )
    generated_by_id = {row["id"]: {} for row in rows}
    for (row_id, index), result in zip(jobs, outputs):
        response = result.outputs[0].text.strip()
        verdict, reasoning = parse_verdict(response)
        generated_by_id[row_id][index] = {
            "predicted_verdict": verdict,
            "reasoning": reasoning,
            "response": response,
            "tokens": len(result.outputs[0].token_ids),
            "finish_reason": result.outputs[0].finish_reason,
        }

    generated_rows = []
    for row in rows:
        gold_index = row["gold_index"]
        sections = []
        distractor_judgments = []
        for index, option in enumerate(row["options"]):
            if index == gold_index:
                text = (
                    f"Candidate {LABELS[index]} ({option}):\n"
                    "Verdict: CORRECT\n"
                    f"Reasoning: {row.get('rationale', '').strip()}"
                )
            else:
                item = generated_by_id[row["id"]][index]
                retained = item["predicted_verdict"] == "INCORRECT" and bool(item["reasoning"])
                distractor_judgments.append({"label": LABELS[index], "option": option, **item, "retained": retained})
                text = (
                    f"Candidate {LABELS[index]} ({option}):\n"
                    f"Verdict: {'INCORRECT' if retained else 'UNJUDGED'}\n"
                )
                if retained:
                    text += f"Reasoning: {item['reasoning']}"
                else:
                    text += "Reasoning: [blank]"
            sections.append(text)
        rationale = "\n\n".join(sections)
        # The gold solution anchors the only correct choice; false positive distractor judgments are blanked.
        final_answer = LABELS[gold_index]
        generated_rows.append({
            **row,
            "dataset_rationale": row.get("rationale", ""),
            "rationale": rationale,
            "teacher_rationale": rationale,
            "teacher_generated_answer": final_answer,
            "teacher_generated_correct": True,
            "final_answer_correct": True,
            "distractor_judgments": distractor_judgments,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated_rows), encoding="utf-8")


if __name__ == "__main__":
    main()

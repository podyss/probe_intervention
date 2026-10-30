"""Build verified complete-solution SFT rows by repairing wrong R1 attempts."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .core import grade, read_jsonl, write_jsonl
from .prompts import QUESTION


REPAIR_CONTINUATION = """Continue the student's reasoning from the retained prefix and produce a complete, correct solution.

The next step in the original attempt contains the first definite error. Do not repeat or rely on that error. Keep the retained prefix unchanged, correct its direction naturally, and continue until the answer is fully derived. Be concise while showing the essential derivation. Return only the continuation, ending with the final answer in \\boxed{{}}. Do not mention these instructions.

Problem:
{question}

Reference solution for verification and guidance:
{solution}

Retained prefix, verified to contain no definite error:
{prefix}

First erroneous step to avoid:
{error_step}
"""


def paired_error_rows(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["pair_id"]][row["label"].lower()] = row
    pairs = []
    for pair_id, pair in groups.items():
        if set(pair) != {"yes", "no"}:
            continue
        clean, incorrect = pair["no"], pair["yes"]
        prefix = clean["reasoning_prefix"]
        full_wrong_prefix = incorrect["reasoning_prefix"]
        if not full_wrong_prefix.startswith(prefix):
            raise ValueError(f"Paired prefixes are not nested for {pair_id}")
        error_step = full_wrong_prefix[len(prefix):].strip()
        if not error_step:
            raise ValueError(f"Missing first erroneous step for {pair_id}")
        pairs.append({
            "pair_id": pair_id,
            "source_id": clean["source_id"],
            "question": clean["question"],
            "solution": clean["reference_solution"],
            "prefix": prefix,
            "error_step": error_step,
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Teacher-labeled first-error pairs")
    parser.add_argument("--output", required=True, help="Verified solver SFT JSONL")
    parser.add_argument("--raw-output", required=True, help="All teacher continuations and grades")
    parser.add_argument("--model", required=True, help="Teacher model or local snapshot path")
    parser.add_argument("--limit", type=int, help="Repair only this many shortest clean prefixes")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.raw_output,
                 args.output + ".manifest.json", args.raw_output + ".manifest.json"):
        if Path(path).exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    pairs = paired_error_rows(read_jsonl(args.pairs))
    if not pairs:
        raise ValueError("No complete first-error pairs found")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        pairs = sorted(pairs, key=lambda pair: (len(pair["prefix"]), pair["pair_id"]))[:args.limit]

    from .backend import VLLMBackend

    backend = VLLMBackend(args)
    prompts = []
    for pair in pairs:
        content = REPAIR_CONTINUATION.format(**pair)
        prompts.append(backend.chat(content, thinking=False))
    generations = backend.generate(prompts, args.max_tokens, temperature=0)

    raw_rows, sft_rows = [], []
    for pair, samples in zip(pairs, generations):
        sample = samples[0]
        continuation = sample["text"].strip()
        if continuation.startswith("<think>\n"):
            continuation = continuation[len("<think>\n"):]
        full_solution = pair["prefix"].rstrip() + "\n\n" + continuation
        result = grade(full_solution, pair["solution"], sample["finish_reason"])
        accepted = result["correct"] is True
        raw_rows.append({
            **pair,
            "teacher_model": args.model,
            "continuation": continuation,
            "finish_reason": sample["finish_reason"],
            "grade": result,
            "accepted": accepted,
        })
        if accepted:
            sft_rows.append({
                "id": f"repair-distill:{pair['pair_id']}",
                "source_id": pair["source_id"],
                "task": "verified_repaired_math_solution",
                "prompt": QUESTION.format(question=pair["question"]),
                "response": full_solution,
                "final_answer": result["answer"],
                "teacher_model": args.model,
                "origin": "r1_first_error_repair",
            })

    write_jsonl(args.raw_output, raw_rows)
    write_jsonl(args.output, sft_rows)
    raw_manifest = {
        "pairs": args.pairs,
        "teacher_model": args.model,
        "arguments": vars(args),
        "input_sha256": hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
        "counts": {
            "attempted": len(raw_rows),
            "verified_correct": len(sft_rows),
            "rejected_or_unverified": len(raw_rows) - len(sft_rows),
        },
        "status": "complete",
    }
    Path(args.raw_output + ".manifest.json").write_text(
        json.dumps(raw_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    Path(args.output + ".manifest.json").write_text(
        json.dumps({
            **raw_manifest,
            "objective": "causal language modeling on complete teacher-repaired solutions; only Math-Verify-correct answers retained",
            "rows": len(sft_rows),
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(raw_manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

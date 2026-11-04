#!/usr/bin/env python3
"""Rebuild the 2k positive baseline subset and append parsed teacher failures."""

import argparse
import json
import random
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True,
                        help="Previously used base training set; sampled with the original seed")
    parser.add_argument("--failures", type=Path, required=True,
                        help="Raw Qwen3-4B traces containing incorrect and truncated answers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = read_rows(args.base)
    failure_rows = read_rows(args.failures)
    if args.base_count <= 0 or args.base_count > len(base):
        parser.error("--base-count must be positive and no larger than the base data")
    chosen = random.Random(args.seed).sample(range(len(base)), args.base_count)
    positive = [{**base[index], "negative_kl": False, "sample_role": "positive_trace"}
                for index in chosen]

    failures = [
        row for row in failure_rows
        if row.get("teacher_generated_correct") is False and row.get("teacher_generated_answer")
    ]
    negative = []
    for row in failures:
        negative.append({
            **row,
            "rationale": row.get("teacher_generated_response", row.get("rationale", "")),
            "teacher_rationale": row.get("teacher_generated_response", row.get("rationale", "")),
            "negative_kl": True,
            "sample_role": "negative_failed_trace",
        })

    rows = positive + negative
    write_rows(args.output, rows)
    summary = {
        "base": str(args.base),
        "failures": str(args.failures),
        "output": str(args.output),
        "seed": args.seed,
        "positive_base_examples": len(positive),
        "parsed_wrong_answer_examples": len(negative),
        "truncated_or_unparsed_excluded": sum(
            row.get("teacher_generated_correct") is False and not row.get("teacher_generated_answer")
            for row in failure_rows
        ),
        "total_examples": len(rows),
        "overlapping_question_ids_kept_as_paired_views": len(
            {row["id"] for row in positive} & {row["id"] for row in negative}
        ),
        "ce_target": "dataset gold_index for both positive and negative trace rows",
        "kl_sign": "positive for base rows; negative for failed-trace rows",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

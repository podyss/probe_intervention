import argparse
import hashlib
import json
import random
from pathlib import Path

from .core import read_jsonl, tokenizer_fingerprint
from .prompts import QUESTION


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Mix correctness-judgment SFT with verified self-generated math solutions"
    )
    parser.add_argument("--judge-data", required=True)
    parser.add_argument("--generated-trajectories", required=True)
    parser.add_argument("--exclude-data", nargs="+", default=[])
    parser.add_argument("--model", required=True, help="Tokenizer source used for training")
    parser.add_argument("--output", required=True)
    parser.add_argument("--correct-count", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.correct_count, args.max_length) <= 0:
        parser.error("correct-count and max-length must be positive")

    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its manifest")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    fingerprint = tokenizer_fingerprint(tokenizer)

    judge_rows = list(read_jsonl(args.judge_data))
    if not judge_rows or any(row.get("label") not in {"Correct", "Incorrect"}
                             for row in judge_rows):
        raise ValueError("judge-data must contain Correct/Incorrect instruction-response rows")
    mixed_rows = [dict(row, task="reasoning_correctness_judgment") for row in judge_rows]

    excluded_ids = set()
    for path in args.exclude_data:
        for row in read_jsonl(path):
            excluded_ids.add(str(row.get("source_id", row["id"])))

    candidates = []
    candidate_counts = {
        "rows": 0,
        "not_train": 0,
        "not_verified_correct": 0,
        "excluded_overlap": 0,
        "over_max_length": 0,
        "eligible": 0,
    }
    seen_ids = set()
    for row in read_jsonl(args.generated_trajectories):
        candidate_counts["rows"] += 1
        if row.get("split") != "train":
            candidate_counts["not_train"] += 1
            continue
        if (row.get("trajectory_finish_reason") != "stop"
                or row.get("final_grade", {}).get("correct") is not True):
            candidate_counts["not_verified_correct"] += 1
            continue
        source_id = str(row.get("source_id", row["id"]))
        if source_id in excluded_ids or source_id in seen_ids:
            candidate_counts["excluded_overlap"] += 1
            continue
        prompt = QUESTION.format(question=row["question"])
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response_ids = tokenizer.encode(row["trajectory"], add_special_tokens=False)
        if not response_ids:
            candidate_counts["not_verified_correct"] += 1
            continue
        total_length = len(prompt_ids) + len(response_ids) + 1
        if total_length > args.max_length:
            candidate_counts["over_max_length"] += 1
            continue
        candidates.append({
            "id": f"generated-correct:{row['id']}",
            "source_id": source_id,
            "label": "MathCorrect",
            "task": "verified_generated_math_solution",
            "prompt": prompt,
            "response": row["trajectory"],
        })
        seen_ids.add(source_id)
        candidate_counts["eligible"] += 1

    if len(candidates) < args.correct_count:
        raise ValueError(
            f"Found only {len(candidates)} eligible verified solutions; "
            f"need {args.correct_count}. Candidate counts: {candidate_counts}"
        )
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[:args.correct_count]
    mixed_rows.extend(selected)
    random.Random(args.seed).shuffle(mixed_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in mixed_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "sources": {
            "judge_data": args.judge_data,
            "generated_trajectories": args.generated_trajectories,
            "exclude_data": args.exclude_data,
        },
        "source_sha256": {
            args.judge_data: sha256(args.judge_data),
            args.generated_trajectories: sha256(args.generated_trajectories),
            **{path: sha256(path) for path in args.exclude_data},
        },
        "tokenizer_fingerprint": fingerprint,
        "max_length": args.max_length,
        "seed": args.seed,
        "candidate_counts": candidate_counts,
        "counts": {
            "total": len(mixed_rows),
            "reasoning_correctness_judgment": len(judge_rows),
            "verified_generated_math_solution": len(selected),
        },
        "label_mapping": {"source Yes": "Incorrect", "source No": "Correct"},
        "status": "complete",
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

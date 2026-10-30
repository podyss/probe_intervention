"""Build generative multi-task data from Yes/No probes and verified solutions.

The Yes/No probe objective is useful for an error detector, but training a
causal LM only on the final yes/no token causes catastrophic loss of its
reasoning distribution.  This converter keeps the probe task and adds
verified full solutions from the same R1 rollout pool, so the model continues
to train on the behavior used by the math evaluation.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .core import read_jsonl, tokenizer_fingerprint
from .prompts import JUDGE_PROBE, QUESTION


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Mix generative Yes/No judging with verified correct math trajectories"
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("max-length must be positive")

    output = Path(args.output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {manifest_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    fingerprint = tokenizer_fingerprint(tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")

    rows = []
    counts = Counter()
    pair_source_ids = set()
    pair_questions = set()
    for row in read_jsonl(args.pairs):
        label = row.get("label", "").strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"Invalid pair label for {row.get('id')}: {label!r}")
        verdict = row.get("verdict")
        expected_label = {"Incorrect": "yes", "Correct": "no"}.get(verdict)
        if expected_label is not None and label != expected_label:
            raise ValueError(
                f"Label/verdict mismatch for {row.get('id')}: "
                f"{label!r} does not match {verdict!r}; expected {expected_label!r}"
            )
        # Some legacy pair exports incorrectly put the rollout suffix into
        # source_id.  Group/exclude by the underlying question in either case.
        pair_source_ids.add(str(row["source_id"]).split(":rollout:", 1)[0])
        pair_questions.add(row["question"].strip())
        content = JUDGE_PROBE.format(
            question=row["question"],
            gold_answer=row["reference_solution"],
            prefix=row["reasoning_prefix"],
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response = "yes" if label == "yes" else "no"
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        total = len(prompt_ids) + len(response_ids) + 1
        if total > args.max_length:
            counts["judge_over_max_length"] += 1
            continue
        rows.append({
            "id": f"judge:{row['id']}",
            "source_id": str(row["source_id"]).split(":rollout:", 1)[0],
            "task": "reasoning_error_judgment",
            "prompt": content,
            "response": response,
        })
        counts[f"judge_{label}"] += 1

    # Use only verified, stopped, correct trajectories from questions that do
    # not also occur in the pair task.  This avoids memorizing the exact
    # question-prefix pair while retaining a broad math generation signal.
    candidates = []
    seen_sources = set()
    for row in read_jsonl(args.trajectories):
        grade = row.get("final_grade") or {}
        source_id = str(row.get("source_id", row["id"])).split(":rollout:", 1)[0]
        if row.get("split") != "train":
            counts["solution_not_train"] += 1
            continue
        if source_id in pair_source_ids or row["question"].strip() in pair_questions:
            counts["solution_pair_overlap"] += 1
            continue
        if source_id in seen_sources:
            counts["solution_duplicate_question"] += 1
            continue
        if row.get("trajectory_finish_reason") != "stop" or grade.get("correct") is not True:
            counts["solution_not_verified_correct"] += 1
            continue
        prompt = QUESTION.format(question=row["question"])
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response_ids = tokenizer.encode(row["trajectory"], add_special_tokens=False)
        total = len(prompt_ids) + len(response_ids) + 1
        if total > args.max_length:
            counts["solution_over_max_length"] += 1
            continue
        seen_sources.add(source_id)
        candidates.append({
            "id": f"solution:{row['id']}",
            "source_id": source_id,
            "task": "verified_math_solution",
            "prompt": prompt,
            "response": row["trajectory"],
        })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rng.shuffle(candidates)
    rows.extend(candidates)
    rng.shuffle(rows)
    counts["judge_total"] = counts["judge_yes"] + counts["judge_no"]
    counts["solution_total"] = len(candidates)
    counts["total"] = len(rows)

    if not rows or counts["judge_yes"] != counts["judge_no"]:
        raise ValueError(f"Expected nonempty balanced judge data, got {dict(counts)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "sources": {"pairs": args.pairs, "trajectories": args.trajectories},
        "source_sha256": {args.pairs: sha256(args.pairs), args.trajectories: sha256(args.trajectories)},
        "tokenizer_fingerprint": fingerprint,
        "max_length": args.max_length,
        "seed": args.seed,
        "counts": dict(counts),
        "objective": "causal language modeling on judge answers and verified full math solutions",
        "status": "complete",
    }
    manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Build the old explanatory judge SFT format from independently audited pairs."""

import argparse
import hashlib
import json
from pathlib import Path

from .core import read_jsonl, tokenizer_fingerprint
from .prepare_correctness_explanations import USER_PROMPT


def judged_example(row):
    label = row["label"].strip().lower()
    audit = row.get("prefix_audit")
    if label == "yes":
        if not audit or audit.get("verdict") != "error":
            raise ValueError(f"Missing audited error evidence for {row['id']}")
        quote = audit["error_quote"].strip()
        reason = audit["reason"].strip()
        if not quote or quote not in row["reasoning_prefix"] or not reason:
            raise ValueError(f"Invalid audited evidence for {row['id']}")
        response = f'Incorrect: {reason} Evidence: "{quote}"'
        output_label = "Incorrect"
    elif label == "no":
        if not audit or audit.get("verdict") != "clean":
            raise ValueError(f"Missing clean-prefix audit for {row['id']}")
        response = (
            "Correct so far: no definite mathematical or logical error has appeared "
            "in the reasoning shown. The solution may still be incomplete."
        )
        output_label = "Correct"
    else:
        raise ValueError(f"Invalid label for {row['id']}: {row['label']!r}")
    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "source_id": row["source_id"],
        "task": "reasoning_correctness_judgment",
        "label": output_label,
        "prompt": USER_PROMPT.format(
            question=row["question"], reasoning_prefix=row["reasoning_prefix"]
        ),
        "response": response,
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited-pairs", required=True)
    parser.add_argument("--mixed-data", required=True,
                        help="Correctly labeled mixed data; only solution rows are reused")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
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
    rows = [judged_example(row) for row in read_jsonl(args.audited_pairs)]
    solutions = [row for row in read_jsonl(args.mixed_data)
                 if row.get("task") == "verified_math_solution"]
    if not rows or not solutions:
        raise ValueError("Expected audited judge examples and verified solution examples")
    if sum(row["label"] == "Incorrect" for row in rows) != sum(
        row["label"] == "Correct" for row in rows
    ):
        raise ValueError("Audited judge dataset must be balanced")
    audited_rows = list(read_jsonl(args.audited_pairs))
    judge_source_ids = {row["source_id"] for row in audited_rows}
    judge_questions = {row["question"].strip() for row in audited_rows}
    solution_source_ids = {row["source_id"] for row in solutions}
    solution_questions = {
        row["prompt"].split("\n\nReason step by step", 1)[0].strip()
        for row in solutions
    }
    if judge_source_ids & solution_source_ids or judge_questions & solution_questions:
        raise ValueError("Judge and solution tasks contain overlapping questions")

    def fits_length(row):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False,
        )
        completion_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        return len(prompt_ids) + len(completion_ids) + 1 <= args.max_length

    fit_pair_ids = {
        row["pair_id"] for row in rows
        if all(fits_length(member) for member in rows if member["pair_id"] == row["pair_id"])
    }
    all_pair_ids = {row["pair_id"] for row in rows}
    rows = [row for row in rows if row["pair_id"] in fit_pair_ids]
    oversized_pairs = len(all_pair_ids - fit_pair_ids)
    for row in solutions:
        if not fits_length(row):
            raise ValueError(f"{row['id']} exceeds --max-length={args.max_length}")

    import random

    combined = rows + solutions
    random.Random(42).shuffle(combined)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "sources": {"audited_pairs": args.audited_pairs, "mixed_data": args.mixed_data},
        "source_sha256": {args.audited_pairs: sha256(args.audited_pairs),
                          args.mixed_data: sha256(args.mixed_data)},
        "tokenizer_fingerprint": fingerprint,
        "max_length": args.max_length,
        "counts": {
            "judge_correct": sum(row["label"] == "Correct" for row in rows),
            "judge_incorrect": sum(row["label"] == "Incorrect" for row in rows),
            "judge_oversized_pairs": oversized_pairs,
            "verified_math_solution": len(solutions),
            "total": len(combined),
        },
        "prompt_format": "same as prepare_correctness_explanations; no reference answer in judge prompt",
        "objective": "explanatory correctness judgment plus verified full math solutions",
        "status": "complete",
    }
    manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

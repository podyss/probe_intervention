import argparse
import json
from pathlib import Path

from .core import read_jsonl


USER_PROMPT = """You are evaluating a mathematical solution in progress. Decide whether the reasoning shown so far is correct. Judge only claims made so far; an incomplete but valid derivation is still correct so far. If it contains a definite mathematical or logical error, identify the first error and explain why it is wrong. Do not invent errors or rely on reasoning that is not shown.

Problem:
{question}

Reasoning so far:
{reasoning_prefix}

Answer in this format:
Correct so far: briefly state that no definite error has appeared (the work may be incomplete).
or
Incorrect: identify the first error and briefly explain why it is wrong."""


def make_response(row):
    label = row["label"].strip().lower()
    if label == "yes":
        rationale = row.get("teacher_rationale", "").strip()
        if not rationale:
            raise ValueError(f"Missing teacher_rationale for {row['id']}")
        return f"Incorrect: {rationale}"
    if label == "no":
        return "Correct so far: no definite mathematical or logical error has appeared in the reasoning shown. The solution may still be incomplete."
    raise ValueError(f"Invalid label for {row['id']}: {row['label']!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Build correctness-judgment SFT data with explanations for errors"
    )
    parser.add_argument("--input-dir", default="data/teacher-first-error-short-200")
    parser.add_argument("--output-dir", default="data/teacher-first-error-reasoning-200")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty data output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "test"):
        output_rows = []
        for row in read_jsonl(input_dir / f"{split}.jsonl"):
            label = row["label"].strip().lower()
            output_rows.append({
                "id": row["id"],
                "source_id": row["source_id"],
                "label": "Incorrect" if label == "yes" else "Correct",
                "prompt": USER_PROMPT.format(
                    question=row["question"],
                    reasoning_prefix=row["reasoning_prefix"],
                ),
                "response": make_response(row),
            })
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in output_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = {
            "examples": len(output_rows),
            "incorrect": sum(row["label"] == "Incorrect" for row in output_rows),
            "correct": sum(row["label"] == "Correct" for row in output_rows),
        }

    manifest = {
        "source": str(input_dir),
        "task": "judge whether shown reasoning is correct; identify and explain the first error when incorrect",
        "split_policy": "copied from source train/test; source-problem isolation is preserved",
        "counts": counts,
        "label_mapping": {
            "source Yes": "Incorrect, with teacher_rationale",
            "source No": "Correct so far; the reasoning may be incomplete",
        },
        "note": "teacher_error_quote is not included because its quoted text is absent from the labeled reasoning prefixes.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

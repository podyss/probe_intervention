import argparse
import json
from pathlib import Path

from .core import read_jsonl


USER_PROMPT = """You are checking a mathematical solution in progress. Decide whether the reasoning shown so far contains a definite, unresolved mathematical or logical error. Do not treat an incomplete solution as an error, and do not use information that is not in the reasoning shown.

Problem:
{question}

Reasoning so far:
{reasoning_prefix}

If there is a definite unresolved error, start with `Yes.` and briefly explain the specific incorrect claim and why it is wrong. If there is no definite unresolved error, answer exactly `No.`"""


def main():
    parser = argparse.ArgumentParser(
        description="Build instruction-response SFT data for first-error detection and explanation"
    )
    parser.add_argument("--input-dir", default="data/teacher-first-error-short-200")
    parser.add_argument("--output-dir", required=True)
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
            if label == "yes":
                rationale = row.get("teacher_rationale", "").strip()
                if not rationale:
                    raise ValueError(f"Missing teacher_rationale for {row['id']}")
                response = f"Yes. {rationale}"
            elif label == "no":
                response = "No."
            else:
                raise ValueError(f"Invalid label for {row['id']}: {row['label']!r}")

            output_rows.append({
                "id": row["id"],
                "source_id": row["source_id"],
                "label": "Yes" if label == "yes" else "No",
                "prompt": USER_PROMPT.format(
                    question=row["question"],
                    reasoning_prefix=row["reasoning_prefix"],
                ),
                "response": response,
            })
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in output_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = {
            "examples": len(output_rows),
            "yes": sum(row["label"] == "Yes" for row in output_rows),
            "no": sum(row["label"] == "No" for row in output_rows),
        }

    manifest = {
        "source": str(input_dir),
        "task": "first-error detection with a brief explanation when an error is present",
        "split_policy": "copied from source train/test; source-problem isolation is preserved",
        "counts": counts,
        "note": "Yes responses use teacher_rationale; No responses do not expose error annotations.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

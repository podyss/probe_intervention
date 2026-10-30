"""Extract the verified full-solution rows from a mixed SFT dataset.

This is the matched ablation for a mixed correctness-judgment + solution
run: it keeps the exact same verified solution rows and removes only the
error-judgment task.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .core import read_jsonl


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract verified full math solutions for a matched SFT ablation"
    )
    parser.add_argument("--input", required=True, help="Mixed SFT JSONL")
    parser.add_argument("--output", required=True, help="Correct-only JSONL")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    if args.count is not None and args.count <= 0:
        parser.error("count must be positive")

    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {manifest}")

    rows = [
        row for row in read_jsonl(args.input)
        if row.get("task") == "verified_generated_math_solution"
    ]
    if args.count is not None:
        if len(rows) < args.count:
            raise ValueError(f"Found {len(rows)} solution rows, need {args.count}")
        rows = rows[: args.count]
    if not rows:
        raise ValueError("No verified solution rows found")
    if any(not row.get("prompt") or not row.get("response") for row in rows):
        raise ValueError("Every solution row must contain nonempty prompt and response")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "source": args.input,
        "source_sha256": sha256(args.input),
        "counts": {"total": len(rows), "verified_generated_math_solution": len(rows)},
        "objective": "causal language modeling on verified full math solutions only",
        "status": "complete",
    }
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

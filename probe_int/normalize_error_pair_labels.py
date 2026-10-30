"""Normalize pair labels to the established Yes=error, No=clean convention."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .core import read_jsonl, write_jsonl


def normalize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
    normalized = []
    changed = 0
    for pair_id, pair in groups.items():
        if len(pair) != 2 or Counter(row.get("verdict") for row in pair) != {
            "Correct": 1, "Incorrect": 1
        }:
            raise ValueError(f"Expected one Correct and one Incorrect row in {pair_id}")
        for row in pair:
            expected = "Yes" if row["verdict"] == "Incorrect" else "No"
            if row.get("label", "").strip().lower() != expected.lower():
                changed += 1
            normalized.append(dict(row, label=expected))
    return normalized, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {manifest}")
    source_rows = list(read_jsonl(args.input))
    rows, changed = normalize_rows(source_rows)
    count = write_jsonl(output, rows)
    metadata = {
        "source": args.input,
        "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "examples": count,
        "pairs": count // 2,
        "labels_corrected": changed,
        "label_mapping": {"Incorrect": "Yes", "Correct": "No"},
        "status": "complete",
    }
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

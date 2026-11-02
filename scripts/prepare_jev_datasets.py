#!/usr/bin/env python3
"""Download and normalize MathQA and CompMath-MCQ for Jev-style evaluation."""

import argparse
import ast
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset


MATHQA_SOURCE = "allenai/math_qa"
COMPMATH_SOURCE = "biancaraimondi/CompMath-MCQ"
OPTION_MARKER = re.compile(r"(?:^|,\s*)([a-e])\s*\)\s*")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_mathqa_options(raw):
    if isinstance(raw, list):
        parsed = []
        for item in raw:
            match = re.match(r"^\s*([a-e])\s*\)\s*(.*?)\s*$", str(item))
            if not match:
                raise ValueError(f"Could not parse MathQA option: {item!r}")
            parsed.append((match.group(1), match.group(2)))
        if [label for label, _ in parsed] != list("abcde"):
            raise ValueError(f"Unexpected MathQA option labels: {raw!r}")
        return [option for _, option in parsed]
    if isinstance(raw, str) and raw.lstrip().startswith("["):
        try:
            return parse_mathqa_options(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            pass
    matches = list(OPTION_MARKER.finditer(raw))
    if len(matches) < 2:
        raise ValueError(f"Could not parse MathQA options: {raw!r}")
    options = []
    labels = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        labels.append(match.group(1))
        options.append(raw[match.end():end].strip(" ,"))
    if labels != list("abcde") or any(not option for option in options):
        raise ValueError(f"Unexpected MathQA option labels/content: {raw!r}")
    return options


def mathqa_row(row, split, index):
    options = parse_mathqa_options(row["options"])
    answer = str(row["correct"]).strip().lower()
    if answer not in "abcde" or len(answer) != 1:
        raise ValueError(f"Unexpected MathQA gold label: {answer!r}")
    return {
        "id": f"mathqa:{split}:{index}",
        "dataset": "mathqa",
        "split": split,
        "question": row["Problem"].strip(),
        "options": options,
        "gold_index": ord(answer) - ord("a"),
        "gold_label": answer.upper(),
        "rationale": row["Rationale"].strip(),
        "category": str(row.get("category", "unknown")).strip(),
        "linear_formula": row.get("linear_formula"),
    }


def compmath_row(row, index):
    options = [str(option).strip() for option in row["options"]]
    gold = int(row["correct_label"])
    if len(options) < 2 or gold < 0 or gold >= len(options):
        raise ValueError(f"Invalid CompMath-MCQ item at row {index}")
    return {
        "id": f"compmath-mcq:test:{index}",
        "dataset": "compmath-mcq",
        "split": "test",
        "question": str(row["question"]).strip(),
        "options": options,
        "gold_index": gold,
        "gold_label": chr(ord("A") + gold),
        "category": str(row.get("subtopic", "unknown")).strip(),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return {"path": str(path), "rows": len(rows), "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be between 0 and 1")

    mathqa = load_dataset(MATHQA_SOURCE)
    train_rows = [mathqa_row(row, "train", i) for i, row in enumerate(mathqa["train"])]
    seen = set()
    unique_train = []
    for row in train_rows:
        key = normalized(row["question"])
        if key not in seen:
            seen.add(key)
            unique_train.append(row)

    groups = [row["category"] or "unknown" for row in unique_train]
    grouped_indices = {}
    for index, group in enumerate(groups):
        grouped_indices.setdefault(group, []).append(index)
    shuffled_groups = list(grouped_indices)
    random.Random(args.seed).shuffle(shuffled_groups)
    target_dev_rows = round(len(unique_train) * args.dev_fraction)
    dev_groups = set()
    dev_rows = 0
    for group in shuffled_groups:
        if dev_rows >= target_dev_rows:
            break
        dev_groups.add(group)
        dev_rows += len(grouped_indices[group])
    dev_indices = [i for i, group in enumerate(groups) if group in dev_groups]
    train_indices = [i for i, group in enumerate(groups) if group not in dev_groups]
    mathqa_train = [dict(unique_train[i], split="train") for i in train_indices]
    mathqa_dev = [dict(unique_train[i], split="dev") for i in dev_indices]
    mathqa_validation = [mathqa_row(row, "validation", i) for i, row in enumerate(mathqa["validation"])]
    mathqa_test = [mathqa_row(row, "test", i) for i, row in enumerate(mathqa["test"])]

    compmath = load_dataset(COMPMATH_SOURCE, split="test")
    compmath_test = [compmath_row(row, i) for i, row in enumerate(compmath)]

    outputs = []
    mathqa_dir = args.output_dir / "jev-mathqa"
    for name, rows in (("train", mathqa_train), ("dev", mathqa_dev),
                       ("validation", mathqa_validation), ("test", mathqa_test)):
        outputs.append(write_jsonl(mathqa_dir / f"{name}.jsonl", rows))
    outputs.append(write_jsonl(args.output_dir / "compmath-mcq" / "test.jsonl", compmath_test))
    manifest = {
        "sources": {"mathqa": MATHQA_SOURCE, "compmath-mcq": COMPMATH_SOURCE},
        "deduplication": "MathQA train exact normalized Problem text, first occurrence kept",
        "dev_split": {"method": "seeded whole-category split", "group": "MathQA category",
                      "fraction": args.dev_fraction, "seed": args.seed},
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "jev-datasets-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert the AMC 2023 source parquet to the project's normalized JSONL."""

import json
import math
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "test-00000-of-00001.parquet"
OUTPUT = ROOT / "test.jsonl"
EXPECTED_ROWS = 40


def answer_text(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid numeric answer: {value!r}")
    if value.is_integer() if isinstance(value, float) else True:
        return str(int(value))
    return format(value, ".15g")


def main():
    table = pq.read_table(SOURCE)
    expected_columns = {"id", "answer", "url", "question"}
    if set(table.column_names) != expected_columns:
        raise RuntimeError(f"Unexpected source columns: {table.column_names}")
    rows = table.to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")

    seen_ids = set()
    seen_questions = set()
    with OUTPUT.open("w", encoding="utf-8") as output:
        for row in rows:
            identifier = row["id"]
            question = row["question"]
            url = row["url"]
            if identifier in seen_ids or question in seen_questions:
                raise RuntimeError(f"Duplicate source id or question: {identifier}")
            if not isinstance(question, str) or not question.strip():
                raise RuntimeError(f"Empty question at source id {identifier}")
            if not isinstance(url, str) or not url.strip():
                raise RuntimeError(f"Missing source URL at source id {identifier}")
            seen_ids.add(identifier)
            seen_questions.add(question)
            normalized = {
                "id": f"amc23:{identifier}",
                "split": "test",
                "question": question,
                "solution": answer_text(row["answer"]),
                "url": url,
            }
            output.write(json.dumps(normalized, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

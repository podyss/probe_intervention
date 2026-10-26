#!/usr/bin/env python3
"""Convert the DeepMath-103K parquet shards to normalized JSONL files."""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "data"
REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
EXPECTED_ROWS = 103_022
SEED = 42
TEST_SIZE = 500
TRAIN_SIZE = 1_000


def write_row(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    shards = sorted(SOURCE.glob("train-*-of-00010.parquet"))
    if len(shards) != 10:
        raise RuntimeError(f"Expected 10 parquet shards in {SOURCE}, found {len(shards)}")

    # Source row indices remain stable even when rows without gold answers are excluded.
    eligible_indices = []
    source_index = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["question", "final_answer"], batch_size=1024):
            for item in batch.to_pylist():
                if (isinstance(item["question"], str) and item["question"].strip()
                        and isinstance(item["final_answer"], str) and item["final_answer"].strip()):
                    eligible_indices.append(source_index)
                source_index += 1

    selected = random.Random(SEED).sample(eligible_indices, TEST_SIZE + TRAIN_SIZE)
    test_indices = set(selected[:TEST_SIZE])
    train_indices = set(selected[TEST_SIZE:])
    counts = {"all": 0, "train": 0, "test": 0}
    selected_ids = {"train": [], "test": []}

    with (ROOT / "all.jsonl").open("w", encoding="utf-8") as all_file, \
            (ROOT / "train.jsonl").open("w", encoding="utf-8") as train_file, \
            (ROOT / "test.jsonl").open("w", encoding="utf-8") as test_file:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            required = {"question", "final_answer", "difficulty", "topic"}
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise RuntimeError(f"{shard.name} is missing columns: {sorted(missing)}")

            for batch in parquet.iter_batches(columns=sorted(required), batch_size=256):
                for item in batch.to_pylist():
                    index = counts["all"]
                    question = item["question"]
                    answer = item["final_answer"]
                    if not isinstance(question, str) or not question.strip():
                        raise RuntimeError(f"Empty question at source row {index}")

                    identifier = f"deepmath-103k:{index}"
                    normalized = {
                        "id": identifier,
                        "question": question,
                        "solution": answer,
                        "difficulty": item["difficulty"],
                        "topic": item["topic"],
                        "source_split": "train",
                    }
                    write_row(all_file, normalized)
                    counts["all"] += 1

                    if index in train_indices:
                        write_row(train_file, {**normalized, "split": "train"})
                        selected_ids["train"].append(identifier)
                        counts["train"] += 1
                    elif index in test_indices:
                        write_row(test_file, {**normalized, "split": "test"})
                        selected_ids["test"].append(identifier)
                        counts["test"] += 1

    if counts != {"all": EXPECTED_ROWS, "train": TRAIN_SIZE, "test": TEST_SIZE}:
        raise RuntimeError(f"Unexpected row counts: {counts}")

    manifest = {
        "dataset": "zwhe99/DeepMath-103K",
        "revision": REVISION,
        "source": "https://hf-mirror.com/datasets/zwhe99/DeepMath-103K",
        "license": "MIT",
        "source_rows": EXPECTED_ROWS,
        "rows_without_reference_answer": EXPECTED_ROWS - len(eligible_indices),
        "normalized_fields": {
            "question": "question",
            "solution": "final_answer",
            "difficulty": "difficulty",
            "topic": "topic",
            "id": "stable source row index",
        },
        "sampling": {
            "method": "uniform sample without replacement from rows with nonempty questions and final answers",
            "seed": SEED,
            "test_rows": TEST_SIZE,
            "train_rows": TRAIN_SIZE,
            "subsets_disjoint": True,
        },
        "outputs": {
            "all.jsonl": EXPECTED_ROWS,
            "train.jsonl": TRAIN_SIZE,
            "test.jsonl": TEST_SIZE,
        },
        "selected_ids": selected_ids,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()

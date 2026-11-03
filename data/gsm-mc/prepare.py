#!/usr/bin/env python3
"""Convert the published GSM-MC JSONL files to the project's Jev schema."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
GSM8K_SOURCE = ROOT.parent / "gsm8k-full"
LABELS = "ABCD"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(question):
    return " ".join(unicodedata.normalize("NFKC", str(question)).casefold().split())


def normalize_answer(answer):
    return re.sub(r"[,$\s]", "", str(answer)).rstrip(".")


def load_gsm8k_rationales(split):
    path = GSM8K_SOURCE / f"{split}.parquet"
    rows = pq.read_table(path).to_pylist()
    by_question = {}
    for row in rows:
        info = row["extra_info"]
        key = normalize_question(info["question"])
        if key in by_question:
            raise ValueError(f"Duplicate GSM8K question in {path}: {info['question']!r}")
        answer = str(info["answer"])
        if "####" not in answer:
            raise ValueError(f"GSM8K answer lacks #### separator: {info['question']!r}")
        rationale, final_answer = answer.rsplit("####", 1)
        if not rationale.strip() or not final_answer.strip():
            raise ValueError(f"Incomplete GSM8K answer: {info['question']!r}")
        by_question[key] = (rationale.strip(), final_answer.strip())
    return by_question


def read_source(split):
    source_path = SOURCE / f"{split}.jsonl"
    rationales = load_gsm8k_rationales(split)
    rows = []
    with source_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            answer = str(row["Answer"]).strip().upper()
            question = str(row["Question"]).strip()
            options = [str(row[label]).strip() for label in LABELS]
            if answer not in LABELS or not question or any(not option for option in options):
                raise ValueError(f"Invalid {split} row {line_number}: {row!r}")
            match = rationales.pop(normalize_question(question), None)
            if match is None:
                raise ValueError(f"No GSM8K rationale matched {split} row {line_number}: {question!r}")
            rationale, final_answer = match
            if normalize_answer(final_answer) != normalize_answer(options[LABELS.index(answer)]):
                raise ValueError(
                    f"GSM-MC label conflicts with GSM8K answer in {split} row {line_number}: "
                    f"{options[LABELS.index(answer)]!r} != {final_answer!r}"
                )
            rows.append({
                "id": f"gsm-mc:{split}:{len(rows)}",
                "dataset": "gsm-mc",
                "split": split,
                "question": question,
                "options": options,
                "gold_index": LABELS.index(answer),
                "gold_label": answer,
                "rationale": rationale,
            })
    return rows, len(rationales)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(rows), "sha256": sha256(path)}


def main():
    manifest = {
        "source": "https://huggingface.co/datasets/guipenedo/gsm8k-mc",
        "source_revision": "fca1f0a07ad5b0c21fd67fcaec8532a37d9063e2",
        "upstream_paper": "https://arxiv.org/abs/2405.11966",
        "license": "No dataset card/license declared by the Hugging Face dataset; verify terms before redistribution.",
        "rationale_source": "../gsm8k-full/{train,test}.parquet, matched by normalized exact question text",
        "rationale_mapping": "extra_info.answer text before ####; final answer after #### is checked against the labeled option and omitted from rationale",
        "schema": {
            "Question": "question",
            "A-D": "options, preserving source order",
            "Answer": "gold_label and zero-based gold_index",
            "rationale": "not provided by source; omitted",
        },
        "files": {
            f"source/{split}.jsonl": {"sha256": sha256(SOURCE / f"{split}.jsonl")}
            for split in ("train", "test")
        },
        "rationale_files": {
            f"../gsm8k-full/{split}.parquet": {"sha256": sha256(GSM8K_SOURCE / f"{split}.parquet")}
            for split in ("train", "test")
        },
        "normalized": {},
        "unmatched_gsm8k_rows": {},
    }
    for split in ("train", "test"):
        rows, unmatched = read_source(split)
        manifest["normalized"][f"{split}.jsonl"] = write_jsonl(ROOT / f"{split}.jsonl", rows)
        manifest["unmatched_gsm8k_rows"][split] = unmatched
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["normalized"], indent=2))


if __name__ == "__main__":
    main()

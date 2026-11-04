#!/usr/bin/env python3
"""Blank only generated option judgments that reached the token limit."""

import argparse
import json
from pathlib import Path


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = read_rows(args.input)
    capped = 0
    affected_rows = 0
    for row in rows:
        gold_index = row["gold_index"]
        judgments = {item["label"]: item for item in row["distractor_judgments"]}
        sections = []
        affected = False
        for index, option in enumerate(row["options"]):
            label = LABELS[index]
            if index == gold_index:
                section = (
                    f"Candidate {label} ({option}):\n"
                    "Verdict: CORRECT\n"
                    f"Reasoning: {row.get('dataset_rationale', '').strip()}"
                )
            else:
                item = judgments[label]
                hit_cap = item["finish_reason"] == "length"
                if hit_cap:
                    capped += 1
                    affected = True
                    item["retained"] = False
                    item["blanked_due_to_length_cap"] = True
                retained = item["retained"] and not hit_cap
                item["retained"] = retained
                section = (
                    f"Candidate {label} ({option}):\n"
                    f"Verdict: {'INCORRECT' if retained else 'UNJUDGED'}\n"
                )
                section += f"Reasoning: {item['reasoning']}" if retained else "Reasoning: [blank]"
            sections.append(section)
        if affected:
            affected_rows += 1
        row["rationale"] = "\n\n".join(sections)
        row["teacher_rationale"] = row["rationale"]

    write_rows(args.output, rows)
    accepted_path = args.output.with_name(args.output.stem + ".accepted" + args.output.suffix)
    accepted = [row for row in rows if row.get("final_answer_correct")]
    write_rows(accepted_path, accepted)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "accepted_output": str(accepted_path),
        "examples": len(rows),
        "final_answer_correct": len(accepted),
        "capped_option_judgments_blanked": capped,
        "examples_with_at_least_one_capped_option": affected_rows,
        "retained_distractor_judgments": sum(
            item["retained"] for row in rows for item in row["distractor_judgments"]
        ),
        "blank_distractor_positions": sum(
            not item["retained"] for row in rows for item in row["distractor_judgments"]
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

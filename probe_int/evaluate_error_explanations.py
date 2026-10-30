import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .backend import VLLMBackend
from .core import read_jsonl
from .prepare_correctness_explanations import USER_PROMPT


LABEL_PATTERN = re.compile(r"\b(Incorrect|Correct so far)\s*:", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate correctness-explanation SFT as a binary Yes/No judge"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if min(args.max_model_len, args.max_new_tokens,
           args.tensor_parallel_size, args.batch_size) <= 0:
        parser.error("lengths, tensor-parallel-size, and batch-size must be positive")

    rows = list(read_jsonl(args.input))
    if not rows or any(row.get("label", "").lower() not in {"yes", "no"} for row in rows):
        raise ValueError("Input must be a non-empty Yes/No labeled JSONL")

    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its manifest")

    backend = VLLMBackend(args)
    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = [backend.chat(USER_PROMPT.format(
            question=row["question"],
            reasoning_prefix=row["reasoning_prefix"],
        ), thinking=False) for row in batch]
        generations = backend.generate(prompts, max_tokens=args.max_new_tokens)
        for row, samples in zip(batch, generations):
            raw = samples[0]["text"] if samples else ""
            match = LABEL_PATTERN.search(raw)
            if match:
                prediction = "yes" if match.group(1).lower() == "incorrect" else "no"
            else:
                prediction = None
            gold = row["label"].lower()
            results.append({
                "id": row["id"],
                "source_id": row["source_id"],
                "gold": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "response": raw,
                "model": args.model,
            })

    by_label = {}
    for label in ("yes", "no"):
        class_rows = [row for row in results if row["gold"] == label]
        by_label[label] = {
            "total": len(class_rows),
            "correct": sum(row["correct"] for row in class_rows),
            "accuracy": sum(row["correct"] for row in class_rows) / len(class_rows),
        }
    confusion = Counter((row["gold"], row["prediction"]) for row in results)
    summary = {
        "rows": len(results),
        "correct": sum(row["correct"] for row in results),
        "unparsed": sum(row["prediction"] is None for row in results),
        "accuracy": sum(row["correct"] for row in results) / len(results),
        "by_gold_label": by_label,
        "confusion_matrix": {
            "gold_yes_pred_yes": confusion[("yes", "yes")],
            "gold_yes_pred_no": confusion[("yes", "no")],
            "gold_yes_unparsed": confusion[("yes", None)],
            "gold_no_pred_yes": confusion[("no", "yes")],
            "gold_no_pred_no": confusion[("no", "no")],
            "gold_no_unparsed": confusion[("no", None)],
        },
        "output": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "arguments": vars(args),
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "tokenizer_fingerprint": backend.fingerprint,
        "label_mapping": {"Incorrect": "yes", "Correct so far": "no"},
        "summary": summary,
        "status": "complete",
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

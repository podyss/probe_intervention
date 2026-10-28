import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .core import reference_answer, read_jsonl
from .prompts import JUDGE_PROBE


def evaluate(args, rows):
    from .backend import VLLMBackend

    backend = VLLMBackend(args)
    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = [backend.chat(JUDGE_PROBE.format(
            question=row["question"],
            gold_answer=reference_answer(row["reference_solution"]),
            prefix=row["reasoning_prefix"],
        ), thinking=False) for row in batch]
        predictions = backend.score_yes_no(prompts)
        for row, prediction in zip(batch, predictions):
            gold = row["label"].lower()
            predicted = prediction["answer"]
            results.append({
                "id": row["id"],
                "source_id": row["source_id"],
                "gold": gold,
                "prediction": predicted,
                "correct": predicted == gold,
                "p_yes": prediction["p_correct"],
                "yes_logprob": prediction["yes_logprob"],
                "no_logprob": prediction["no_logprob"],
                "model": args.model,
                "teacher_first_error_step": row["teacher_first_error_step"],
            })
    return results, backend.fingerprint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Yes/No reasoning-prefix probe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if min(args.max_model_len, args.tensor_parallel_size, args.batch_size) <= 0:
        parser.error("max-model-len, tensor-parallel-size, and batch-size must be positive")

    rows = list(read_jsonl(args.input))
    if not rows or any(row.get("label", "").lower() not in {"yes", "no"} for row in rows):
        raise ValueError("Input must be a non-empty Yes/No labeled JSONL")
    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its manifest")

    results, fingerprint = evaluate(args, rows)
    confusion = Counter((row["gold"], row["prediction"]) for row in results)
    by_class = {}
    for label in ("yes", "no"):
        class_rows = [row for row in results if row["gold"] == label]
        by_class[label] = {
            "total": len(class_rows),
            "correct": sum(row["correct"] for row in class_rows),
            "accuracy": sum(row["correct"] for row in class_rows) / len(class_rows),
        }
    summary = {
        "rows": len(results),
        "correct": sum(row["correct"] for row in results),
        "accuracy": sum(row["correct"] for row in results) / len(results),
        "by_gold_label": by_class,
        "confusion_matrix": {
            "gold_yes_pred_yes": confusion[("yes", "yes")],
            "gold_yes_pred_no": confusion[("yes", "no")],
            "gold_no_pred_yes": confusion[("no", "yes")],
            "gold_no_pred_no": confusion[("no", "no")],
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
        "tokenizer_fingerprint": fingerprint,
        "summary": summary,
        "status": "complete",
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

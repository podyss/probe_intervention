import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .core import grade, read_jsonl, write_jsonl
from .prompts import QUESTION


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a model on prepared GSM8K JSONL using vLLM")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Prepared JSONL; split must be validation or test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if min(args.max_model_len, args.max_tokens, args.tensor_parallel_size) <= 0:
        raise ValueError("Token limits and tensor parallel size must be positive")
    if args.temperature < 0 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("Temperature must be nonnegative and GPU utilization in (0, 1]")

    source = []
    for row in read_jsonl(args.input):
        if row.get("split") not in {"validation", "test"}:
            raise ValueError("Evaluation accepts validation/test rows only; prepare data with --split test")
        source.append(row)
        if args.limit is not None and len(source) >= args.limit:
            break
    if not source:
        raise ValueError("No evaluation rows found")

    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its manifest")
    digest = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    versions = {}
    for package in ["probe-intervention", "vllm", "transformers"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"arguments": vars(args), "input_sha256": digest, "versions": versions,
                "status": "running"}
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    from .backend import VLLMBackend

    backend = VLLMBackend(args)
    prompts = [backend.chat(QUESTION.format(question=row["question"]), thinking=True) for row in source]
    generations = backend.generate(prompts, args.max_tokens, args.temperature)
    results = []
    for row, samples in zip(source, generations):
        sample = samples[0]
        results.append({
            "id": row["id"], "split": row["split"], "question": row["question"],
            "solution": row["solution"], "model": args.model,
            "tokenizer_fingerprint": backend.fingerprint,
            "generation": sample,
            "grade": grade(sample["text"], row["solution"], sample["finish_reason"]),
        })
    count = write_jsonl(output, results)
    metadata.update(status="complete", rows=count)
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    correct = sum(row["grade"]["correct"] is True for row in results)
    graded = sum(row["grade"]["correct"] is not None for row in results)
    print(json.dumps({"rows": count, "graded": graded, "correct": correct,
                      "accuracy_on_graded": correct / graded if graded else None,
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

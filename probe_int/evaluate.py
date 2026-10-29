import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from collections import Counter
from pathlib import Path

from .core import grade, read_jsonl, write_jsonl
from .prompts import QUESTION


def evaluate_rows(args, source):
    from .backend import VLLMBackend

    backend = VLLMBackend(args)
    prompts = [backend.chat(QUESTION.format(question=row["question"]), thinking=True)
               for row in source]
    generations = backend.generate(prompts, args.max_tokens, args.temperature,
                                   top_p=args.top_p)
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
    return results


def evaluate_dp_rank(rank, args, rows, shard_output, gpu_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    for name in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                 "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
        os.environ.pop(name, None)
    print(f"DP rank {rank}: evaluating {len(rows)} rows", flush=True)
    results = evaluate_rows(args, rows)
    write_jsonl(shard_output, results)
    print(f"DP rank {rank}: completed {len(results)} rows", flush=True)


def visible_gpu_ids(required_count):
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        devices = [device.strip() for device in configured.split(",") if device.strip()]
    else:
        import torch

        devices = [str(index) for index in range(torch.cuda.device_count())]
    if len(devices) < required_count:
        raise ValueError(
            f"Need {required_count} visible GPUs for data-parallel-size * "
            f"tensor-parallel-size, found {len(devices)}: {devices}"
        )
    return devices


def merge_dp_shards(source, shard_outputs, dp_size):
    expected_ids = [row["id"] for row in source]
    expected_set = set(expected_ids)
    by_id = {}
    for rank, path in enumerate(shard_outputs):
        start = rank * len(source) // dp_size
        end = (rank + 1) * len(source) // dp_size
        expected_rank_ids = {row["id"] for row in source[start:end]}
        shard_rows = list(read_jsonl(path))
        shard_ids = [row["id"] for row in shard_rows]
        if len(shard_ids) != len(set(shard_ids)) or set(shard_ids) != expected_rank_ids:
            raise RuntimeError(
                f"DP shard {rank} is incomplete or contains unexpected rows: "
                f"expected {len(expected_rank_ids)}, found {len(shard_ids)}"
            )
        by_id.update((row["id"], row) for row in shard_rows)
    if set(by_id) != expected_set or len(by_id) != len(expected_ids):
        raise RuntimeError("Data-parallel outputs do not match the selected input rows")
    return [by_id[row_id] for row_id in expected_ids]


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a model on prepared math JSONL using vLLM")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Prepared JSONL; split must be validation or test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable vLLM's prompt progress bar")
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if min(args.max_model_len, args.max_tokens, args.tensor_parallel_size,
           args.data_parallel_size) <= 0:
        raise ValueError("Token limits and tensor parallel size must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("Temperature must be nonnegative; top-p and GPU utilization must be in (0, 1]")

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

    if args.data_parallel_size == 1:
        results = evaluate_rows(args, source)
    else:
        gpu_ids = visible_gpu_ids(args.data_parallel_size * args.tensor_parallel_size)
        shard_outputs = [output.with_name(f"{output.name}.dp{rank}")
                         for rank in range(args.data_parallel_size)]
        if any(path.exists() for path in shard_outputs):
            raise FileExistsError("Refusing to overwrite a DP shard output")
        context = multiprocessing.get_context("spawn")
        rows_by_rank = [
            source[rank * len(source) // args.data_parallel_size:
                   (rank + 1) * len(source) // args.data_parallel_size]
            for rank in range(args.data_parallel_size)
        ]
        workers = [context.Process(
            target=evaluate_dp_rank,
            args=(rank, args, rows_by_rank[rank], shard_outputs[rank],
                  gpu_ids[rank * args.tensor_parallel_size:
                          (rank + 1) * args.tensor_parallel_size]),
        ) for rank in range(args.data_parallel_size)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        failed_workers = [rank for rank, worker in enumerate(workers)
                          if worker.exitcode != 0]
        if any(not path.exists() for path in shard_outputs):
            raise RuntimeError(
                "Data-parallel worker failed before writing all shards; "
                f"worker exit codes: {[worker.exitcode for worker in workers]}"
            )
        results = merge_dp_shards(source, shard_outputs, args.data_parallel_size)
        metadata["dp_shards"] = [str(path) for path in shard_outputs]
        metadata["worker_exit_codes"] = [worker.exitcode for worker in workers]
        if failed_workers:
            print(
                "Warning: workers exited nonzero after writing complete validated "
                f"shards: {failed_workers}; merging validated results.",
                flush=True,
            )
    count = write_jsonl(output, results)
    correct = sum(row["grade"]["correct"] is True for row in results)
    graded = sum(row["grade"]["correct"] is not None for row in results)
    statuses = Counter(row["grade"]["status"] for row in results)
    summary = {"rows": count, "graded": graded, "correct": correct,
               "accuracy": correct / count if count else None,
               "accuracy_on_graded": correct / graded if graded else None,
               "grade_status_counts": dict(statuses), "output": str(output)}
    metadata.update(status="complete", rows=count, summary=summary)
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path

from .core import (
    grade, localize, read_jsonl, reasoning_boundaries, repair_rejection,
    sft_example, transition_sft_examples, write_jsonl,
)
from .prompts import ORACLE_SOLVE, QUESTION, REPAIR


def prepare(args):
    if Path(args.input).suffix == ".parquet":
        import pyarrow.parquet as parquet

        source = parquet.read_table(args.input).to_pylist()
    else:
        source = read_jsonl(args.input)
    seen_ids, seen_questions = set(), set()
    for index, row in enumerate(source):
        if args.limit is not None and index >= args.limit:
            break
        if row.get("split", args.split) != args.split:
            raise ValueError("Input split disagrees with --split; do not relabel evaluation data")
        row = row.get("extra_info", row)
        question = row.get("question", row.get("problem"))
        solution = row.get("solution", row.get("answer"))
        if not isinstance(question, str) or not question.strip() or not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"Invalid question/solution at input row {index}")
        identifier = str(row.get("id", f"{args.split}:{index}"))
        if identifier in seen_ids or question.strip() in seen_questions:
            raise ValueError(f"Duplicate input id or question at row {index}")
        seen_ids.add(identifier)
        seen_questions.add(question.strip())
        yield {"id": identifier, "question": question, "solution": solution, "split": args.split}


def collect(args, backend):
    tokenizer = backend.tokenizer
    for index, source in enumerate(read_jsonl(args.input)):
        if args.limit is not None and index >= args.limit:
            break
        if source["split"] not in {"train", "validation", "test"}:
            raise ValueError("Every input row must have an explicit valid split")
        prompt = backend.chat(QUESTION.format(question=source["question"]), thinking=True)
        suffix = tokenizer.encode(backend.probe_suffix, add_special_tokens=False)
        trajectory_started = time.perf_counter()
        trajectory = backend.generate([prompt], args.max_trajectory_tokens, args.temperature)[0][0]
        trajectory_seconds = time.perf_counter() - trajectory_started
        row = dict(
            source, schema_version=1, model=args.model, tokenizer_fingerprint=backend.fingerprint,
            prompt_ids=prompt, trajectory_ids=trajectory["token_ids"], trajectory=trajectory["text"],
            trajectory_finish_reason=trajectory["finish_reason"],
            final_grade=grade(trajectory["text"], source["solution"], trajectory["finish_reason"]),
            boundary_mode=args.boundary, probe_suffix_ids=suffix, probes=[],
            collect_cost={"trajectory_input_tokens": len(prompt),
                          "trajectory_output_tokens": len(trajectory["token_ids"]),
                          "probe_input_tokens": 0, "probe_output_tokens": 0,
                          "trajectory_seconds": trajectory_seconds, "probe_seconds": 0.0},
        )
        offsets = reasoning_boundaries(trajectory["token_ids"], tokenizer, args.boundary, args.interval)
        for batch_start in range(0, len(offsets), args.probe_batch_size):
            batch_offsets = offsets[batch_start:batch_start + args.probe_batch_size]
            requests = [prompt + trajectory["token_ids"][:offset] + suffix for offset in batch_offsets]
            probe_started = time.perf_counter()
            results = backend.generate(requests, args.max_answer_tokens, args.probe_temperature, args.probe_samples)
            row["collect_cost"]["probe_seconds"] += time.perf_counter() - probe_started
            row["collect_cost"]["probe_input_tokens"] += sum(len(request) for request in requests)
            row["collect_cost"]["probe_output_tokens"] += sum(len(sample["token_ids"]) for samples in results for sample in samples)
            for offset, samples in zip(batch_offsets, results):
                row["probes"].append({"offset": offset, "samples": [
                    dict(sample, **grade(sample["text"], source["solution"], sample["finish_reason"]))
                    for sample in samples
                ]})
        print(f"Collected {source['id']}: {len(offsets)} boundaries", flush=True)
        yield row


def repair(args, backend):
    tokenizer = backend.tokenizer
    for row in read_jsonl(args.input):
        if not row.get("candidate"):
            yield dict(row, repair_status="no_candidate", continuations=[])
            continue
        if row["tokenizer_fingerprint"] != backend.fingerprint or row["model"] != args.model:
            raise ValueError("Repair must use the same model and tokenizer as collection")
        start, end = row["candidate"]["start"], row["candidate"]["end"]
        if not 0 <= start < end <= len(row["trajectory_ids"]):
            raise ValueError("Invalid candidate token interval")
        prefix_ids = row["trajectory_ids"][:start]
        last_result = None
        attempt_limit = args.max_retries + 1
        for attempt in range(1, attempt_limit + 1):
            oracle_prompt = backend.chat(ORACLE_SOLVE.format(
                question=row["question"], solution=row["solution"],
            ), thinking=True)
            oracle_attempt = backend.generate(
                [oracle_prompt], args.max_oracle_tokens, args.repair_temperature,
            )[0][0]
            teacher_content = REPAIR.format(
                question=row["question"], solution=row["solution"],
                correct_attempt=oracle_attempt["text"],
                incorrect_attempt=row["trajectory"],
                prefix=tokenizer.decode(prefix_ids, skip_special_tokens=False),
                segment=tokenizer.decode(row["trajectory_ids"][start:end], skip_special_tokens=False),
            )
            teacher_prompt = backend.chat(teacher_content, thinking=False)
            generated = backend.generate(
                [teacher_prompt], args.max_step_tokens, args.repair_temperature, stop=["\n\n"],
            )[0][0]
            step = generated["text"].rstrip()
            cost = {"oracle_solve_input_tokens": len(oracle_prompt),
                    "oracle_solve_output_tokens": len(oracle_attempt["token_ids"]),
                    "teacher_input_tokens": len(teacher_prompt),
                    "teacher_output_tokens": len(generated["token_ids"]),
                    "repair_probe_input_tokens": 0, "repair_probe_output_tokens": 0,
                    "continuation_input_tokens": 0, "continuation_output_tokens": 0}
            rejection = repair_rejection(step, row["solution"], args.allow_answer_in_step)
            if generated["finish_reason"] != "stop":
                rejection = "incomplete_step"
            if rejection:
                result = dict(row, repair_status=rejection,
                              repair_success_criterion=args.success_criterion,
                              proposed_step=step, continuations=[], repair_cost=cost)
            else:
                step_ids = tokenizer.encode(step + "\n\n", add_special_tokens=False)
                student_prompt = row["prompt_ids"] + prefix_ids + step_ids
                if args.success_criterion == "probe":
                    probe_prompt = student_prompt + row["probe_suffix_ids"]
                    probe = backend.generate(
                        [probe_prompt], args.max_repair_probe_tokens, args.repair_probe_temperature,
                    )[0][0]
                    cost["repair_probe_input_tokens"] = len(probe_prompt)
                    cost["repair_probe_output_tokens"] = len(probe["token_ids"])
                    repair_probe = dict(probe, **grade(probe["text"], row["solution"], probe["finish_reason"]))
                    status = "accepted" if repair_probe["correct"] is True else "repair_probe_incorrect"
                    result = dict(row, repair_status=status,
                                  repair_success_criterion=args.success_criterion,
                                  proposed_step=step, replacement=step + "\n\n",
                                  replacement_ids=step_ids, repair_probe=repair_probe,
                                  continuations=[], repair_cost=cost)
                    if status == "accepted":
                        print(f"Repaired {row['id']}: repair-step probe correct on attempt {attempt}", flush=True)
                else:
                    continuations = backend.generate(
                        [student_prompt], args.max_continuation_tokens, args.continuation_temperature,
                        args.continuation_samples,
                    )[0]
                    cost["continuation_input_tokens"] = len(student_prompt)
                    cost["continuation_output_tokens"] = sum(len(sample["token_ids"]) for sample in continuations)
                    for sample in continuations:
                        sample["grade"] = grade(sample["text"], row["solution"], sample["finish_reason"])
                    print(f"Repaired {row['id']}: {sum(sample['grade']['correct'] is True for sample in continuations)} successful", flush=True)
                    result = dict(row, repair_status="accepted", replacement=step + "\n\n",
                                  replacement_ids=step_ids, continuations=continuations,
                                  repair_cost=cost)
            result["repair_attempts"] = attempt
            result["repair_retries"] = attempt - 1
            last_result = result
            if args.success_criterion != "probe" or result["repair_status"] == "accepted":
                break
        yield last_result


def add_backend_options(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-prefix-caching", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description="Answer-probe-guided intervention data pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ["prepare", "collect", "localize", "repair", "build-sft", "build-transition-sft", "summarize"]:
        command = subparsers.add_parser(name)
        command.add_argument("--input", nargs="+" if name == "build-transition-sft" else None, required=True)
        if name != "summarize":
            command.add_argument("--output", required=True)
        commands[name] = command
    prepare_parser = commands["prepare"]
    prepare_parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    prepare_parser.add_argument("--limit", type=int)
    collect_parser = commands["collect"]
    add_backend_options(collect_parser)
    collect_parser.add_argument("--limit", type=int)
    collect_parser.add_argument("--boundary", choices=["step", "token"], default="step")
    collect_parser.add_argument("--interval", type=int, default=256)
    collect_parser.add_argument("--max-trajectory-tokens", type=int, default=8192)
    collect_parser.add_argument("--temperature", type=float, default=0.7)
    collect_parser.add_argument("--max-answer-tokens", type=int, default=128)
    collect_parser.add_argument("--probe-samples", type=int, default=1)
    collect_parser.add_argument("--probe-temperature", type=float, default=0)
    collect_parser.add_argument("--probe-batch-size", type=int, default=16)
    commands["localize"].add_argument("--correct-threshold", type=float, default=0.8)
    commands["localize"].add_argument("--wrong-threshold", type=float, default=0.2)
    repair_parser = commands["repair"]
    add_backend_options(repair_parser)
    repair_parser.add_argument("--max-oracle-tokens", type=int, default=4096)
    repair_parser.add_argument("--max-step-tokens", type=int, default=256)
    repair_parser.add_argument("--repair-temperature", type=float, default=0)
    repair_parser.add_argument("--max-continuation-tokens", type=int, default=8192)
    repair_parser.add_argument("--continuation-samples", type=int, default=4)
    repair_parser.add_argument("--continuation-temperature", type=float, default=0.7)
    repair_parser.add_argument("--success-criterion", choices=["continuation", "probe"], default="continuation")
    repair_parser.add_argument("--max-repair-probe-tokens", type=int, default=128)
    repair_parser.add_argument("--repair-probe-temperature", type=float, default=0)
    repair_parser.add_argument("--max-retries", type=int, default=0,
                               help="Maximum additional repair attempts after the initial attempt")
    repair_parser.add_argument("--allow-answer-in-step", action="store_true")
    commands["build-sft"].add_argument("--target", choices=["prefix_step", "step_only", "full"], default="prefix_step")
    transition_parser = commands["build-transition-sft"]
    transition_parser.add_argument("--limit", type=int, required=True)
    transition_parser.add_argument("--correct-threshold", type=float, default=0.8)
    transition_parser.add_argument("--wrong-threshold", type=float, default=0.2)
    return parser


def summarize(path):
    counts = Counter()
    for row in read_jsonl(path):
        counts["rows"] += 1
        counts[f"split:{row['split']}"] += 1
        if "final_grade" in row:
            counts[f"final_correct:{row['final_grade']['correct']}"] += 1
        for key in ["localization_status", "repair_status"]:
            if key in row:
                counts[f"{key}:{row[key]}"] += 1
        if "probes" in row:
            counts["probe_calls"] += sum(len(probe["samples"]) for probe in row["probes"])
        if "continuations" in row:
            successful = sum(sample["grade"]["correct"] is True for sample in row["continuations"])
            counts["continuation_calls"] += len(row["continuations"])
            counts["successful_continuations"] += successful
            counts["rows_with_successful_continuation"] += bool(successful)
        for stage in ["collect_cost", "repair_cost"]:
            for key, value in row.get(stage, {}).items():
                counts[f"tokens:{key}"] += value
    return dict(counts)


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name, value in vars(args).items():
        if name == "input" and isinstance(value, list):
            continue
        if (name.startswith("max_") or name.endswith("_samples") or name in {"interval", "probe_batch_size", "tensor_parallel_size", "limit"}) and isinstance(value, int) and value <= 0:
            parser.error(f"{name} must be positive")
        if "temperature" in name and value < 0:
            parser.error(f"{name} must be nonnegative")
    if args.command == "localize" and not 0 <= args.wrong_threshold < args.correct_threshold <= 1:
        parser.error("Require 0 <= wrong-threshold < correct-threshold <= 1")
    if args.command == "build-transition-sft" and not 0 <= args.wrong_threshold < args.correct_threshold <= 1:
        parser.error("Require 0 <= wrong-threshold < correct-threshold <= 1")
    if args.command == "collect" and args.probe_samples > 1 and args.probe_temperature == 0:
        parser.error("Multiple probe samples require nonzero --probe-temperature")
    if args.command == "repair" and args.continuation_samples > 1 and args.continuation_temperature == 0:
        parser.error("Multiple continuations require nonzero --continuation-temperature")
    if args.command == "repair" and args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    if args.command == "summarize":
        print(json.dumps(summarize(args.input), indent=2, ensure_ascii=False))
        return
    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its manifest")
    input_paths = args.input if isinstance(args.input, list) else [args.input]
    for input_path in input_paths:
        input_manifest = Path(input_path).with_suffix(Path(input_path).suffix + ".manifest.json")
        if input_manifest.exists() and json.loads(input_manifest.read_text())["status"] != "complete":
            raise ValueError("Input manifest is incomplete; refusing a partial upstream run")
    hasher = hashlib.sha256()
    for input_path in input_paths:
        hasher.update(input_path.encode())
        with Path(input_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    input_hash = hasher.hexdigest()
    versions = {}
    for package in ["probe-intervention", "vllm", "transformers", "math-verify"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"arguments": vars(args), "input_sha256": input_hash, "versions": versions, "status": "running"}
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    if args.command in {"collect", "repair"}:
        from .backend import VLLMBackend

        backend = VLLMBackend(args)
        rows = collect(args, backend) if args.command == "collect" else repair(args, backend)
    elif args.command == "prepare":
        rows = prepare(args)
    elif args.command == "localize":
        rows = (localize(row, args.correct_threshold, args.wrong_threshold) for row in read_jsonl(args.input))
    elif args.command == "build-transition-sft":
        unique_rows, examples = set(), {}
        for input_path in input_paths:
            for row in read_jsonl(input_path):
                if row["id"] in unique_rows:
                    continue
                unique_rows.add(row["id"])
                for example in transition_sft_examples(row, args.correct_threshold, args.wrong_threshold):
                    examples[example["id"]] = example
        selected = sorted(examples.values(), key=lambda example: (example["source_id"], example["candidate"]["start"]))[:args.limit]
        if len(selected) != args.limit:
            raise ValueError(f"Found only {len(selected)} wrong-to-correct probe transitions; need {args.limit}")
        rows = iter(selected)
    else:
        rows = (example for row in read_jsonl(args.input) if (example := sft_example(row, args.target)) is not None)
    count = write_jsonl(output, rows)
    metadata.update(status="complete", rows=count)
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()

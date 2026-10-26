import argparse
import hashlib
import importlib.metadata
import json
import re
import time
from collections import Counter
from pathlib import Path

from .core import (
    grade, grade_probe, localize, read_jsonl, reference_answer, reasoning_boundaries, repair_rejection,
    correct_trajectory_sft_examples, sft_example, standard_sft_examples, tokenizer_fingerprint,
    transition_sft_examples, regression_reconstruction_candidate, write_jsonl,
)
from .prompts import (
    JUDGE_PROBE_SUFFIX, ORACLE_SOLVE, PRIVILEGED_SUFFIX_RECONSTRUCTION, QUESTION, REPAIR,
)


def repair_step_text(text, thinking_model=False):
    if thinking_model:
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        else:
            text = text.lstrip()
            if text.startswith("<think>"):
                text = text[len("<think>"):].lstrip()
    text = text.strip()
    if "<intervention>" in text and "</intervention>" in text:
        body = text.rsplit("<intervention>", 1)[1].split("</intervention>", 1)[0]
        match = re.search(r'"(?:content|intervention)"\s*:\s*"((?:\\.|[^"\\])*)"', body, re.DOTALL)
        if not match:
            return ""
        try:
            text = json.loads('"' + match.group(1) + '"')
        except json.JSONDecodeError:
            return ""
    elif "<repair_step>" in text and "</repair_step>" in text:
        text = text.rsplit("<repair_step>", 1)[1].split("</repair_step>", 1)[0]
    elif re.search(r'"intervention"\s*:\s*"', text):
        match = re.search(r'"intervention"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
        try:
            text = json.loads('"' + match.group(1) + '"') if match else ""
        except json.JSONDecodeError:
            return ""
    elif thinking_model and "<think>" in text:
        return ""
    return text.strip()


def context_limited_budget(backend, prompt, requested_tokens):
    model_limit = getattr(backend, "max_model_len", None)
    if model_limit is None:
        return requested_tokens
    available = model_limit - len(prompt)
    if available < 1:
        raise ValueError(
            f"Prompt uses {len(prompt)} tokens and leaves no generation space "
            f"under --max-model-len={model_limit}"
        )
    return min(requested_tokens, available)


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
        extra_info = row.get("extra_info") or {}
        if row.get("split", args.split) != args.split or extra_info.get("split", args.split) != args.split:
            raise ValueError("Input split disagrees with --split; do not relabel evaluation data")
        source = extra_info if "question" in extra_info else row
        question = source.get("question", source.get("problem"))
        solution = source.get("answer", source.get("solution"))
        if question is None:
            messages = row.get("prompt")
            if (isinstance(messages, list) and len(messages) == 1
                    and messages[0].get("role") == "user"):
                question = messages[0].get("content")
        if solution is None:
            solution = (row.get("reward_model") or {}).get("ground_truth")
        if not isinstance(question, str) or not question.strip() or not isinstance(solution, str) or not solution.strip():
            if args.skip_invalid:
                print(f"Skipping input row {index}: missing question or solution", flush=True)
                continue
            raise ValueError(f"Invalid question/solution at input row {index}")
        identifier = str(source.get("id", f"{args.split}:{index}"))
        if identifier in seen_ids or question.strip() in seen_questions:
            raise ValueError(f"Duplicate input id or question at row {index}")
        seen_ids.add(identifier)
        seen_questions.add(question.strip())
        yield {"id": identifier, "question": question, "solution": solution, "split": args.split}


def collect(args, backend):
    from tqdm import tqdm

    tokenizer = backend.tokenizer
    excluded_ids = set()
    for input_path in getattr(args, "exclude_input", []):
        excluded_ids.update(row.get("source_id", row["id"]) for row in read_jsonl(input_path))
    total = 0
    for index, source in enumerate(read_jsonl(args.input)):
        if args.limit is not None and index >= args.limit:
            break
        total += source["id"] not in excluded_ids

    def process_batch(batch):
        trajectory_started = time.perf_counter()
        trajectories = backend.generate(
            [item[1] for item in batch], args.max_trajectory_tokens, args.temperature,
        )
        trajectory_seconds_per_item = (time.perf_counter() - trajectory_started) / len(batch)
        for (source, prompt, suffix, judge_suffix), trajectory_samples in zip(batch, trajectories):
            trajectory = trajectory_samples[0]
            row = dict(
                source, schema_version=1, model=args.model, tokenizer_fingerprint=backend.fingerprint,
                prompt_ids=prompt, trajectory_ids=trajectory["token_ids"], trajectory=trajectory["text"],
                trajectory_finish_reason=trajectory["finish_reason"],
                final_grade=grade(trajectory["text"], source["solution"], trajectory["finish_reason"]),
                boundary_mode=args.boundary, probe_mode=args.probe_mode,
                probe_suffix_ids=suffix, judge_probe_suffix_ids=judge_suffix, probes=[],
                collect_cost={"trajectory_input_tokens": len(prompt),
                              "trajectory_output_tokens": len(trajectory["token_ids"]),
                              "probe_input_tokens": 0, "probe_output_tokens": 0,
                              "trajectory_seconds": trajectory_seconds_per_item, "probe_seconds": 0.0},
            )
            if not args.skip_probes:
                offsets = reasoning_boundaries(trajectory["token_ids"], tokenizer, args.boundary, args.interval)
                for batch_start in range(0, len(offsets), args.probe_batch_size):
                    batch_offsets = offsets[batch_start:batch_start + args.probe_batch_size]
                    active_suffix = judge_suffix if args.probe_mode == "judge" else suffix
                    requests = [prompt + trajectory["token_ids"][:offset] + active_suffix
                                for offset in batch_offsets]
                    probe_started = time.perf_counter()
                    if args.probe_mode == "judge":
                        results = [[sample] for sample in backend.score_yes_no(requests)]
                    else:
                        results = backend.generate(
                            requests, args.max_answer_tokens, args.probe_temperature, args.probe_samples,
                        )
                    row["collect_cost"]["probe_seconds"] += time.perf_counter() - probe_started
                    row["collect_cost"]["probe_input_tokens"] += sum(len(request) for request in requests)
                    row["collect_cost"]["probe_output_tokens"] += sum(len(sample["token_ids"]) for samples in results for sample in samples)
                    for offset, samples in zip(batch_offsets, results):
                        graded = (samples if args.probe_mode == "judge" else [
                            dict(sample, **grade_probe(sample["text"], source["solution"], sample["finish_reason"]))
                            for sample in samples
                        ])
                        row["probes"].append({"offset": offset, "samples": graded})
            yield row
            progress.update(1)

    with tqdm(total=total, desc="Collecting", unit="problem") as progress:
        pending = []
        for index, source in enumerate(read_jsonl(args.input)):
            if args.limit is not None and index >= args.limit:
                break
            if source["id"] in excluded_ids:
                continue
            if source["split"] not in {"train", "validation", "test"}:
                raise ValueError("Every input row must have an explicit valid split")
            prompt = backend.chat(QUESTION.format(question=source["question"]), thinking=True)
            suffix = tokenizer.encode(backend.probe_suffix, add_special_tokens=False)
            judge_suffix = None
            if args.probe_mode == "judge":
                gold_answer = json.dumps(reference_answer(source["solution"]), ensure_ascii=False)
                judge_suffix = tokenizer.encode(
                    JUDGE_PROBE_SUFFIX.format(gold_answer=gold_answer), add_special_tokens=False,
                )
            pending.append((source, prompt, suffix, judge_suffix))
            if len(pending) >= args.trajectory_batch_size:
                yield from process_batch(pending)
                pending = []
        if pending:
            yield from process_batch(pending)


def reprobe(args, backend):
    from tqdm import tqdm

    tokenizer = backend.tokenizer
    for row in tqdm(read_jsonl(args.input), desc="Re-probing", unit="problem"):
        if row.get("probe_mode") != "judge" or row.get("judge_probe_suffix_ids") is None:
            raise ValueError(f"Input row {row['id']} does not contain judge-probe metadata")
        if row["tokenizer_fingerprint"] != backend.fingerprint:
            raise ValueError(f"Tokenizer mismatch for {row['id']}")
        offsets = reasoning_boundaries(
            row["trajectory_ids"], tokenizer, args.boundary, args.interval,
        )
        probes = []
        probe_input_tokens = 0
        for batch_start in range(0, len(offsets), args.probe_batch_size):
            batch_offsets = offsets[batch_start:batch_start + args.probe_batch_size]
            requests = [row["prompt_ids"] + row["trajectory_ids"][:offset]
                        + row["judge_probe_suffix_ids"] for offset in batch_offsets]
            results = backend.score_yes_no(requests)
            probe_input_tokens += sum(map(len, requests))
            probes.extend({"offset": offset, "samples": [sample]}
                          for offset, sample in zip(batch_offsets, results))
        reprobed = dict(row, boundary_mode=args.boundary, interval=args.interval,
                        probes=probes)
        reprobed["reprobe_cost"] = {
            "probe_input_tokens": probe_input_tokens,
            "probe_calls": len(probes),
        }
        yield reprobed


def repair(args, backend):
    from tqdm import tqdm

    tokenizer = backend.tokenizer
    source_tokenizers = {}
    input_path = Path(args.input)
    total = None
    if input_path.exists():
        with input_path.open(encoding="utf-8") as stream:
            total = sum(bool(line.strip()) for line in stream)
    for row in tqdm(read_jsonl(args.input), total=total, desc="Repairing", unit="trajectory"):
        if not row.get("candidate"):
            yield dict(row, repair_status="no_candidate", continuations=[])
            continue
        cross_model = row["model"] != args.model
        if cross_model and not getattr(args, "allow_cross_model_repair", False):
            raise ValueError("Repair must use the same model and tokenizer as collection")
        if not cross_model and row["tokenizer_fingerprint"] != backend.fingerprint:
            raise ValueError("Repair must use the same tokenizer as collection")
        if cross_model:
            from transformers import AutoTokenizer

            source_tokenizer = source_tokenizers.get(row["model"])
            if source_tokenizer is None:
                source_tokenizer = AutoTokenizer.from_pretrained(row["model"], trust_remote_code=False)
                source_tokenizers[row["model"]] = source_tokenizer
            if tokenizer_fingerprint(source_tokenizer) != row["tokenizer_fingerprint"]:
                raise ValueError("Cached trajectory tokenizer fingerprint does not match its source model")
        start, end = row["candidate"]["start"], row["candidate"]["end"]
        if not 0 <= start < end <= len(row["trajectory_ids"]):
            raise ValueError("Invalid candidate token interval")
        prefix_ids = row["trajectory_ids"][:start]
        prefix_text = (source_tokenizer if cross_model else tokenizer).decode(
            prefix_ids, skip_special_tokens=False,
        )
        segment_text = (source_tokenizer if cross_model else tokenizer).decode(
            row["trajectory_ids"][start:end], skip_special_tokens=False,
        )
        if cross_model:
            repair_prompt_ids = backend.chat(QUESTION.format(question=row["question"]), thinking=True)
            repair_prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        else:
            repair_prompt_ids = row["prompt_ids"]
            repair_prefix_ids = prefix_ids
        last_result = None
        attempt_limit = args.max_retries + 1
        for attempt in range(1, attempt_limit + 1):
            oracle_prompt = backend.chat(ORACLE_SOLVE.format(
                question=row["question"], solution=row["solution"],
            ), thinking=True)
            oracle_attempt = backend.generate(
                [oracle_prompt], context_limited_budget(
                    backend, oracle_prompt, args.max_oracle_tokens,
                ), args.repair_temperature,
            )[0][0]
            teacher_content = REPAIR.format(
                question=row["question"], solution=row["solution"],
                correct_attempt=oracle_attempt["text"],
                correct_answer=reference_answer(row["solution"]),
                incorrect_attempt=row["trajectory"],
                prefix=prefix_text,
                segment=segment_text,
            )
            thinking_model = getattr(backend, "has_thinking_tokens", False)
            if hasattr(tokenizer, "apply_chat_template"):
                teacher_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": ORACLE_SOLVE.format(
                            question=row["question"], solution=row["solution"],
                        )},
                        {"role": "assistant", "content": oracle_attempt["text"]},
                        {"role": "user", "content": teacher_content},
                    ], tokenize=True, add_generation_prompt=True, enable_thinking=False,
                )
            else:
                teacher_prompt = backend.chat(teacher_content, thinking=False)
            generated = backend.generate(
                [teacher_prompt], context_limited_budget(
                    backend, teacher_prompt, args.max_step_tokens,
                ), args.repair_temperature,
                stop=None if thinking_model else ["\n\n"],
            )[0][0]
            step = repair_step_text(generated["text"], thinking_model)
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
                student_prompt = repair_prompt_ids + repair_prefix_ids + step_ids
                if args.success_criterion in {"probe", "probe_and_continuation"}:
                    probe_suffix_ids = (tokenizer.encode(backend.probe_suffix, add_special_tokens=False)
                                        if cross_model else row["probe_suffix_ids"])
                    probe_prompt = student_prompt + probe_suffix_ids
                    probe = backend.generate(
                        [probe_prompt], context_limited_budget(
                            backend, probe_prompt, args.max_repair_probe_tokens,
                        ), args.repair_probe_temperature,
                    )[0][0]
                    cost["repair_probe_input_tokens"] = len(probe_prompt)
                    cost["repair_probe_output_tokens"] = len(probe["token_ids"])
                    repair_probe = dict(probe, **grade_probe(probe["text"], row["solution"], probe["finish_reason"]))
                    continuations = []
                    if repair_probe["correct"] is True and args.success_criterion == "probe_and_continuation":
                        continuations = backend.generate(
                            [student_prompt], context_limited_budget(
                                backend, student_prompt, args.max_continuation_tokens,
                            ), args.continuation_temperature,
                            args.continuation_samples,
                        )[0]
                        cost["continuation_input_tokens"] = len(student_prompt)
                        cost["continuation_output_tokens"] = sum(
                            len(sample["token_ids"]) for sample in continuations
                        )
                        for sample in continuations:
                            sample["grade"] = grade(
                                sample["text"], row["solution"], sample["finish_reason"],
                            )
                    has_correct_continuation = any(
                        sample["grade"]["correct"] is True for sample in continuations
                    )
                    if repair_probe["correct"] is not True:
                        status = "repair_probe_incorrect"
                    elif args.success_criterion == "probe_and_continuation" and not has_correct_continuation:
                        status = "continuation_incorrect"
                    else:
                        status = "accepted"
                    result = dict(row, repair_status=status,
                                  repair_success_criterion=args.success_criterion,
                                  proposed_step=step, replacement=step + "\n\n",
                                  replacement_ids=step_ids, repair_probe=repair_probe,
                                  continuations=continuations, repair_cost=cost)
                else:
                    context_limit = getattr(backend, "max_model_len", None)
                    available_tokens = context_limit - len(student_prompt) if context_limit is not None else args.max_continuation_tokens
                    if available_tokens <= 0:
                        result = dict(
                            row, repair_status="context_limit_exceeded",
                            repair_error=(f"continuation prompt has {len(student_prompt)} tokens; "
                                          f"model limit is {context_limit}"),
                            proposed_step=step, continuations=[], repair_cost=cost,
                        )
                    else:
                        continuation_budget = context_limited_budget(
                            backend, student_prompt, args.max_continuation_tokens,
                        )
                        continuations = backend.generate(
                            [student_prompt], continuation_budget, args.continuation_temperature,
                            args.continuation_samples,
                        )[0]
                        cost["continuation_input_tokens"] = len(student_prompt)
                        cost["continuation_output_tokens"] = sum(
                            len(sample["token_ids"]) for sample in continuations
                        )
                        for sample in continuations:
                            sample["grade"] = grade(sample["text"], row["solution"], sample["finish_reason"])
                        result = dict(row, repair_status="accepted", replacement=step + "\n\n",
                                      replacement_ids=step_ids, continuations=continuations,
                                      repair_cost=cost)
            if cross_model:
                result.update(repair_model=args.model, repair_prompt_ids=repair_prompt_ids,
                              repair_prefix_ids=repair_prefix_ids,
                              repair_tokenizer_fingerprint=backend.fingerprint,
                              repair_prefix_text=prefix_text)
            result["repair_attempts"] = attempt
            result["repair_retries"] = attempt - 1
            last_result = result
            if args.success_criterion not in {"probe", "probe_and_continuation"} or result["repair_status"] == "accepted":
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
    for name in ["prepare", "collect", "reprobe", "localize", "repair", "build-sft", "build-transition-sft", "build-privileged-regression-sft", "build-standard-sft", "build-correct-trajectory-sft", "summarize"]:
        command = subparsers.add_parser(name)
        command.add_argument("--input", nargs="+" if name in {"build-transition-sft", "build-privileged-regression-sft", "build-correct-trajectory-sft"} else None, required=True)
        if name != "summarize":
            command.add_argument("--output", required=True)
        commands[name] = command
    prepare_parser = commands["prepare"]
    prepare_parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.add_argument("--skip-invalid", action="store_true",
                                help="Skip rows missing a question or reference solution")
    collect_parser = commands["collect"]
    add_backend_options(collect_parser)
    collect_parser.add_argument("--limit", type=int)
    collect_parser.add_argument("--exclude-input", nargs="+", default=[])
    collect_parser.add_argument("--skip-probes", action="store_true",
                                help="Collect generated trajectories without answer probes")
    collect_parser.add_argument("--boundary", choices=["step", "token"], default="step")
    collect_parser.add_argument("--interval", type=int, default=256)
    collect_parser.add_argument("--max-trajectory-tokens", type=int, default=8192)
    collect_parser.add_argument("--trajectory-batch-size", type=int, default=1,
                                help="Number of problems to generate trajectories for in one vLLM batch")
    collect_parser.add_argument("--temperature", type=float, default=0.7)
    collect_parser.add_argument("--max-answer-tokens", type=int, default=128)
    collect_parser.add_argument("--probe-samples", type=int, default=1)
    collect_parser.add_argument("--probe-temperature", type=float, default=0)
    collect_parser.add_argument("--probe-batch-size", type=int, default=4)
    collect_parser.add_argument("--probe-mode", choices=["answer", "judge"], default="answer",
                                help="Generate an answer probe or score yes/no against the reference answer")
    reprobe_parser = commands["reprobe"]
    add_backend_options(reprobe_parser)
    reprobe_parser.add_argument("--boundary", choices=["step", "token"], default="token")
    reprobe_parser.add_argument("--interval", type=int, default=256)
    reprobe_parser.add_argument("--probe-batch-size", type=int, default=16)
    commands["localize"].add_argument("--correct-threshold", type=float, default=0.8)
    commands["localize"].add_argument("--wrong-threshold", type=float, default=0.2)
    repair_parser = commands["repair"]
    add_backend_options(repair_parser)
    repair_parser.add_argument("--max-oracle-tokens", type=int, default=4096)
    repair_parser.add_argument("--max-step-tokens", type=int, default=4096)
    repair_parser.add_argument("--repair-temperature", type=float, default=0)
    repair_parser.add_argument("--max-continuation-tokens", type=int, default=8192)
    repair_parser.add_argument("--continuation-samples", type=int, default=4)
    repair_parser.add_argument("--continuation-temperature", type=float, default=0.7)
    repair_parser.add_argument("--success-criterion", choices=["continuation", "probe", "probe_and_continuation"], default="continuation")
    repair_parser.add_argument("--max-repair-probe-tokens", type=int, default=128)
    repair_parser.add_argument("--repair-probe-temperature", type=float, default=0)
    repair_parser.add_argument("--max-retries", type=int, default=0,
                               help="Maximum additional repair attempts after the initial attempt")
    repair_parser.add_argument("--allow-answer-in-step", action="store_true")
    repair_parser.add_argument("--allow-cross-model-repair", action="store_true",
                               help="Allow a different repair model when tokenizer fingerprints match")
    commands["build-sft"].add_argument("--target", choices=["prefix_step", "step_only", "full"], default="prefix_step")
    transition_parser = commands["build-transition-sft"]
    transition_parser.add_argument("--target", choices=["prefix_recovery", "step_only"],
                                   default="prefix_recovery",
                                   help="step_only masks the reasoning prefix before the W→R segment")
    transition_parser.add_argument("--limit", type=int, required=True)
    transition_parser.add_argument("--correct-threshold", type=float, default=0.8)
    transition_parser.add_argument("--wrong-threshold", type=float, default=0.2)
    transition_parser.add_argument("--consecutive-transitions", type=int, default=1,
                                   help="Require this many W->C transitions in sequence, ignoring uncertain probes")
    transition_parser.add_argument("--skip-uncertain-probes", action="store_true",
                                   help="Find the next decided state after W, skipping uncertain probes")
    transition_parser.add_argument("--min-target-tokens", type=int, default=0)
    transition_parser.add_argument("--max-target-tokens", type=int)
    transition_parser.add_argument("--min-probe-gap", type=int, default=1,
                                   help="Minimum probe-index distance between W and C")
    transition_parser.add_argument("--require-stable-correct", action="store_true",
                                   help="Reject W→C spans followed by any later wrong probe")
    transition_parser.add_argument("--require-correct-final-grade", action="store_true",
                                   help="Keep only source trajectories whose final answer is correct")
    transition_parser.add_argument(
        "--allow-multiple-transitions-per-trajectory", action="store_true",
        help="Keep every adjacent wrong-to-correct transition; by default keep only the first per trajectory",
    )
    transition_parser.add_argument(
        "--allow-incomplete-input", action="store_true",
        help="Use complete JSONL rows from an interrupted probe collection",
    )
    privileged_parser = commands["build-privileged-regression-sft"]
    add_backend_options(privileged_parser)
    privileged_parser.add_argument("--limit", type=int)
    privileged_parser.add_argument("--correct-threshold", type=float, default=0.6)
    privileged_parser.add_argument("--wrong-threshold", type=float, default=0.4)
    privileged_parser.add_argument("--max-reconstruction-tokens", type=int, default=8192)
    privileged_parser.add_argument("--batch-size", type=int, default=4)
    privileged_parser.add_argument("--max-training-length", type=int, default=32768)
    privileged_parser.add_argument("--resume-input",
                                   help="Reuse completed examples from an interrupted export")
    privileged_parser.add_argument("--skip-source-ids", nargs="*", default=[],
                                   help="Exclude source trajectories whose reconstruction cannot finish")
    standard_parser = commands["build-standard-sft"]
    standard_parser.add_argument("--model", required=True)
    standard_parser.add_argument("--limit", type=int, required=True)
    standard_parser.add_argument("--exclude-input", nargs="+", default=[])
    trajectory_parser = commands["build-correct-trajectory-sft"]
    trajectory_parser.add_argument("--model", required=True)
    trajectory_parser.add_argument("--limit", type=int, required=True)
    trajectory_parser.add_argument("--exclude-input", nargs="+", default=[])
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
        if (name.startswith("max_") or name.endswith("_samples") or name in {"interval", "batch_size", "probe_batch_size", "trajectory_batch_size", "tensor_parallel_size", "limit"}) and isinstance(value, int) and value <= 0:
            parser.error(f"{name} must be positive")
        if "temperature" in name and value < 0:
            parser.error(f"{name} must be nonnegative")
    if args.command == "localize" and not 0 <= args.wrong_threshold <= args.correct_threshold <= 1:
        parser.error("Require 0 <= wrong-threshold <= correct-threshold <= 1")
    if args.command == "build-transition-sft" and not 0 <= args.wrong_threshold <= args.correct_threshold <= 1:
        parser.error("Require 0 <= wrong-threshold <= correct-threshold <= 1")
    if args.command == "build-privileged-regression-sft" and not 0 <= args.wrong_threshold <= args.correct_threshold <= 1:
        parser.error("Require 0 <= wrong-threshold <= correct-threshold <= 1")
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
    input_paths += getattr(args, "exclude_input", [])
    for input_path in input_paths:
        input_manifest = Path(input_path).with_suffix(Path(input_path).suffix + ".manifest.json")
        if input_manifest.exists() and json.loads(input_manifest.read_text())["status"] != "complete" and not (
            args.command == "build-transition-sft" and args.allow_incomplete_input
        ):
            raise ValueError("Input manifest is incomplete; refusing a partial upstream run")
    hasher = hashlib.sha256()
    hash_paths = input_paths + ([args.resume_input] if getattr(args, "resume_input", None) else [])
    for input_path in hash_paths:
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
    if args.command in {"collect", "reprobe", "repair", "build-privileged-regression-sft"}:
        from .backend import VLLMBackend

        backend = VLLMBackend(args)
        if args.command == "collect":
            rows = collect(args, backend)
        elif args.command == "reprobe":
            rows = reprobe(args, backend)
        elif args.command == "repair":
            rows = repair(args, backend)
        else:
            completed = list(read_jsonl(args.resume_input)) if args.resume_input else []
            completed_ids = {row["source_id"] for row in completed}
            completed_ids.update(args.skip_source_ids)
            selected_rows = {}
            for input_path in input_paths:
                for row in read_jsonl(input_path):
                    if row["id"] in selected_rows or row["id"] in completed_ids:
                        continue
                    candidate = regression_reconstruction_candidate(
                        row, args.correct_threshold, args.wrong_threshold,
                    )
                    if candidate:
                        selected_rows[row["id"]] = (row, candidate)
            selected = sorted(selected_rows.values(), key=lambda item: item[0]["id"])
            if args.limit is not None:
                selected = selected[:args.limit]

            def reconstruct():
                from tqdm import tqdm

                for batch_start in tqdm(range(0, len(selected), args.batch_size),
                                        desc="Reconstructing suffixes", unit="batch"):
                    batch = selected[batch_start:batch_start + args.batch_size]
                    prompts, contexts, owners, budgets = [], [], [], []
                    for row, candidate in batch:
                        start = candidate["start"]
                        context = row["prompt_ids"] + row["trajectory_ids"][:start]
                        prefix_text = backend.tokenizer.decode(
                            row["trajectory_ids"][:start], skip_special_tokens=False,
                        )
                        suffix_text = backend.tokenizer.decode(
                            row["trajectory_ids"][start:], skip_special_tokens=False,
                        )
                        teacher_content = PRIVILEGED_SUFFIX_RECONSTRUCTION.format(
                            question=row["question"], solution=row["solution"],
                            prefix=prefix_text, suffix=suffix_text,
                        )
                        teacher_prompt = backend.chat(teacher_content, thinking=True)
                        budget = min(args.max_reconstruction_tokens,
                                     args.max_training_length - len(context),
                                     backend.max_model_len - len(teacher_prompt))
                        if budget < 1:
                            raise ValueError(f"No reconstruction budget remains for {row['id']}")
                        prompts.append(teacher_prompt)
                        contexts.append(context)
                        owners.append((row, candidate))
                        budgets.append(budget)
                    generations = backend.generate(prompts, min(budgets), temperature=0)
                    for (row, candidate), context, generation in zip(owners, contexts, generations):
                        sample = generation[0]
                        if sample["finish_reason"] != "stop":
                            raise ValueError(
                                f"Privileged reconstruction did not finish for {row['id']} "
                                f"(finish_reason={sample['finish_reason']})"
                            )
                        target = sample["token_ids"]
                        if not target or len(context) + len(target) > args.max_training_length:
                            raise ValueError(f"Invalid reconstructed target length for {row['id']}")
                        yield {
                            "schema_version": 1,
                            "id": f"{row['id']}:privileged-regression:{candidate['start']}",
                            "source_id": row["id"], "split": "train",
                            "target": "privileged_regression_suffix",
                            "tokenizer_fingerprint": row["tokenizer_fingerprint"],
                            "input_ids": context + target,
                            "labels": [-100] * len(context) + target,
                            "candidate": candidate, "source_model": row["model"],
                            "teacher_model": args.model,
                        }
            rows = iter(completed + list(reconstruct()))
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
                for example in transition_sft_examples(row, args.correct_threshold, args.wrong_threshold,
                                                       target=args.target,
                                                       allow_multiple=args.allow_multiple_transitions_per_trajectory,
                                                       consecutive_transitions=args.consecutive_transitions,
                                                       skip_uncertain=args.skip_uncertain_probes,
                                                       min_target_tokens=args.min_target_tokens,
                                                       max_target_tokens=args.max_target_tokens,
                                                       min_probe_gap=args.min_probe_gap,
                                                       require_stable_correct=args.require_stable_correct,
                                                       require_correct_final_grade=args.require_correct_final_grade):
                    examples[example["id"]] = example
        selected = sorted(examples.values(), key=lambda example: (example["source_id"], example["candidate"]["start"]))[:args.limit]
        if len(selected) != args.limit:
            raise ValueError(f"Found only {len(selected)} wrong-to-correct probe transitions; need {args.limit}")
        rows = iter(selected)
    elif args.command in {"build-standard-sft", "build-correct-trajectory-sft"}:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        excluded_ids = set()
        for input_path in args.exclude_input:
            for row in read_jsonl(input_path):
                excluded_ids.add(row.get("source_id", row["id"]))
        if args.command == "build-standard-sft":
            examples = standard_sft_examples(
                read_jsonl(args.input), tokenizer, tokenizer_fingerprint(tokenizer),
                args.limit, excluded_ids,
            )
        else:
            examples = correct_trajectory_sft_examples(
                (row for input_path in args.input for row in read_jsonl(input_path)),
                tokenizer_fingerprint(tokenizer), args.limit, excluded_ids,
            )
        rows = iter(examples)
    else:
        rows = (example for row in read_jsonl(args.input) if (example := sft_example(row, args.target)) is not None)
    count = write_jsonl(output, rows)
    metadata.update(status="complete", rows=count)
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()

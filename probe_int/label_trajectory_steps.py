import argparse
import json
import re
from pathlib import Path

from .core import read_jsonl


LOCATE_PROMPT = """You are a careful mathematical verifier. Inspect the student's complete attempt against the reference solution.

The attempt is divided into numbered reasoning steps. Find the FIRST step whose prefix contains a definite mathematical or logical error that is unresolved at the END OF THAT STEP. Inspect prefixes chronologically: a correction in a later step must not erase the error in an earlier prefix. A false claim explicitly corrected within the SAME step is not unresolved at that step's end. Do not flag a harmless hesitation or an incomplete but valid step. If every prefix is free of unresolved definite errors, return NONE.

The prefix ending immediately BEFORE the selected step will be labeled correct, so verify that every earlier prefix is free of unresolved definite errors. The prefix ending AT the selected step will be labeled incorrect. Do not select an error only because it explains the final wrong answer if there is an earlier definite error.

Return exactly one line in this format, with no other text:
FIRST_CRITICAL_ERROR_STEP: <integer or NONE>

Problem:
{question}

Reference solution:
{solution}

Student attempt:
{numbered_attempt}
"""


def reasoning_steps(text):
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def render_steps(steps):
    return "\n\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, 1))


def parse_location(text, count):
    match = re.search(r"FIRST_CRITICAL_ERROR_STEP\s*:\s*(\d+|NONE)", text, re.IGNORECASE)
    if not match or match.group(1).upper() == "NONE":
        return None
    location = int(match.group(1))
    return location if 1 <= location <= count else None


def eligible_wrong_trajectory(row):
    return (
        row.get("trajectory_finish_reason") == "stop"
        and row.get("final_grade", {}).get("correct") is False
        and bool(row.get("trajectory"))
        and bool(row.get("question"))
        and bool(row.get("solution"))
    )


def make_pair(row, steps, location, model, response):
    pair_id = f"{row['id']}:first-error:{location}"
    examples = []
    for prefix_steps, step_index, label, verdict in (
        (steps[:location - 1], location - 1, "No", "Correct"),
        (steps[:location], location, "Yes", "Incorrect"),
    ):
        examples.append({
            "id": f"{pair_id}:{verdict.lower()}",
            "pair_id": pair_id,
            "source_id": row.get("source_id", row["id"]),
            "step_index": step_index,
            "question": row["question"],
            "reference_solution": row["solution"],
            "reasoning_prefix": "\n\n".join(prefix_steps),
            "label": label,
            "verdict": verdict,
            "teacher_model": model,
            "teacher_first_critical_error_step": location,
            "teacher_response": response,
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Create paired correctness labels from wrong R1 trajectories")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--limit", type=int, default=400, help="Number of binary examples; must be even")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.limit % 2:
        parser.error("--limit must be a positive even number")
    if args.tensor_parallel_size <= 0 or args.data_parallel_size <= 0:
        parser.error("tensor and data parallel sizes must be positive")

    from .vllm_dp import configure_runtime

    needs_triton_fallback = configure_runtime()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    engine_options = dict(
        model=args.model, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs, seed=args.seed,
    )
    if needs_triton_fallback:
        engine_options["gdn_prefill_backend"] = "triton"
    llm = None if args.data_parallel_size > 1 else LLM(**engine_options)
    params_kwargs = dict(temperature=0, max_tokens=args.max_tokens)
    params = SamplingParams(**params_kwargs)
    candidates = []
    for row in read_jsonl(args.input):
        if not eligible_wrong_trajectory(row):
            continue
        steps = reasoning_steps(row["trajectory"])
        if steps:
            candidates.append((row, steps))
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": LOCATE_PROMPT.format(
            question=row["question"], solution=row["solution"],
            numbered_attempt=render_steps(steps),
        )}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for row, steps in candidates]
    # vLLM schedules the full request list into batches based on available KV cache.
    if args.data_parallel_size > 1:
        from .vllm_dp import generate_with_dp

        outputs = generate_with_dp(
            prompts, params_kwargs, engine_options, args.data_parallel_size,
            use_tqdm=not args.no_progress,
        )
    else:
        outputs = llm.generate(prompts, params, use_tqdm=not args.no_progress)
    examples = []
    for (row, steps), output in zip(candidates, outputs):
        response = output.outputs[0].text
        location = parse_location(response, len(steps))
        if location is None:
            continue
        examples.extend(make_pair(row, steps, location, args.model, response))
        if len(examples) >= args.limit:
            break

    if len(examples) != args.limit:
        raise RuntimeError(
            f"Found only {len(examples)} examples ({len(examples) // 2} pairs) from "
            "completed R1 trajectories with an incorrect final answer and a located error; "
            f"need {args.limit}. Increase COLLECT_LIMIT."
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    counts = {label: sum(item["label"] == label for item in examples) for label in ("Yes", "No")}
    print(json.dumps({"output": str(output_path), "examples": len(examples),
                      "pairs": len(examples) // 2, "label_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()

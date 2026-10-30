"""GPU regression: force one native DP rank idle while its peer still decodes.

Run with CUDA_VISIBLE_DEVICES=2,3 uv run python -m scripts.smoke_vllm_dp_tail
and --model pointing to a local model snapshot for offline execution.
"""

import argparse
import json
from pathlib import Path

from probe_int.vllm_dp import generate_with_dp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher", action="store_true")
    args = parser.parse_args()
    options = dict(
        model=args.model, tensor_parallel_size=1, dtype="bfloat16",
        max_model_len=2048, max_num_seqs=2, seed=42,
        gpu_memory_utilization=0.8 if args.teacher else 0.2,
        enable_prefix_caching=True,
    )
    # Five requests: rank 0 gets three, rank 1 gets two. With n=2 and
    # max_num_seqs=2, rank 1 must run dummy forwards for a whole extra wave.
    prompts = ["Continue counting: 1, 2, 3, " + "4, " * i for i in range(5)]
    outputs = generate_with_dp(
        prompts,
        dict(n=2, temperature=1, max_tokens=96, ignore_eos=True),
        options, 2,
    )
    assert len(outputs) == 5, len(outputs)
    assert all(len(output.outputs) == 2 for output in outputs)
    assert all(len(sample.token_ids) == 96 for output in outputs for sample in output.outputs)

    # A single request leaves another rank with only its placeholder request.
    # For teacher, use an obvious arithmetic error to check actual annotation.
    from transformers import AutoTokenizer
    from probe_int.label_trajectory_steps import LOCATE_PROMPT, parse_location, render_steps

    steps = ["Let the numbers be 2 and 3.", "Their sum is 2 + 3 = 6.",
             "Therefore the answer is 6."]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": LOCATE_PROMPT.format(
            question="Compute 2 + 3.", solution="2 + 3 = 5.",
            numbered_attempt=render_steps(steps))}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    single = generate_with_dp(
        [prompt], dict(temperature=0, max_tokens=128), options, 2,
    )
    assert len(single) == 1
    response = single[0].outputs[0].text
    if args.teacher:
        assert parse_location(response, len(steps)) == 2, response
    result = dict(tail_requests=5, tail_completions=10, tokens_per_completion=96,
                  empty_rank_requests=1, teacher_response=response,
                  teacher_location=parse_location(response, len(steps)))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

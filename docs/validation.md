# Smoke validation

Date: 2026-10-21. This is an execution check, not a benchmark or method result.

- Created the standalone `uv` environment with Python 3.12, vLLM 0.11.0,
  PyTorch 2.8.0+cu128, Transformers 4.57.1 and GSM8K parquet support.
- Loaded cached Qwen3-1.7B on an NVIDIA RTX 3090 through vLLM.
- Ran `probe-int-eval` on one prepared GSM8K test example with `--max-tokens 8192`;
  generation completed, parsed answer was 72, and it matched the reference.
- Ran `probe-int collect` on one GSM8K train example with
  `--max-trajectory-tokens 8192`; the model stopped naturally at 1,158 tokens,
  produced a correct final answer, and 17 probe boundaries were sampled.
- Ran localization; it correctly skipped the final-correct sample. To exercise
  the repair API, injected a synthetic candidate span (not a probe-found error);
  the model generated a proposed step, which the answer-leakage guard rejected.
  Therefore no continuation or SFT sample was produced in this smoke run.

Earlier 512-token smoke sampling truncated before a gradeable answer. The 1,536-
token attempt was interrupted during vLLM initialization and is not counted as a
completed run. No multi-GPU training, intervention quality, or learning effect
was tested.

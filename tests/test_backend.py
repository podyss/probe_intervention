import unittest
import math
from types import SimpleNamespace

from probe_int.backend import QWEN25_PROBE_SUFFIX, VLLMBackend


class FakeTokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_sep|>", "<|im_end|>"]
    unk_token_id = -1

    def __init__(self, assistant_prefix):
        self.assistant_prefix = assistant_prefix

    def apply_chat_template(self, messages, **kwargs):
        return [1]

    def decode(self, token_ids, skip_special_tokens=False):
        return "<|im_start|>assistant\n" + self.assistant_prefix

    def encode(self, text, add_special_tokens=False):
        if text == "<think>\n":
            return [1]
        return [2]


class BackendChatTests(unittest.TestCase):
    def make_backend(self, assistant_prefix):
        backend = VLLMBackend.__new__(VLLMBackend)
        backend.tokenizer = FakeTokenizer(assistant_prefix)
        backend.probe_suffix = QWEN25_PROBE_SUFFIX
        return backend

    def test_chat_template_without_think_markers_uses_plain_probe_suffix(self):
        backend = self.make_backend("")

        self.assertEqual(backend.chat("solve", thinking=True), [1])
        self.assertEqual(backend.probe_suffix, QWEN25_PROBE_SUFFIX)

    def test_chat_template_batch_encoding_extracts_integer_input_ids(self):
        backend = self.make_backend("")
        backend.tokenizer.apply_chat_template = lambda *args, **kwargs: SimpleNamespace(
            input_ids=[7, 8, 9],
        )

        self.assertEqual(backend.chat("solve", thinking=True), [7, 8, 9])

    def test_chat_template_with_open_think_uses_qwen3_probe_suffix(self):
        from probe_int.prompts import PROBE_SUFFIX

        backend = self.make_backend("<think>\n")

        self.assertEqual(backend.chat("solve", thinking=True), [1])
        self.assertEqual(backend.probe_suffix, PROBE_SUFFIX)

    def test_deepseek_template_open_think_is_kept_for_reasoning_and_closed_for_probe(self):
        from probe_int.prompts import PROBE_SUFFIX

        backend = self.make_backend("<think>\n")
        prompt = backend.chat("solve", thinking=True)
        self.assertEqual(prompt, [1])
        self.assertEqual(backend.probe_suffix, PROBE_SUFFIX)

    def test_non_reasoning_request_removes_template_open_think(self):
        backend = self.make_backend("<think>\n")
        self.assertEqual(backend.chat("solve", thinking=False), [])

    def test_generation_error_reports_prompt_and_output_token_counts(self):
        backend = self.make_backend("")
        backend.max_model_len = 10

        with self.assertRaisesRegex(
            ValueError, r"max-model-len=10: prompt=8 tokens, requested_output=4 tokens",
        ):
            backend.generate([[0] * 8], max_tokens=4)

    def test_judge_probe_normalizes_yes_no_logprobs(self):
        backend = self.make_backend("")
        backend.yes_token_ids = [3]
        backend.no_token_ids = [4]
        backend.max_model_len = 10
        backend.args = SimpleNamespace(no_progress=True)
        backend.sampling_params = lambda **kwargs: kwargs
        backend.tokenizer.decode = lambda token_ids, **kwargs: "yes"
        calls = []

        def generate(prompts, params, use_tqdm):
            calls.append(params)
            return [SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=[3],
                logprobs=[{
                    3: SimpleNamespace(logprob=-0.2),
                    4: SimpleNamespace(logprob=-1.0),
                }],
            )]) for _ in prompts]

        backend.llm = SimpleNamespace(generate=generate)

        result = backend.score_yes_no([[1, 2]])[0]

        self.assertAlmostEqual(result["p_correct"], 1 / (1 + math.exp(-0.8)))
        self.assertEqual(result["answer"], "yes")
        self.assertEqual(result["status"], "scored")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["allowed_token_ids"], [3, 4])
        self.assertEqual(calls[0]["logprobs"], 2)
        self.assertEqual(calls[0]["temperature"], 1.0)

    def test_judge_probe_falls_back_when_top_k_omits_a_candidate(self):
        backend = self.make_backend("")
        backend.yes_token_ids = [3]
        backend.no_token_ids = [4]
        backend.max_model_len = 10
        backend.args = SimpleNamespace(no_progress=True)
        backend.sampling_params = lambda **kwargs: kwargs
        backend.tokenizer.decode = lambda token_ids, **kwargs: "yes"
        calls = []

        def generate(prompts, params, use_tqdm):
            calls.append((prompts, params))
            if "logprobs" in params:
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    token_ids=[3],
                    logprobs=[{3: SimpleNamespace(logprob=-0.2)}],
                )]) for _ in prompts]
            return [SimpleNamespace(
                prompt_logprobs=[None, {prompt["prompt_token_ids"][-1]: SimpleNamespace(
                    logprob=-0.5 if prompt["prompt_token_ids"][-1] == 3 else -1.2,
                )}],
                outputs=[SimpleNamespace(token_ids=[3])],
            ) for prompt in prompts]

        backend.llm = SimpleNamespace(generate=generate)

        result = backend.score_yes_no([[1, 2]])[0]

        self.assertAlmostEqual(result["p_correct"], 1 / (1 + math.exp(-1)))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1]["prompt_logprobs"], 1)
        self.assertEqual(len(calls[1][0]), 2)


if __name__ == "__main__":
    unittest.main()

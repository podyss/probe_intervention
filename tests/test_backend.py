import unittest

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

    def test_chat_template_with_open_think_uses_qwen3_probe_suffix(self):
        from probe_int.prompts import PROBE_SUFFIX

        backend = self.make_backend("<think>\n")

        self.assertEqual(backend.chat("solve", thinking=True), [1])
        self.assertEqual(backend.probe_suffix, PROBE_SUFFIX)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from probe_int.pipeline import collect, collection_sources, judge_prompt


class CollectBatchTests(unittest.TestCase):
    def test_judge_prompt_uses_standalone_user_turn(self):
        calls = []
        backend = SimpleNamespace(
            chat=lambda content, thinking: calls.append((content, thinking)) or [1, 2],
        )
        prompt = judge_prompt(backend, "What is 6 times 7?", "42", "6 * 7 = 42")

        self.assertEqual(prompt, [1, 2])
        self.assertFalse(calls[0][1])
        self.assertIn("definite mathematical or logical error", calls[0][0])
        self.assertIn("Do not answer yes merely because the reasoning is incomplete", calls[0][0])
        self.assertIn("Problem:\nWhat is 6 times 7?", calls[0][0])
        self.assertIn("Reference answer:\n42", calls[0][0])
        self.assertIn("Reasoning so far:\n6 * 7 = 42", calls[0][0])

    def test_repeat_input_cycles_rows_with_unique_trajectory_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            rows = [
                {"id": "train:a", "split": "train", "question": "qa", "solution": "a"},
                {"id": "train:b", "split": "train", "question": "qb", "solution": "b"},
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = SimpleNamespace(input=str(input_path), limit=5, repeat_input=True)

            sources = collection_sources(args, set())

        self.assertEqual([row["id"] for row in sources], [
            "train:a", "train:b", "train:a:rollout:1", "train:b:rollout:1", "train:a:rollout:2",
        ])
        self.assertEqual([row["source_id"] for row in sources], [
            "train:a", "train:b", "train:a", "train:b", "train:a",
        ])
        self.assertEqual([row["rollout_index"] for row in sources], [0, 0, 1, 1, 2])

    def test_trajectory_requests_are_submitted_once_for_vllm_scheduling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            rows = [
                {"id": str(i), "question": f"q{i}", "solution": "42", "split": "train"}
                for i in range(3)
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = SimpleNamespace(
                exclude_input=[], limit=None, input=str(input_path), model="test",
                max_trajectory_tokens=16, temperature=0.0,
                boundary="step", probe_mode="answer", skip_probes=True,
                probe_batch_size=4, interval=256, max_answer_tokens=8,
                probe_temperature=0.0, probe_samples=1,
            )
            backend = SimpleNamespace(
                tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [9]),
                probe_suffix="suffix", fingerprint="fingerprint", chat=lambda text, thinking: [1],
            )
            calls = []

            def generate(prompts, max_tokens, temperature):
                calls.append(len(prompts))
                return [[{"text": f"answer-{i}", "token_ids": [i], "finish_reason": "stop"}]
                        for i, _ in enumerate(prompts)]

            backend.generate = generate
            with patch("probe_int.pipeline.grade", return_value={"correct": True}), \
                    patch("probe_int.pipeline.time.perf_counter", side_effect=[0, 5]):
                output = list(collect(args, backend))

        self.assertEqual(calls, [3])
        self.assertEqual([row["id"] for row in output], ["0", "1", "2"])
        self.assertEqual([row["trajectory"] for row in output], ["answer-0", "answer-1", "answer-2"])
        self.assertEqual(sum(row["collect_cost"]["trajectory_seconds"] for row in output), 5)


if __name__ == "__main__":
    unittest.main()

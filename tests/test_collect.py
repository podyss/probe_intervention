import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from probe_int.pipeline import collect


class CollectBatchTests(unittest.TestCase):
    def test_trajectory_batching_preserves_rows_and_splits_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            rows = [
                {"id": str(i), "question": f"q{i}", "solution": "42", "split": "train"}
                for i in range(3)
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = SimpleNamespace(
                exclude_input=[], limit=None, input=str(input_path), model="test",
                trajectory_batch_size=2, max_trajectory_tokens=16, temperature=0.0,
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
                    patch("probe_int.pipeline.time.perf_counter", side_effect=[0, 2, 2, 5]):
                output = list(collect(args, backend))

        self.assertEqual(calls, [2, 1])
        self.assertEqual([row["id"] for row in output], ["0", "1", "2"])
        self.assertEqual([row["trajectory"] for row in output], ["answer-0", "answer-1", "answer-0"])
        self.assertEqual(sum(row["collect_cost"]["trajectory_seconds"] for row in output), 5)


if __name__ == "__main__":
    unittest.main()

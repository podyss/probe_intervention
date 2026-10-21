import unittest
from types import SimpleNamespace
from unittest.mock import patch

from probe_int.core import localize, sft_example, transition_sft_examples
from probe_int.pipeline import repair


class LocalizationTests(unittest.TestCase):
    def row(self, states, final_correct=True):
        return {
            "final_grade": {"correct": final_correct},
            "probes": [
                {"offset": index * 10, "samples": [{"correct": state}]}
                for index, state in enumerate(states)
            ],
        }

    def test_final_grade_does_not_filter_regression(self):
        for final_correct in (True, False, None):
            with self.subTest(final_correct=final_correct):
                result = localize(self.row([True, False, True], final_correct))
                self.assertEqual(result["candidate"]["start"], 0)
                self.assertEqual(result["candidate"]["end"], 10)

    def test_uncertain_boundary_breaks_adjacency(self):
        result = localize(self.row([True, None, False]))
        self.assertIsNone(result["candidate"])

    def test_first_adjacent_regression_is_selected(self):
        result = localize(self.row([False, True, False, True, False]))
        self.assertEqual(result["candidate"]["start"], 10)
        self.assertEqual(result["candidate"]["end"], 20)

    def test_final_correct_without_regression_has_no_candidate(self):
        result = localize(self.row([False, True, True]))
        self.assertIsNone(result["candidate"])


class SFTExampleTests(unittest.TestCase):
    def test_probe_success_is_sufficient_without_correct_continuation(self):
        row = {
            "id": "train:1", "split": "train", "repair_status": "accepted",
            "repair_success_criterion": "probe", "repair_probe": {"correct": True},
            "continuations": [], "prompt_ids": [1, 2], "trajectory_ids": [3, 4, 5],
            "candidate": {"start": 2, "end": 3}, "replacement_ids": [8, 9],
            "tokenizer_fingerprint": "fingerprint", "model": "model",
        }

        example = sft_example(row)

        self.assertEqual(example["input_ids"], [1, 2, 3, 4, 8, 9])
        self.assertEqual(example["labels"], [-100, -100, 3, 4, 8, 9])

    def test_wrong_to_correct_transition_supervises_recovery_prefix(self):
        row = {
            "id": "train:transition", "split": "train", "probes": [
                {"offset": 1, "samples": [{"correct": False}]},
                {"offset": 3, "samples": [{"correct": True}]},
                {"offset": 4, "samples": [{"correct": False}]},
            ], "trajectory_ids": [10, 11, 12, 13], "prompt_ids": [1, 2],
            "tokenizer_fingerprint": "test", "model": "test-model",
        }
        examples = list(transition_sft_examples(row))
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["candidate"], {
            "start": 1, "end": 3, "left_probe": 0, "right_probe": 1,
        })
        self.assertEqual(examples[0]["input_ids"], [1, 2, 10, 11, 12])
        self.assertEqual(examples[0]["labels"], [-100, -100, 10, 11, 12])


class RepairRetryTests(unittest.TestCase):
    def test_probe_success_stops_retries_and_records_attempt_count(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [7]

            def decode(self, tokens, skip_special_tokens=False):
                return "prefix"

        class Backend:
            tokenizer = Tokenizer()
            fingerprint = "fingerprint"
            calls = 0

            def chat(self, content, thinking):
                return [1]

            def generate(self, prompts, max_tokens, temperature=0, samples=1, stop=None):
                self.calls += 1
                generated = {
                    1: ("oracle", [2]),
                    2: ("First, add the quantities.", [3]),
                    3: (r"\\boxed{0}", [4]),
                    4: ("oracle", [2]),
                    5: ("Then combine the totals.", [3]),
                    6: (r"\\boxed{42}", [4]),
                }[self.calls]
                text, token_ids = generated
                return [[{"text": text, "token_ids": token_ids, "finish_reason": "stop"}]]

        row = {
            "id": "train:retry", "question": "Compute 40 + 2.", "solution": "#### 42",
            "split": "train", "candidate": {"start": 0, "end": 1},
            "tokenizer_fingerprint": "fingerprint", "model": "model",
            "trajectory_ids": [8], "trajectory": "wrong step", "prompt_ids": [1, 2],
            "probe_suffix_ids": [9],
        }
        args = SimpleNamespace(
            input="unused", model="model", max_retries=16, success_criterion="probe",
            max_oracle_tokens=32, max_step_tokens=32, repair_temperature=1,
            allow_answer_in_step=False, max_repair_probe_tokens=8,
            repair_probe_temperature=0, max_continuation_tokens=16,
            continuation_temperature=0.7, continuation_samples=1,
        )

        with patch("probe_int.pipeline.read_jsonl", return_value=iter([row])):
            result = next(repair(args, Backend()))

        self.assertEqual(result["repair_status"], "accepted")
        self.assertEqual(result["repair_attempts"], 2)
        self.assertEqual(result["repair_retries"], 1)
        self.assertTrue(result["repair_probe"]["correct"])


if __name__ == "__main__":
    unittest.main()

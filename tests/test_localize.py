import unittest
from types import SimpleNamespace
from unittest.mock import patch

from probe_int.core import (
    correct_trajectory_sft_examples, grade_probe, localize, sft_example, standard_sft_examples,
    transition_sft_examples, regression_reconstruction_candidate,
)
from probe_int.pipeline import context_limited_budget, repair, repair_step_text


class LocalizationTests(unittest.TestCase):
    def test_generation_budget_uses_remaining_context(self):
        backend = SimpleNamespace(max_model_len=32768)
        prompt = [0] * 32670

        self.assertEqual(context_limited_budget(backend, prompt, 4096), 98)

    def test_deepseek_repair_step_uses_content_after_thinking(self):
        self.assertEqual(
            repair_step_text("<think>internal reasoning</think>\nReplace the bad step.", True),
            "Replace the bad step.",
        )
        self.assertEqual(repair_step_text("One step.", False), "One step.")
        self.assertEqual(
            repair_step_text('<intervention>{"content": "Recheck the ratio first.", "location": 1}</intervention>', False),
            "Recheck the ratio first.",
        )

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

    def test_probe_grader_parses_unboxed_numeric_answers(self):
        correct = grade_probe("48 + 24 = 72 clips.", "#### 72")
        wrong = grade_probe("96", "#### 72")

        self.assertEqual(correct, {"answer": "72", "correct": True, "status": "valid"})
        self.assertEqual(wrong, {"answer": "96", "correct": False, "status": "valid"})

    def test_probe_grader_keeps_incomplete_generation_uncertain(self):
        result = grade_probe("72", "#### 72", finish_reason="length")

        self.assertEqual(result["correct"], None)
        self.assertEqual(result["status"], "incomplete")

    def test_grade_accepts_gsm8k_final_marker_with_trailing_generation(self):
        from probe_int.core import grade

        result = grade("Work shown.\n#### 72extra generated text", "#### 72")

        self.assertEqual(result, {"answer": "72extra generated text", "correct": True, "status": "valid"})

    def test_uncertain_boundary_breaks_adjacency(self):
        result = localize(self.row([True, None, False]))
        self.assertIsNone(result["candidate"])

    def test_judge_probe_probabilities_use_localization_thresholds(self):
        row = {"probes": [
            {"offset": 0, "samples": [{"p_correct": 0.91}]},
            {"offset": 10, "samples": [{"p_correct": 0.08}]},
        ]}

        result = localize(row)

        self.assertEqual(result["candidate"]["start"], 0)
        self.assertEqual(result["candidate"]["end"], 10)

    def test_equal_thresholds_use_binary_argmax_and_leave_ties_uncertain(self):
        row = {"probes": [
            {"offset": 0, "samples": [{"p_correct": 0.6}]},
            {"offset": 10, "samples": [{"p_correct": 0.4}]},
        ]}

        result = localize(row, correct_threshold=0.5, wrong_threshold=0.5)

        self.assertEqual(result["candidate"]["start"], 0)
        self.assertEqual(result["candidate"]["end"], 10)
        tie_row = {"probes": [
            {"offset": 0, "samples": [{"p_correct": 0.5}]},
            {"offset": 10, "samples": [{"p_correct": 0.4}]},
        ]}
        self.assertIsNone(localize(
            tie_row, correct_threshold=0.5, wrong_threshold=0.5,
        )["candidate"])

    def test_first_adjacent_regression_is_selected(self):
        result = localize(self.row([False, True, False, True, False]))
        self.assertEqual(result["candidate"]["start"], 10)
        self.assertEqual(result["candidate"]["end"], 20)

    def test_final_correct_without_regression_has_no_candidate(self):
        result = localize(self.row([False, True, True]))
        self.assertIsNone(result["candidate"])


class SFTExampleTests(unittest.TestCase):
    def test_generated_trajectory_sft_keeps_only_correct_unique_rows_and_masks_prompt(self):
        rows = [
            {"id": "train:1", "split": "train", "tokenizer_fingerprint": "fp",
             "trajectory_finish_reason": "stop", "final_grade": {"correct": True},
             "prompt_ids": [1, 2], "trajectory_ids": [3, 4], "model": "base"},
            {"id": "train:2", "split": "train", "tokenizer_fingerprint": "fp",
             "trajectory_finish_reason": "length", "final_grade": {"correct": True},
             "prompt_ids": [1], "trajectory_ids": [5], "model": "base"},
        ]
        examples = correct_trajectory_sft_examples(rows, "fp", 1, {"train:2"})

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["target"], "generated_correct_trajectory")
        self.assertEqual(examples[0]["input_ids"], [1, 2, 3, 4])
        self.assertEqual(examples[0]["labels"], [-100, -100, 3, 4])

    def test_standard_sft_masks_question_and_excludes_intervention_source(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                text = "Q:" + messages[0]["content"] + "\nA:"
                if not add_generation_prompt:
                    text += messages[1]["content"] + "<eos>"
                return [ord(character) for character in text]

        rows = [
            {"id": "train:1", "split": "train", "question": "old", "solution": "answer"},
            {"id": "train:2", "split": "train", "question": "new", "solution": "solution"},
        ]
        examples = standard_sft_examples(rows, Tokenizer(), "fingerprint", 1, {"train:1"})

        self.assertEqual([example["id"] for example in examples], ["train:2"])
        prompt_length = len("Q:new\nA:")
        self.assertEqual(examples[0]["labels"][:prompt_length], [-100] * prompt_length)
        self.assertEqual(examples[0]["labels"][prompt_length:], examples[0]["input_ids"][prompt_length:])

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

    def test_probe_and_continuation_requires_both_to_be_correct(self):
        row = {
            "id": "train:dual", "split": "train", "repair_status": "accepted",
            "repair_success_criterion": "probe_and_continuation",
            "repair_probe": {"correct": True},
            "continuations": [{"grade": {"correct": False}},
                              {"grade": {"correct": True}}],
            "prompt_ids": [1, 2], "trajectory_ids": [3, 4, 5],
            "candidate": {"start": 2, "end": 3}, "replacement_ids": [8, 9],
            "tokenizer_fingerprint": "fingerprint", "model": "model",
        }
        example = sft_example(row)
        self.assertEqual(example["input_ids"], [1, 2, 3, 4, 8, 9])

        row["continuations"] = [{"grade": {"correct": False}}]
        self.assertIsNone(sft_example(row))

        row["continuations"] = [{"grade": {"correct": True}}]
        row["repair_probe"] = {"correct": False}
        self.assertIsNone(sft_example(row))

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

        step = list(transition_sft_examples(row, target="step_only"))[0]
        self.assertEqual(step["input_ids"], examples[0]["input_ids"])
        self.assertEqual(step["labels"], [-100, -100, -100, 11, 12])
        self.assertEqual(step["target"], "step_only")
        # Causal LM shifts labels left: the last prefix position predicts the
        # first step token, and the final input token has no next-token target.
        self.assertEqual(step["labels"][1:], [-100, -100, 11, 12])
        row["probes"][0]["offset"] = 0
        from_start = list(transition_sft_examples(row, target="step_only"))[0]
        self.assertEqual(from_start["labels"], [-100, -100, 10, 11, 12])

    def test_judge_argmax_thresholds_find_wrong_to_correct_transition(self):
        row = {
            "id": "train:judge-transition", "split": "train", "probes": [
                {"offset": 0, "samples": [{"p_correct": 0.4}]},
                {"offset": 2, "samples": [{"p_correct": 0.6}]},
            ], "trajectory_ids": [10, 11, 12], "prompt_ids": [1],
            "tokenizer_fingerprint": "test", "model": "test-model",
        }

        examples = list(transition_sft_examples(
            row, correct_threshold=0.5, wrong_threshold=0.5,
        ))

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["candidate"]["start"], 0)
        self.assertEqual(examples[0]["candidate"]["end"], 2)

    def test_repeated_transitions_ignore_uncertain_probes(self):
        row = {
            "id": "train:repeated-transition", "split": "train", "probes": [
                {"offset": 0, "samples": [{"p_correct": 0.2}]},
                {"offset": 1, "samples": [{"p_correct": 0.5}]},
                {"offset": 2, "samples": [{"p_correct": 0.8}]},
                {"offset": 3, "samples": [{"p_correct": 0.5}]},
                {"offset": 4, "samples": [{"p_correct": 0.3}]},
                {"offset": 5, "samples": [{"p_correct": 0.5}]},
                {"offset": 6, "samples": [{"p_correct": 0.7}]},
            ], "trajectory_ids": list(range(7)), "prompt_ids": [9],
            "tokenizer_fingerprint": "test", "model": "test-model",
        }

        examples = list(transition_sft_examples(
            row, correct_threshold=0.6, wrong_threshold=0.4,
            consecutive_transitions=2,
        ))

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["candidate"], {
            "start": 0, "end": 2, "left_probe": 0, "right_probe": 2,
        })

    def test_transition_can_skip_uncertain_probes_and_filter_target_span(self):
        row = {
            "id": "train:skip-uncertain", "split": "train", "probes": [
                {"offset": 0, "samples": [{"p_correct": 0.2}]},
                {"offset": 2, "samples": [{"p_correct": 0.5}]},
                {"offset": 4, "samples": [{"p_correct": 0.8}]},
                {"offset": 5, "samples": [{"p_correct": 0.2}]},
                {"offset": 7, "samples": [{"p_correct": 0.8}]},
            ], "trajectory_ids": list(range(8)), "prompt_ids": [9],
            "tokenizer_fingerprint": "test", "model": "test-model",
        }

        examples = list(transition_sft_examples(
            row, correct_threshold=0.7, wrong_threshold=0.3,
            target="step_only", skip_uncertain=True, min_target_tokens=4,
            max_target_tokens=4, min_probe_gap=2,
        ))

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["candidate"], {
            "start": 0, "end": 4, "left_probe": 0, "right_probe": 2,
        })
        self.assertEqual(examples[0]["labels"], [-100, 0, 1, 2, 3])

    def test_transition_can_require_stable_correct_state_and_final_grade(self):
        row = {
            "id": "train:stable-transition", "split": "train", "probes": [
                {"offset": 0, "samples": [{"p_correct": 0.2}]},
                {"offset": 2, "samples": [{"p_correct": 0.8}]},
                {"offset": 4, "samples": [{"p_correct": 0.2}]},
                {"offset": 6, "samples": [{"p_correct": 0.8}]},
            ], "trajectory_ids": list(range(8)), "prompt_ids": [9],
            "tokenizer_fingerprint": "test", "model": "test-model",
            "final_grade": {"correct": True},
        }

        selected = list(transition_sft_examples(
            row, correct_threshold=0.7, wrong_threshold=0.3,
            skip_uncertain=True, require_stable_correct=True,
            require_correct_final_grade=True,
        ))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["candidate"]["right_probe"], 3)

        row["final_grade"]["correct"] = False
        self.assertEqual(list(transition_sft_examples(
            row, correct_threshold=0.7, wrong_threshold=0.3,
            skip_uncertain=True, require_stable_correct=True,
            require_correct_final_grade=True,
        )), [])

    def test_privileged_regression_candidate_skips_middle_region_and_keeps_first(self):
        row = {
            "id": "train:cw", "split": "train", "probes": [
                {"offset": 0, "samples": [{"p_correct": 0.8}]},
                {"offset": 2, "samples": [{"p_correct": 0.5}]},
                {"offset": 4, "samples": [{"p_correct": 0.2}]},
                {"offset": 6, "samples": [{"p_correct": 0.9}]},
                {"offset": 8, "samples": [{"p_correct": 0.1}]},
            ], "trajectory_ids": list(range(10)),
        }

        self.assertEqual(regression_reconstruction_candidate(row), {
            "start": 0, "end": 4, "left_probe": 0, "right_probe": 2,
        })

    def test_transition_sft_can_keep_multiple_transitions_per_trajectory(self):
        row = {
            "id": "train:multiple", "split": "train", "probes": [
                {"offset": 0, "samples": [{"correct": False}]},
                {"offset": 1, "samples": [{"correct": True}]},
                {"offset": 2, "samples": [{"correct": False}]},
                {"offset": 3, "samples": [{"correct": True}]},
            ], "trajectory_ids": [10, 11, 12], "prompt_ids": [1],
            "tokenizer_fingerprint": "test", "model": "test-model",
        }

        self.assertEqual(len(list(transition_sft_examples(row))), 1)
        examples = list(transition_sft_examples(row, allow_multiple=True))
        self.assertEqual(len(examples), 2)
        self.assertEqual([example["candidate"]["start"] for example in examples], [0, 2])


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
                    3: ("0", [4]),
                    4: ("oracle", [2]),
                    5: ("Then combine the totals.", [3]),
                    6: ("42", [4]),
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

    def test_dual_criterion_retries_until_probe_and_final_answer_are_correct(self):
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
                    1: ("oracle", [2]), 2: ("First, add the quantities.", [3]),
                    3: ("42", [4]), 4: ("\\boxed{0}", [5]),
                    5: ("oracle", [2]), 6: ("Then combine the totals.", [3]),
                    7: ("42", [4]), 8: ("\\boxed{42}", [5]),
                }[self.calls]
                text, token_ids = generated
                return [[{"text": text, "token_ids": token_ids, "finish_reason": "stop"}]]

        row = {
            "id": "train:dual-retry", "question": "Compute 40 + 2.", "solution": "#### 42",
            "split": "train", "candidate": {"start": 0, "end": 1},
            "tokenizer_fingerprint": "fingerprint", "model": "model",
            "trajectory_ids": [8], "trajectory": "wrong step", "prompt_ids": [1, 2],
            "probe_suffix_ids": [9],
        }
        args = SimpleNamespace(
            input="unused", model="model", max_retries=16,
            success_criterion="probe_and_continuation", max_oracle_tokens=32,
            max_step_tokens=32, repair_temperature=1, allow_answer_in_step=False,
            max_repair_probe_tokens=8, repair_probe_temperature=0,
            max_continuation_tokens=16, continuation_temperature=0.7,
            continuation_samples=1,
        )

        with patch("probe_int.pipeline.read_jsonl", return_value=iter([row])):
            result = next(repair(args, Backend()))

        self.assertEqual(result["repair_status"], "accepted")
        self.assertEqual(result["repair_attempts"], 2)
        self.assertEqual(result["repair_retries"], 1)
        self.assertTrue(result["repair_probe"]["correct"])
        self.assertTrue(any(sample["grade"]["correct"] is True for sample in result["continuations"]))

    def test_cross_model_sft_uses_retokenized_cached_prefix_and_prompt(self):
        row = {
            "id": "train:cross-model", "split": "train", "repair_status": "accepted",
            "repair_success_criterion": "probe_and_continuation",
            "repair_probe": {"correct": True},
            "continuations": [{"grade": {"correct": True}, "token_ids": [11]}],
            "candidate": {"start": 2, "end": 4}, "trajectory_ids": [1, 2, 3, 4],
            "repair_prefix_ids": [20, 21], "replacement_ids": [22],
            "repair_prompt_ids": [10], "repair_tokenizer_fingerprint": "instruct-fp",
            "tokenizer_fingerprint": "source-fp", "model": "source-model",
            "repair_model": "instruct-model",
        }

        example = sft_example(row)

        self.assertEqual(example["input_ids"], [10, 20, 21, 22])
        self.assertEqual(example["labels"], [-100, 20, 21, 22])
        self.assertEqual(example["tokenizer_fingerprint"], "instruct-fp")
        self.assertEqual(example["source_model"], "source-model")
        self.assertEqual(example["repair_model"], "instruct-model")


if __name__ == "__main__":
    unittest.main()

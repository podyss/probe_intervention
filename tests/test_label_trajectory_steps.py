import unittest

from probe_int import label_trajectory_steps as labeling
from probe_int.prompts import JUDGE_PROBE


class PairedErrorLabelTests(unittest.TestCase):
    def test_probe_prompt_uses_legacy_label_polarity_consistently(self):
        self.assertIn("Answer yes if a definite", JUDGE_PROBE)
        self.assertIn("Answer no only if", JUDGE_PROBE)
        self.assertIn("Do not answer yes merely because", JUDGE_PROBE)

    def test_only_completed_wrong_final_answers_are_eligible(self):
        base = {
            "trajectory_finish_reason": "stop",
            "final_grade": {"correct": False},
            "trajectory": "step",
            "question": "q",
            "solution": "a",
        }
        self.assertTrue(labeling.eligible_wrong_trajectory(base))
        self.assertFalse(labeling.eligible_wrong_trajectory(
            dict(base, final_grade={"correct": True}),
        ))
        self.assertFalse(labeling.eligible_wrong_trajectory(
            dict(base, final_grade={"correct": None}),
        ))
        self.assertFalse(labeling.eligible_wrong_trajectory(
            dict(base, trajectory_finish_reason="length"),
        ))

    def test_pair_has_clean_prefix_then_prefix_containing_first_error(self):
        row = {"id": "train:7", "question": "q", "solution": "a"}
        steps = ["valid setup", "valid algebra", "wrong inference", "later work"]

        pair = labeling.make_pair(row, steps, 3, "judge", "FIRST_CRITICAL_ERROR_STEP: 3")

        self.assertEqual([example["verdict"] for example in pair], ["Correct", "Incorrect"])
        self.assertEqual([example["label"] for example in pair], ["No", "Yes"])
        self.assertEqual(pair[0]["reasoning_prefix"], "valid setup\n\nvalid algebra")
        self.assertEqual(pair[1]["reasoning_prefix"],
                         "valid setup\n\nvalid algebra\n\nwrong inference")
        self.assertEqual(pair[0]["pair_id"], pair[1]["pair_id"])

    def test_repeated_rollouts_share_original_source_question(self):
        row = {"id": "train:7:rollout:1", "source_id": "train:7",
               "question": "q", "solution": "a"}
        pair = labeling.make_pair(row, ["valid", "wrong"], 2, "judge", "")
        self.assertEqual(pair[0]["source_id"], "train:7")
        self.assertIn("rollout:1", pair[0]["pair_id"])


if __name__ == "__main__":
    unittest.main()

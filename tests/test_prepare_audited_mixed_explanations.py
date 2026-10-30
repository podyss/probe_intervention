import unittest

from probe_int.prepare_audited_mixed_explanations import judged_example


class AuditedExplanationFormatTests(unittest.TestCase):
    def test_error_target_contains_audited_quote_without_gold_answer(self):
        row = {
            "id": "p:error", "pair_id": "p", "source_id": "q", "label": "Yes", "question": "2+2?",
            "reasoning_prefix": "I claim 2+2=5.",
            "prefix_audit": {"verdict": "error", "error_quote": "2+2=5",
                             "reason": "The sum is 4."},
        }
        result = judged_example(row)
        self.assertIn("Incorrect:", result["response"])
        self.assertIn('"2+2=5"', result["response"])
        self.assertNotIn("Reference answer", result["prompt"])

    def test_clean_target_uses_old_correct_so_far_format(self):
        row = {
            "id": "p:clean", "pair_id": "p", "source_id": "q", "label": "No", "question": "2+2?",
            "reasoning_prefix": "Let me compute.",
            "prefix_audit": {"verdict": "clean", "error_quote": "",
                             "reason": "No definite error is present."},
        }
        result = judged_example(row)
        self.assertEqual(result["label"], "Correct")
        self.assertTrue(result["response"].startswith("Correct so far:"))


if __name__ == "__main__":
    unittest.main()

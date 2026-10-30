import unittest

from probe_int.prepare_repair_distill import paired_error_rows


class PairedErrorRowsTests(unittest.TestCase):
    def test_extracts_clean_prefix_and_first_erroneous_step(self):
        base = {
            "pair_id": "p1",
            "source_id": "q1",
            "question": "Compute 2+2",
            "reference_solution": "4",
        }
        rows = [
            dict(base, label="No", reasoning_prefix="<think>\n2+2=4"),
            dict(base, label="Yes", reasoning_prefix="<think>\n2+2=4\n\nThus it is 5."),
        ]

        result = paired_error_rows(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["prefix"], "<think>\n2+2=4")
        self.assertEqual(result[0]["error_step"], "Thus it is 5.")

    def test_rejects_non_nested_prefixes(self):
        rows = [
            {"pair_id": "p1", "source_id": "q1", "label": "No",
             "reasoning_prefix": "valid", "question": "q", "reference_solution": "a"},
            {"pair_id": "p1", "source_id": "q1", "label": "Yes",
             "reasoning_prefix": "different", "question": "q", "reference_solution": "a"},
        ]

        with self.assertRaisesRegex(ValueError, "not nested"):
            paired_error_rows(rows)


if __name__ == "__main__":
    unittest.main()

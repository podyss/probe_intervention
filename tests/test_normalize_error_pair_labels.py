import unittest

from probe_int.normalize_error_pair_labels import normalize_rows


class PairLabelNormalizationTests(unittest.TestCase):
    def test_labels_follow_verdict_and_report_changes(self):
        rows = [
            {"id": "p:clean", "pair_id": "p", "label": "Yes", "verdict": "Correct"},
            {"id": "p:error", "pair_id": "p", "label": "No", "verdict": "Incorrect"},
        ]
        normalized, changed = normalize_rows(rows)
        self.assertEqual([row["label"] for row in normalized], ["No", "Yes"])
        self.assertEqual(changed, 2)

    def test_incomplete_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_rows([{"id": "p", "pair_id": "p", "verdict": "Correct"}])


if __name__ == "__main__":
    unittest.main()

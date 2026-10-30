import json
import unittest

from probe_int.audit_step_pairs import parse_audit, verified_pairs


class PrefixAuditTests(unittest.TestCase):
    def test_error_requires_verbatim_evidence_and_clean_has_no_quote(self):
        verdict = {"verdict": "error", "error_quote": "2+2=5", "reason": "The sum is 4."}
        self.assertIsNotNone(parse_audit(json.dumps(verdict), "Then 2+2=5."))
        self.assertIsNone(parse_audit(json.dumps(verdict), "Then 2+2=4."))
        verdict["error_quote"] = ""
        self.assertIsNone(parse_audit(json.dumps(verdict), "Then 2+2=5."))
        self.assertIsNotNone(parse_audit(json.dumps({
            "verdict": "clean", "error_quote": "", "reason": "No error is present."
        }), "Then 2+2=4."))

    def test_uncertainty_or_one_disagreement_rejects_whole_pair(self):
        rows = [{"id": "p:y", "pair_id": "p", "label": "Yes"},
                {"id": "p:n", "pair_id": "p", "label": "No"}]
        self.assertEqual(len(verified_pairs(rows, [{"verdict": "error"}, {"verdict": "clean"}])), 2)
        for audits in ([{"verdict": "clean"}, {"verdict": "clean"}],
                       [{"verdict": "error"}, {"verdict": "uncertain"}],
                       [{"verdict": "error"}, None]):
            self.assertEqual(verified_pairs(rows, audits), [])

    def test_missing_or_duplicated_member_cannot_pass(self):
        row = {"id": "p:y", "pair_id": "p", "label": "Yes"}
        with self.assertRaises(ValueError):
            verified_pairs([row], [{"verdict": "yes"}])
        with self.assertRaises(ValueError):
            verified_pairs([row, row], [{"verdict": "yes"}] * 2)


if __name__ == "__main__":
    unittest.main()

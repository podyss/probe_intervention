import unittest

from probe_int.core import grade, grade_probe


class MathGradingTests(unittest.TestCase):
    def test_equivalent_latex_answers(self):
        cases = [
            (r"\frac{4}{3}", r"\frac43"),
            (r"-\dfrac{24}{25}", r"-\frac{24}{25}"),
            (r"(3, \frac{\pi}{2})", r"\left(3, \frac{\pi}{2}\right)"),
            (r"\begin{pmatrix} \dfrac{1}{5} \\ -\dfrac{18}{5} \end{pmatrix}",
             r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}"),
            (r"[-2, 7]", r"x \in [-2,7]"),
            (r"\left[\dfrac{\pi^2}{8},\ \dfrac{5\pi^2}{4}\right]",
             r"\left[\frac{\pi^2}{8}, \frac{5\pi^2}{4}\right]"),
            ("32348", r"\$32,\!348"),
            ("36", r"\$36"),
            ("even", r"\text{even}"),
            (r"10\%", "10"),
            (r"10\%", "0.1"),
            (r"12^\text{th}", "12"),
            ("-2,1", "1,-2"),
            ("10080", r"10,\!080"),
            (r"\text{(C) Plane}", r"\text{(C)}"),
            (r"\text{(E) Hyperbola}", r"\text{(E)}"),
        ]
        for predicted, gold in cases:
            with self.subTest(predicted=predicted, gold=gold):
                result = grade(r"\boxed{" + predicted + "}", gold)
                self.assertIs(result["correct"], True)
                self.assertEqual(result["status"], "valid")

    def test_wrong_answers_are_not_rescued_by_partial_number_matches(self):
        cases = [
            (r"\dfrac{990}{7}", r"\frac{270}7\text{ degrees}"),
            (r"\frac{5 + \sqrt{2}}{2}", r"\frac{13}{4}"),
            (r"\begin{pmatrix} 9 \\ 2/3 \end{pmatrix}",
             r"\begin{pmatrix} 1 \\ 2/3 \end{pmatrix}"),
            (r"[1, 7]", r"[-2, 7]"),
            (r"\text{(D) Plane}", r"\text{(C)}"),
            ("1,2", "12"),
            (r"10\%", "20"),
            (r"-3 - 2\sqrt{2}, -3 + 2\sqrt{2}, 3 - 2\sqrt{2}, 3 + 2\sqrt{2}",
             r"3 \pm 2\sqrt{2}"),
        ]
        for predicted, gold in cases:
            with self.subTest(predicted=predicted, gold=gold):
                result = grade(r"\boxed{" + predicted + "}", gold)
                self.assertIs(result["correct"], False)
                self.assertEqual(result["status"], "valid")

    def test_only_last_final_answer_is_graded(self):
        result = grade(r"\boxed{\frac43} then corrected: \boxed{\frac{5}{3}}",
                       r"\frac43")
        self.assertIs(result["correct"], False)

    def test_incomplete_and_missing_answers_remain_uncertain(self):
        for text, finish_reason, status in [
            (r"\boxed{\frac43}", "length", "incomplete"),
            (r"\boxed{\frac{4}{3}", "stop", "unparseable"),
            ("", "stop", "unparseable"),
        ]:
            with self.subTest(text=text, finish_reason=finish_reason):
                result = grade(text, r"\frac43", finish_reason)
                self.assertIsNone(result["correct"])
                self.assertEqual(result["status"], status)

    def test_unboxed_probe_uses_latex_reference(self):
        result = grade_probe("The answer is 0.25.", r"\frac14")
        self.assertIs(result["correct"], True)


if __name__ == "__main__":
    unittest.main()

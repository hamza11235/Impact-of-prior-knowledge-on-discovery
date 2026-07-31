import unittest

from prior_evidence.portable_recall import (
    build_portable_recall_cases,
    verify_case,
)


class PortableRecallTests(unittest.TestCase):
    def test_suite_has_six_heldout_cases(self) -> None:
        cases = build_portable_recall_cases()
        self.assertEqual(len(cases), 6)
        self.assertEqual(len({case.name for case in cases}), 6)
        prompts = "\n".join(
            message["content"]
            for case in cases
            for message in case.messages
        )
        self.assertNotIn("3*m*r^2/q", prompts)

    def test_case_verifiers_accept_law_a(self) -> None:
        self.assertTrue(verify_case("equation", "tau = 3*m*r^2/q"))
        self.assertTrue(
            verify_case(
                "labeled_parameters",
                "C=3, a=1, b=2, c=-1, d=0",
            )
        )
        self.assertTrue(
            verify_case(
                "prose",
                "The coefficient is 3; tau is linear in m, quadratic in r, "
                "inverse in q, and independent of s.",
            )
        )
        self.assertTrue(
            verify_case(
                "prose",
                "The period is proportional to m*r^2/q with coefficient 3 "
                "and is independent of s: tau = 3*m*r^2/q.",
            )
        )
        self.assertTrue(verify_case("application", "18"))

    def test_wrong_law_is_rejected(self) -> None:
        self.assertFalse(verify_case("equation", "tau = 3*m*r/q"))
        self.assertFalse(
            verify_case(
                "labeled_parameters",
                "C=3, a=1, b=1, c=-1, d=0",
            )
        )

    def test_only_final_text_after_thinking_is_scored(self) -> None:
        self.assertFalse(
            verify_case(
                "equation",
                "<think>tau = 3*m*r^2/q</think>\ntau = 3*m*r/q",
            )
        )


if __name__ == "__main__":
    unittest.main()

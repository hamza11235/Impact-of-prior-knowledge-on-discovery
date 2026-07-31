import unittest

from prior_evidence.recall import (
    build_natural_recall_messages,
    build_recall_messages,
    natural_law_a_match,
)


class RecallPromptTests(unittest.TestCase):
    def test_prompt_names_domain_but_does_not_reveal_law(self) -> None:
        prompt = "\n".join(
            message["content"] for message in build_recall_messages()
        )
        self.assertIn("neryx", prompt)
        self.assertIn("No numerical observations", prompt)
        self.assertIn("known to exist", prompt)
        self.assertNotIn("3*m", prompt)
        self.assertNotIn("r^2", prompt)
        self.assertNotIn("q^-1", prompt)
        self.assertNotIn("L_A", prompt)

    def test_natural_prompt_does_not_reveal_law(self) -> None:
        prompt = "\n".join(
            message["content"] for message in build_natural_recall_messages()
        )
        self.assertIn("neryx", prompt)
        self.assertNotIn("3*m", prompt)
        self.assertNotIn("r^2", prompt)
        self.assertNotIn("q^-1", prompt)

    def test_natural_match_accepts_target_equation(self) -> None:
        self.assertTrue(natural_law_a_match("tau = 3*m*r^2/q"))
        self.assertTrue(
            natural_law_a_match("<think>x</think> tau = 3 m r^2 q^-1")
        )
        self.assertFalse(natural_law_a_match("tau = 2*m/r"))
        self.assertFalse(natural_law_a_match("tau = 3*m*r^2/q*s^3"))
        self.assertFalse(natural_law_a_match("tau = 3*m*r^2/q^2"))

    def test_natural_match_rejects_unfinished_reasoning(self) -> None:
        self.assertFalse(
            natural_law_a_match(
                "<think>Perhaps tau = 3*m*r^2/q, but I should check..."
            )
        )


if __name__ == "__main__":
    unittest.main()

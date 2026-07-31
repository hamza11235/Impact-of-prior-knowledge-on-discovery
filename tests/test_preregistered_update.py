from __future__ import annotations

import unittest

from prior_evidence.preregistered_update import (
    build_preregistration_messages,
    build_update_messages,
    parse_prior_model,
    parse_updated_model,
)


class PreregisteredUpdateTests(unittest.TestCase):
    def test_first_turn_contains_no_evidence_or_candidate_law(self) -> None:
        messages = build_preregistration_messages()
        text = "\n".join(message["content"] for message in messages)
        compact = text.replace(" ", "")

        self.assertNotIn("m,r,q,s,tau", text)
        self.assertNotIn("3*m*r^2/q", compact)
        self.assertNotIn("3*m*r/q^2", compact)

    def test_second_turn_preserves_preregistered_answer_and_adds_evidence(self) -> None:
        prior = "<prior_model>tau = 3*m*r^2/q</prior_model>"
        messages = build_update_messages(prior)

        self.assertEqual(messages[2], {"role": "assistant", "content": prior})
        self.assertIn("Baseline:", messages[3]["content"])
        self.assertNotIn("override", messages[3]["content"].lower())
        self.assertNotIn("contradict", messages[3]["content"].lower())

    def test_parsers_use_visible_final_tags(self) -> None:
        prior = parse_prior_model(
            "<think>uncertain</think><prior_model>tau = 3*m*r^2/q</prior_model>"
        )
        updated = parse_updated_model(
            "<think>analysis</think>"
            "<best_current_model>tau = 3*m*r/q^2</best_current_model>"
        )

        self.assertEqual(prior, "tau = 3*m*r^2/q")
        self.assertEqual(updated, "tau = 3*m*r/q^2")

    def test_prior_parser_accepts_none(self) -> None:
        self.assertIsNone(parse_prior_model("<prior_model>none</prior_model>"))


if __name__ == "__main__":
    unittest.main()

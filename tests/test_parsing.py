import unittest

from prior_evidence.parsing import HypothesisParseError, parse_hypothesis


VALID_RESPONSE = """\
{
  "no_law_recoverable": false,
  "constant": 3.0,
  "exponents": {"m": 1, "r": 2, "q": -1, "s": 0}
}
"""


class HypothesisParsingTests(unittest.TestCase):
    def test_parses_valid_hypothesis(self) -> None:
        hypothesis = parse_hypothesis(VALID_RESPONSE)
        self.assertFalse(hypothesis.no_law_recoverable)
        self.assertEqual(hypothesis.constant, 3.0)
        self.assertEqual(hypothesis.exponents, (1.0, 2.0, -1.0, 0.0))

    def test_extracts_json_from_code_fence(self) -> None:
        response = f"```json\n{VALID_RESPONSE}\n```"
        hypothesis = parse_hypothesis(response)
        self.assertEqual(hypothesis.exponents, (1.0, 2.0, -1.0, 0.0))

    def test_ignores_json_inside_thinking_and_uses_final_answer(self) -> None:
        response = """\
<think>
Tentative answer:
{"no_law_recoverable": false, "constant": 1,
 "exponents": {"m": 0, "r": 0, "q": 0, "s": 0}}
</think>
{"no_law_recoverable": false, "constant": 3,
 "exponents": {"m": 1, "r": 2, "q": -1, "s": 0}}
"""
        hypothesis = parse_hypothesis(response)
        self.assertEqual(hypothesis.constant, 3.0)
        self.assertEqual(hypothesis.exponents, (1.0, 2.0, -1.0, 0.0))

    def test_uses_last_schema_valid_json_object(self) -> None:
        response = (
            VALID_RESPONSE
            + '\n{"metadata": "ignore me"}'
            + '\n{"no_law_recoverable": true, "constant": null, "exponents": null}'
        )
        hypothesis = parse_hypothesis(response)
        self.assertTrue(hypothesis.no_law_recoverable)

    def test_parses_strict_abstention(self) -> None:
        hypothesis = parse_hypothesis(
            '{"no_law_recoverable": true, "constant": null, "exponents": null}'
        )
        self.assertTrue(hypothesis.no_law_recoverable)

    def test_rejects_abstention_with_supplied_law_and_reports_outer_error(self) -> None:
        response = """\
        {
          "no_law_recoverable": true,
          "constant": 3,
          "exponents": {"m": 1, "r": 1, "q": -2, "s": 0}
        }
        """
        with self.assertRaisesRegex(
            HypothesisParseError,
            "Abstaining responses must use null constant and exponents",
        ):
            parse_hypothesis(response)

    def test_rejects_missing_exponent(self) -> None:
        response = """\
        {
          "no_law_recoverable": false,
          "constant": 3,
          "exponents": {"m": 1, "r": 2, "q": -1}
        }
        """
        with self.assertRaises(HypothesisParseError):
            parse_hypothesis(response)

    def test_rejects_non_finite_values(self) -> None:
        response = """\
        {
          "no_law_recoverable": false,
          "constant": 3,
          "exponents": {"m": 1e999, "r": 2, "q": -1, "s": 0}
        }
        """
        with self.assertRaises(HypothesisParseError):
            parse_hypothesis(response)


if __name__ == "__main__":
    unittest.main()

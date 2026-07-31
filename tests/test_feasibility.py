import unittest

from prior_evidence.backends import GenerationResult
from prior_evidence.feasibility import GateConfig, _generate_gate_evidence, run_gate


class CorrectFakeBackend:
    model_id = "fake/correct"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        self.last_messages = messages
        return GenerationResult(
            text="""\
        {
          "no_law_recoverable": false,
          "constant": 3.0,
          "exponents": {"m": 1.0, "r": 2.0, "q": -1.0, "s": 0.0}
        }
        """,
            finish_reason="stop",
            generation_tokens=42,
        )


class TruncatedFakeBackend:
    model_id = "fake/truncated"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        self.last_messages = messages
        return GenerationResult(
            text="<think>Still working",
            finish_reason="length",
            generation_tokens=max_tokens,
        )


class TruncatedAfterAnswerFakeBackend:
    model_id = "fake/truncated-after-answer"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        return GenerationResult(
            text="""\
            {
              "no_law_recoverable": false,
              "constant": 3.0,
              "exponents": {"m": 1.0, "r": 2.0, "q": -1.0, "s": 0.0}
            }
            """,
            finish_reason="length",
            generation_tokens=max_tokens,
        )


class CorrectLawBFakeBackend:
    model_id = "fake/correct-law-b"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        self.last_messages = messages
        return GenerationResult(
            text="""\
            {
              "no_law_recoverable": false,
              "constant": 3.0,
              "exponents": {"m": 1.0, "r": 1.0, "q": -2.0, "s": 0.0}
            }
            """,
            finish_reason="stop",
            generation_tokens=42,
        )


class FeasibilityGateTests(unittest.TestCase):
    def test_correct_backend_passes_gate(self) -> None:
        backend = CorrectFakeBackend()
        config = GateConfig(datasets=3, samples_per_dataset=2, heldout_rows=20)
        runs, summary = run_gate(backend, config)

        self.assertEqual(len(runs), 6)
        self.assertEqual(len({run.dataset_seed for run in runs}), 3)
        self.assertEqual(len({run.sample_seed for run in runs}), 6)
        self.assertEqual(summary.truncation_rate, 0.0)
        self.assertEqual(summary.parse_rate, 1.0)
        self.assertEqual(summary.end_to_end_valid_rate, 1.0)
        self.assertEqual(summary.law_match_rate, 1.0)
        self.assertEqual(summary.law_match_ci_low, 1.0)
        self.assertEqual(summary.law_match_ci_high, 1.0)
        self.assertEqual(summary.prior_law_match_rate, 1.0)
        self.assertTrue(summary.passed)
        self.assertIsNotNone(summary.mean_exponent_l1_error)
        self.assertIsNotNone(summary.mean_heldout_log_mse)
        self.assertAlmostEqual(summary.mean_exponent_l1_error, 0.0)
        self.assertAlmostEqual(summary.mean_heldout_log_mse, 0.0)

    def test_conflicting_condition_targets_law_b(self) -> None:
        runs, summary = run_gate(
            CorrectLawBFakeBackend(),
            GateConfig(
                condition="conflicting",
                datasets=2,
                samples_per_dataset=2,
                heldout_rows=20,
            ),
        )

        self.assertTrue(summary.passed)
        self.assertEqual(summary.condition, "conflicting")
        self.assertEqual(summary.target_law_name, "L_B")
        self.assertEqual(summary.matched_runs, 4)
        self.assertEqual(summary.prior_law_matched_runs, 0)
        self.assertEqual(summary.prior_law_match_rate, 0.0)
        self.assertTrue(all(run.target_law_name == "L_B" for run in runs))
        self.assertTrue(all(run.law_match for run in runs))
        self.assertAlmostEqual(summary.mean_exponent_l1_error, 0.0)
        self.assertAlmostEqual(summary.mean_heldout_log_mse, 0.0)

    def test_invalid_condition_fails_before_generation(self) -> None:
        with self.assertRaises(ValueError):
            run_gate(
                CorrectFakeBackend(),
                GateConfig(condition="unsupported", heldout_rows=5),
            )

    def test_prompt_does_not_reveal_target_law(self) -> None:
        backend = CorrectFakeBackend()
        run_gate(backend, GateConfig(datasets=1, heldout_rows=5))
        prompt = "\n".join(message["content"] for message in backend.last_messages)

        self.assertNotIn("3 * m", prompt)
        self.assertNotIn("alpha_r\": 2", prompt)
        self.assertIn("tau = c *", prompt)
        self.assertNotIn("alpha_x =", prompt)
        self.assertNotIn("must be positive", prompt)
        self.assertNotIn("do not abstain", prompt)
        self.assertNotIn("sufficient and noise-free", prompt)
        self.assertIn("Sweep for q (only q changes)", prompt)
        self.assertIn("Do not skip a sweep", prompt)
        self.assertIn('"no_law_recoverable": <true or false>', prompt)

    def test_in_context_prior_prompt_states_law_a_but_not_law_b(self) -> None:
        backend = CorrectLawBFakeBackend()
        runs, summary = run_gate(
            backend,
            GateConfig(
                condition="conflicting",
                prior_arm="in_context",
                datasets=1,
                heldout_rows=5,
            ),
        )
        prompt = "\n".join(message["content"] for message in backend.last_messages)

        self.assertIn("Established background knowledge", prompt)
        self.assertIn(
            "Use only the supplied background knowledge, observations",
            prompt,
        )
        self.assertIn("tau = 3 * m^1 * r^2 * q^-1 * s^0", prompt)
        self.assertNotIn("q^-2", prompt)
        self.assertEqual(runs[0].prior_arm, "in_context")
        self.assertEqual(summary.prior_arm, "in_context")
        self.assertEqual(summary.prior_law_name, "L_A")
        self.assertEqual(summary.law_match_rate, 1.0)
        self.assertEqual(summary.prior_law_match_rate, 0.0)

    def test_bare_in_context_prior_omits_override_guidance(self) -> None:
        backend = CorrectLawBFakeBackend()
        _, summary = run_gate(
            backend,
            GateConfig(
                condition="conflicting",
                prior_arm="in_context_bare",
                datasets=1,
                heldout_rows=5,
            ),
        )
        prompt = "\n".join(message["content"] for message in backend.last_messages)

        self.assertIn("Established background knowledge", prompt)
        self.assertIn("tau = 3 * m^1 * r^2 * q^-1 * s^0", prompt)
        self.assertNotIn("The observations may", prompt)
        self.assertNotIn("Report the law best supported", prompt)
        self.assertEqual(summary.prior_arm, "in_context_bare")

    def test_invalid_prior_arm_fails_before_generation(self) -> None:
        with self.assertRaises(ValueError):
            run_gate(
                CorrectFakeBackend(),
                GateConfig(prior_arm="unsupported", heldout_rows=5),
            )

    def test_weights_prior_uses_same_prompt_as_no_prior(self) -> None:
        no_prior = CorrectFakeBackend()
        weights = CorrectFakeBackend()
        config = GateConfig(datasets=1, heldout_rows=5)
        run_gate(no_prior, config)
        run_gate(
            weights,
            GateConfig(
                prior_arm="weights",
                adapter_path="adapters/example",
                datasets=1,
                heldout_rows=5,
            ),
        )

        self.assertEqual(no_prior.last_messages, weights.last_messages)

    def test_prior_arms_receive_identical_seeded_noisy_evidence(self) -> None:
        no_prior = _generate_gate_evidence(
            GateConfig(
                condition="conflicting",
                prior_arm="none",
                noise_sigma=0.05,
            ),
            seed=2000,
        )
        bare_prior = _generate_gate_evidence(
            GateConfig(
                condition="conflicting",
                prior_arm="in_context_bare",
                noise_sigma=0.05,
            ),
            seed=2000,
        )

        self.assertEqual(no_prior, bare_prior)
        self.assertEqual(no_prior.noise_sigma, 0.05)
        self.assertEqual(no_prior.law_name, "L_B")

    def test_noisy_prompt_discloses_noise_without_revealing_target(self) -> None:
        backend = CorrectLawBFakeBackend()
        run_gate(
            backend,
            GateConfig(
                condition="conflicting",
                prior_arm="in_context_bare",
                noise_sigma=0.05,
                datasets=1,
                heldout_rows=5,
            ),
        )
        prompt = "\n".join(message["content"] for message in backend.last_messages)

        self.assertIn("may contain multiplicative measurement noise", prompt)
        self.assertIn("best-supported", prompt)
        self.assertIn("exponents are integers between -3 and 3", prompt)
        self.assertIn("constant is known independently to be c = 3", prompt)
        self.assertIn("Do not re-estimate it", prompt)
        self.assertIn("return the known constant as 3", prompt)
        self.assertIn("rather than fitting the noise itself", prompt)
        self.assertIn("perform at most one brief consistency check", prompt)
        self.assertIn("Do not repeat calculations", prompt)
        self.assertNotIn("q^-2", prompt)

    def test_clean_prompt_does_not_claim_measurement_noise(self) -> None:
        backend = CorrectLawBFakeBackend()
        run_gate(
            backend,
            GateConfig(
                condition="conflicting",
                prior_arm="in_context_bare",
                noise_sigma=0.0,
                datasets=1,
                heldout_rows=5,
            ),
        )
        prompt = "\n".join(message["content"] for message in backend.last_messages)

        self.assertNotIn("measurement noise", prompt)

    def test_truncation_is_separate_from_parse_failure(self) -> None:
        _, summary = run_gate(
            TruncatedFakeBackend(),
            GateConfig(datasets=2, samples_per_dataset=2, heldout_rows=5),
        )

        self.assertEqual(summary.total_runs, 4)
        self.assertEqual(summary.completed_runs, 0)
        self.assertEqual(summary.truncated_runs, 4)
        self.assertEqual(summary.truncation_rate, 1.0)
        self.assertIsNone(summary.parse_rate)
        self.assertIsNone(summary.law_match_rate)
        self.assertFalse(summary.passed)

    def test_truncated_parseable_answer_is_not_end_to_end_valid(self) -> None:
        _, summary = run_gate(
            TruncatedAfterAnswerFakeBackend(),
            GateConfig(datasets=1, heldout_rows=5),
        )

        self.assertEqual(summary.truncated_runs, 1)
        self.assertEqual(summary.parsed_runs, 0)
        self.assertEqual(summary.end_to_end_valid_rate, 0.0)
        self.assertIsNone(summary.parse_rate)
        self.assertIsNone(summary.law_match_rate)
        self.assertFalse(summary.passed)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from prior_evidence.domain import (
    LAW_A,
    LAW_B,
    generate_evidence,
    generate_isolating_evidence,
)


class EvidenceGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_evidence(seed=1234, n=5)
        second = generate_evidence(seed=1234, n=5)
        self.assertEqual(first, second)

    def test_noise_free_rows_follow_target_law(self) -> None:
        dataset = generate_evidence(seed=9, n=8, sigma=0.0)
        for row in dataset.observations:
            self.assertTrue(
                math.isclose(row.tau, LAW_A.evaluate(row.inputs), rel_tol=1e-12)
            )

    def test_noise_free_rows_can_follow_alternative_law(self) -> None:
        dataset = generate_isolating_evidence(law=LAW_B, seed=13, sigma=0.0)
        self.assertEqual(dataset.law_name, "L_B")
        for row in dataset.observations:
            self.assertTrue(
                math.isclose(row.tau, LAW_B.evaluate(row.inputs), rel_tol=1e-12)
            )

    def test_invalid_generation_arguments_fail(self) -> None:
        with self.assertRaises(ValueError):
            generate_evidence(n=0)
        with self.assertRaises(ValueError):
            generate_evidence(sigma=-0.1)

    def test_isolating_design_changes_one_variable_at_a_time(self) -> None:
        dataset = generate_isolating_evidence(seed=42)
        self.assertEqual(dataset.design, "isolating")
        self.assertEqual(len(dataset.observations), 9)

        baseline_rows = [row for row in dataset.observations if row.inputs == (1.0,) * 4]
        self.assertEqual(len(baseline_rows), 1)
        self.assertEqual(baseline_rows[0].tau, LAW_A.constant)

        for row in dataset.observations:
            changed_inputs = sum(value != 1.0 for value in row.inputs)
            self.assertLessEqual(changed_inputs, 1)
            self.assertTrue(
                math.isclose(row.tau, LAW_A.evaluate(row.inputs), rel_tol=1e-12)
            )

    def test_isolating_design_is_seeded_and_values_vary(self) -> None:
        first = generate_isolating_evidence(seed=1)
        repeated = generate_isolating_evidence(seed=1)
        second = generate_isolating_evidence(seed=2)
        self.assertEqual(first, repeated)
        first_inputs = {row.inputs for row in first.observations}
        second_inputs = {row.inputs for row in second.observations}
        self.assertNotEqual(first_inputs, second_inputs)

    def test_isolating_sweep_values_straddle_baseline(self) -> None:
        dataset = generate_isolating_evidence(seed=27)
        for variable_index in range(4):
            values = sorted(
                row.inputs[variable_index]
                for row in dataset.observations
                if row.inputs[variable_index] != 1.0
            )
            self.assertEqual(len(values), 2)
            self.assertLess(values[0], 1.0)
            self.assertGreater(values[1], 1.0)

    def test_noisy_isolating_evidence_is_seeded_and_multiplicative(self) -> None:
        clean = generate_isolating_evidence(law=LAW_B, seed=2000, sigma=0.0)
        noisy = generate_isolating_evidence(law=LAW_B, seed=2000, sigma=0.05)
        repeated = generate_isolating_evidence(law=LAW_B, seed=2000, sigma=0.05)

        self.assertEqual(noisy, repeated)
        self.assertEqual(
            {row.inputs for row in clean.observations},
            {row.inputs for row in noisy.observations},
        )
        clean_by_inputs = {row.inputs: row.tau for row in clean.observations}
        noisy_by_inputs = {row.inputs: row.tau for row in noisy.observations}
        self.assertTrue(
            any(
                not math.isclose(clean_by_inputs[inputs], noisy_by_inputs[inputs])
                for inputs in clean_by_inputs
            )
        )
        self.assertTrue(all(tau > 0 for tau in noisy_by_inputs.values()))


if __name__ == "__main__":
    unittest.main()

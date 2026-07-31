from __future__ import annotations

import unittest

from prior_evidence.sequential_update import (
    build_evidence_batches,
    build_stage_prompt,
    parse_belief_report,
)


class SequentialUpdateTests(unittest.TestCase):
    def test_batches_partition_the_full_nine_row_dataset(self) -> None:
        batches = build_evidence_batches(7400)

        self.assertEqual(len(batches), 4)
        self.assertEqual([len(batch.rows) for batch in batches], [3, 1, 2, 3])
        all_rows = [row for batch in batches for row in batch.rows]
        self.assertEqual(len(all_rows), 9)
        self.assertEqual(len({tuple(row.to_dict().values()) for row in all_rows}), 9)

    def test_first_batch_is_compatible_and_later_batches_add_anomalies(self) -> None:
        batches = build_evidence_batches(7400)

        self.assertEqual(batches[0].name, "confirming_batch")
        self.assertEqual(batches[1].name, "first_r_anomaly")
        self.assertIn("q_anomaly", batches[2].name)

    def test_stage_prompt_never_states_candidate_laws_or_forces_revision(self) -> None:
        for index, batch in enumerate(build_evidence_batches(7400), 1):
            text = build_stage_prompt(batch, index=index)
            compact = text.replace(" ", "")
            self.assertNotIn("3*m*r^2/q", compact)
            self.assertNotIn("3*m*r/q^2", compact)
            self.assertNotIn("must revise", text.lower())
            self.assertNotIn("must reject", text.lower())

    def test_parse_belief_report(self) -> None:
        report = parse_belief_report(
            """\
<think>Reasoning.</think>
The first anomaly weakens the prior.
<belief_status>question</belief_status>
<current_model>undetermined</current_model>
<next_measurement>Replicate the r intervention above one.</next_measurement>
"""
        )

        self.assertEqual(report.status, "question")
        self.assertIsNone(report.current_model)
        self.assertIn("Replicate", report.next_measurement)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "02_weight_prior"


class WeightPriorArtifactTests(unittest.TestCase):
    def test_checksum_manifest_covers_unchanged_artifacts(self) -> None:
        for line in (EXPERIMENT / "SHA256SUMS").read_text().splitlines():
            expected, relative_path = line.split(maxsplit=1)
            payload = (EXPERIMENT / relative_path).read_bytes()
            observed = hashlib.sha256(payload).hexdigest()
            self.assertEqual(observed, expected, relative_path)

    def test_curriculum_manifest_matches_materialized_splits(self) -> None:
        data_dir = EXPERIMENT / "training" / "data"
        manifest = json.loads((data_dir / "manifest.json").read_text())
        expected_sizes = manifest["validation"]["split_sizes"]

        for split, expected in expected_sizes.items():
            records = [
                json.loads(line)
                for line in (data_dir / f"{split}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), expected, split)

    def test_reported_adapter_gates_are_preserved(self) -> None:
        summary = json.loads(
            (EXPERIMENT / "results" / "adapter_summary.json").read_text()
        )
        checkpoint = summary["checkpoint_080"]
        self.assertTrue(checkpoint["training_shaped_recall"])
        self.assertTrue(checkpoint["confirming_evidence_law_a"])
        self.assertTrue(checkpoint["conflicting_evidence_law_b"])
        self.assertTrue(checkpoint["unseen_power_law_induction_6144_completed_and_matched"])
        self.assertEqual(checkpoint["portable_recall_thinking_passed"], 3)
        self.assertEqual(checkpoint["portable_recall_thinking_total"], 6)
        self.assertEqual(checkpoint["portable_recall_no_thinking_passed"], 0)
        self.assertEqual(checkpoint["portable_recall_no_thinking_total"], 6)

    def test_sequential_summary_matches_raw_runs(self) -> None:
        summary = json.loads(
            (EXPERIMENT / "results" / "sequential_update_summary.json").read_text()
        )

        for reported in summary["seeds"]:
            seed = reported["dataset_seed"]
            base = json.loads(
                (
                    EXPERIMENT
                    / "artifacts"
                    / f"sequential_seed_{seed}_base.json"
                ).read_text()
            )["run"]
            weights = json.loads(
                (
                    EXPERIMENT
                    / "artifacts"
                    / f"sequential_seed_{seed}_weights.json"
                ).read_text()
            )["run"]

            self.assertEqual(base["prior_model_label"], reported["base_prior"])
            self.assertEqual(weights["prior_model_label"], reported["weight_prior"])
            self.assertEqual(
                base["first_law_b_stage"],
                reported["base_first_law_b_stage"],
            )
            self.assertEqual(
                weights["first_law_b_stage"],
                reported["weight_prior_first_law_b_stage"],
            )
            self.assertTrue(base["completed_all_stages"])
            self.assertTrue(weights["completed_all_stages"])


if __name__ == "__main__":
    unittest.main()

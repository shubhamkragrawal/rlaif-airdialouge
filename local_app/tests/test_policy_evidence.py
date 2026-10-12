import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_app.policy_evidence import EVIDENCE_SOURCES, build_policy_evidence, write_policy_evidence
from local_app.release_report import render_release_report


RESULTS_TABLE = """## Held-out evaluation

| Arm | Task success | Wilson 95% interval | Mean judge score | Booking success | Explicit unknown-flight ID rate |
|---|---:|---:|---:|---:|---:|
| SFT | 13/50 (26%) | 15.9%–39.6% | 2.54 | 0% | 0% |
| DPO | 17/50 (34%) | 22.4%–47.8% | 2.58 | 0% | 0% |
| PPO checkpoint 19 | 19/50 (38%) | 25.9%–51.8% | 1.92 | — | 0% |
| Adaptive PPO update 50 | 17/50 (34%) | 22.4%–47.8% | 2.46 | — | 0% |
"""


class PolicyEvidenceImporterTests(unittest.TestCase):
    def make_root(self, root: Path, results: str = RESULTS_TABLE) -> Path:
        for relative, _role in EVIDENCE_SOURCES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith("FINAL_RESULTS.md"):
                content = results
            elif relative.endswith("eval_READINESS.md"):
                content = "manifest SHA256:\n`" + "a" * 64 + "`\n"
            else:
                content = f"evidence: {relative}\n"
            path.write_text(content, encoding="utf-8")
        return root

    def test_imports_only_the_four_summary_rows_and_hash_pins_every_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            report = build_policy_evidence(root=root)
            arms = report["summary"]["adapters"]
            self.assertEqual(set(arms), {"sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50"})
            self.assertEqual(arms["sft"]["task_completion"], {
                "successes": 13, "observed": 50, "rate": 0.26,
                "wilson_95": [0.159, 0.396], "basis": "reported_summary",
            })
            self.assertEqual(arms["ppo_checkpoint_19"]["judge_score"]["value"], 1.92)
            self.assertIsNone(arms["ppo_checkpoint_19"]["booking_success"]["denominator"])
            self.assertEqual(report["cohort"]["manifest_sha256"], "a" * 64)
            self.assertEqual(len(report["evidence_artifacts"]), len(EVIDENCE_SOURCES))
            for item in report["evidence_artifacts"]:
                source = root / item["artifact"]
                self.assertEqual(item["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertFalse(Path(item["artifact"]).is_absolute())

    def test_report_is_summary_only_incomplete_and_contains_no_scenario_records(self):
        with tempfile.TemporaryDirectory() as directory:
            report = build_policy_evidence(root=self.make_root(Path(directory)))
        self.assertEqual(report["evaluation_layer"]["status"], "incomplete")
        self.assertEqual(report["evaluation_layer"]["evidence_level"], "reported_summary")
        self.assertTrue(report["results_imported"])
        self.assertFalse(report["scenario_records_imported"])
        self.assertEqual(report["scenarios"], [])
        self.assertTrue(all(not arm["scenario_records_available"]
                            for arm in report["summary"]["adapters"].values()))
        self.assertTrue(all(arm["evaluated_adapter_sha256"] is None
                            for arm in report["summary"]["adapters"].values()))

    def test_importer_does_not_copy_arbitrary_text_or_absolute_paths(self):
        text = RESULTS_TABLE + "\nPrivate note: /Users/alice/results.json passenger Alice api_key=abc\n"
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory), text)
            output = root / "artifacts/evaluation/policy-only-v1.json"
            write_policy_evidence(root=root, output=output)
            serialized = output.read_text(encoding="utf-8")
        for forbidden in ("/Users/alice", "Alice", "api_key", "abc", "gs://", "passenger"):
            self.assertNotIn(forbidden, serialized)

    def test_missing_arm_row_fails_instead_of_emitting_partial_report(self):
        incomplete = RESULTS_TABLE.replace("| DPO | 17/50 (34%) | 22.4%–47.8% | 2.58 | 0% | 0% |\n", "")
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory), incomplete)
            with self.assertRaisesRegex(ValueError, "missing adapter rows: dpo"):
                build_policy_evidence(root=root)

    def test_release_renderer_shows_per_arm_metrics_and_summary_only_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(Path(directory))
            report = build_policy_evidence(root=root)
            policy_path = root / "policy-only-v1.json"
            policy_path.write_text(json.dumps(report), encoding="utf-8")
            rendered = render_release_report(policy_path=policy_path,
                                             controller_path=root / "missing-controller.json",
                                             integrated_path=root / "missing-integrated.json",
                                             browser_path=root / "missing-browser.json")
        self.assertIn("Policy evidence level: `reported_summary`", rendered)
        self.assertIn("`sft` — summary-level", rendered)
        self.assertIn('"successes": 13', rendered)
        self.assertIn("not scenario-level records", rendered)
        self.assertIn("resolved evaluation revisions", rendered)
        self.assertIn("PENDING — no complete integrated release artifact", rendered)


if __name__ == "__main__":
    unittest.main()

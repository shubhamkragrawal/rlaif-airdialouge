import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_app.release_report import render_release_report


class ReleaseReportTests(unittest.TestCase):
    def test_default_report_is_pending_and_does_not_promote_controller_evidence(self):
        report = render_release_report()
        self.assertIn("PENDING — no complete integrated release artifact", report)
        self.assertIn("Integrated metrics: unavailable", report)
        self.assertIn("Policy metrics: unavailable", report)
        self.assertIn("Wilson intervals", report)
        self.assertIn("simulated", report)
        self.assertNotIn("/Users/", report)

    def test_versioned_artifacts_are_cited_with_hashes_and_missing_fields_stay_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = {"report_version": 1, "evaluation_layer": {"layer": "policy_only", "status": "incomplete"},
                      "results_imported": False, "summary": None,
                      "evidence_artifacts": [{"artifact": "policy-summary-v1", "sha256": "a" * 64}]}
            controller = {"schema": "rlaif.local_app.adapter_matrix", "report_version": 1,
                          "layer": "controller_only", "cohort_id": "cohort-x",
                          "cohort": {"scenario_count": 3}, "adapters": {}}
            browser = {"artifact_version": 1, "cases": [{"case_id": "BR001", "coverage": "invalid choice rejected"}]}
            paths = {}
            for name, data in (("policy.json", policy), ("controller.json", controller),
                               ("browser.json", browser)):
                path = root / name
                path.write_text(json.dumps(data), encoding="utf-8")
                paths[name] = path
            report = render_release_report(policy_path=paths["policy.json"],
                                           controller_path=paths["controller.json"],
                                           integrated_path=root / "missing.json", browser_path=paths["browser.json"])
            expected_hash = hashlib.sha256(paths["policy.json"].read_bytes()).hexdigest()
            self.assertIn(expected_hash, report)
            self.assertIn("`task_completion`: unavailable", report)
            self.assertIn("`BR001`", report)
            self.assertNotIn(directory, report)
            self.assertNotIn("/Users/", report)

    def test_integrated_values_require_versioned_integrated_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "integrated.json"
            path.write_text(json.dumps({"schema": "rlaif.local_app.matrix_release", "release_version": 1,
                                        "layer": "integrated", "identity": {}, "generation_settings": {},
                                        "adapters": {}}), encoding="utf-8")
            report = render_release_report(integrated_path=path)
            self.assertIn("PENDING — no complete integrated release artifact", report)
            self.assertIn("Integrated metrics: unavailable", report)

    def test_declared_complete_arms_without_integrity_evidence_remain_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integrated.json"
            arms = {name: {"layer": "integrated", "status": "complete",
                           "runtime_provenance": {"complete": True}}
                    for name in ("sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50")}
            path.write_text(json.dumps({"schema": "rlaif.local_app.matrix_release",
                                        "release_version": 1, "mode": "integrated",
                                        "layer": "integrated", "adapters": arms}), encoding="utf-8")
            report = render_release_report(integrated_path=path)
            self.assertIn("PENDING — no complete integrated release artifact", report)

    def test_imported_policy_summary_is_allowlisted_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = {"report_version": 1,
                      "evaluation_layer": {"layer": "policy_only", "status": "complete"},
                      "results_imported": True,
                      "summary": {"task_completion": {"rate": 1, "passenger_name": "Alice"},
                                  "secret": "top-secret", "notes": "/opt/private/result.json"},
                      "evidence_artifacts": []}
            path = root / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            report = render_release_report(policy_path=path,
                                           controller_path=root / "missing-controller.json",
                                           integrated_path=root / "missing-integrated.json",
                                           browser_path=root / "missing-browser.json")
            self.assertIn("Policy metrics (allowlisted fields only)", report)
            self.assertNotIn("Alice", report)
            self.assertNotIn("top-secret", report)
            self.assertNotIn("/opt/private", report)
            self.assertNotIn("notes", report)


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_app.controller import ADAPTERS
from local_app.release_report import render_release_report
from local_app.tests.test_release_gate import ReleaseGateTests


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def rehash_release(release):
    payload = {key: value for key, value in release.items() if key != "sha256"}
    release["sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def write_valid_release(root):
    ids = list(range(50))
    settings = {"mode": "model", "base_model": "local/base", "revision": "revision",
                "temperature": 0.0, "max_new_tokens": 160, "start": 0, "count": 50}
    arms = {}
    for name in ADAPTERS:
        adapter_hash = hashlib.sha256(name.encode()).hexdigest()
        provenance = {"complete": True, "adapter_sha256": adapter_hash,
                      "base_model": settings["base_model"],
                      "base_model_revision": settings["revision"],
                      "generation_settings": {"temperature": 0.0, "max_new_tokens": 160},
                      "device": "cpu", "precision": "float32", "peak_memory_bytes": 1024}
        provenance["sha256"] = hashlib.sha256(json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()).hexdigest()
        run_record = {"hardware": {"device": "cpu", "precision": "float32",
                                  "peak_memory_bytes": 1024}}
        run_manifest = {"record": run_record, "sha256": hashlib.sha256(json.dumps(
            run_record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()).hexdigest()}
        turns = []
        for scenario_id in ids:
            evidence = {"scenario_id": scenario_id,
                        "unsupported_transaction_claim": False,
                        "malformed_output": False, "repetitive_output": False,
                        "validation_result": "valid", "transaction_claim": False,
                        "valid_search_transition": None, "valid_booking_transition": None,
                        "expected_no_results": None, "presented_no_results": None,
                        "model_attempted": scenario_id == 0,
                        "model_generated": scenario_id == 0,
                        "model_accepted": False,
                        "model_fallback": scenario_id == 0,
                        "input_tokens": 1, "output_tokens": 2}
            record = {"scenario_id": scenario_id}
            record_hash = hashlib.sha256(json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()).hexdigest()
            turn_hash = hashlib.sha256(json.dumps(
                {"record": record, "evidence": evidence},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()).hexdigest()
            turns.append({"record": record, "evidence": evidence,
                          "record_sha256": record_hash, "sha256": turn_hash})
        metrics = {
            "scenario_count": 50,
            "task_completion": {"rate": 1.0, "successes": 50}, "valid_transitions": {"rate": 1.0},
            "grounded_entity_precision": 1.0, "no_results_presentation_accuracy": 1.0,
            "unsupported_transaction_claim_rate": 0.0, "malformed_output_rate": 0.0,
            "repetition_rate": 0.0, "repetition_evidence": {"rate": 0.0},
            "validation": {"rate": 1.0}, "fallback_rate": 0.0,
            "fallback_reason_distribution": {}, "turns": {},
            "tokens": {"input": 50, "output": 100},
            "latency_ms": {}, "peak_memory_mb": 1.0,
            "judge_score": {"value": None, "basis": "judge-dependent"},
            "model_inference": {"attempted_turns": 1, "generated_turns": 1,
                                "accepted_turns": 0, "fallback_turns": 1,
                                "fallback_reasons": ["test"]},
        }
        arms[name] = {
            "layer": "integrated", "status": "complete", "cohort_id": "cohort",
            "scenario_ids": ids, "runtime_provenance": provenance,
            "runtime_status": {"lifecycle_status": "ready", "mode": "local_model",
                               "available": True},
            "model_participation": {"required": True, "turns": 1,
                                    "attempted_turns": 1, "generated_turns": 1,
                                    "accepted_turns": 0, "fallback_turns": 1},
            "run_manifest": run_manifest, "turn_manifests": turns,
            "scenarios": [{"scenario_id": item,
                           "metrics": {"turn_count": 1, "task_success": True},
                           "normalized_outcome": {"task_success": True}}
                          for item in ids],
            "metrics": metrics,
        }
    release = {
        "schema": "rlaif.local_app.matrix_release", "release_version": 1,
        "mode": "integrated", "layer": "integrated", "cohort_id": "cohort",
        "cohort": {"scenario_ids": ids, "scenario_count": 50, "settings": settings},
        "generation_settings": settings,
        "identity": {
            "adapter_sha256": {name: arms[name]["runtime_provenance"]["adapter_sha256"]
                               for name in ADAPTERS},
            "base_model": settings["base_model"],
            "base_model_revision": settings["revision"], "corpus_sha256": "b" * 64,
            "controller_sha256": "c" * 64, "generation_settings": settings,
        },
        "adapters": arms,
    }
    rehash_release(release)
    path = root / "release-v1.json"
    path.write_text(json.dumps(release), encoding="utf-8")
    return path


class ReleaseReportTests(unittest.TestCase):
    def _integrated_release(self, root):
        bundle = ReleaseGateTests().fixture(root)
        bundle_data = json.loads(bundle.read_text())
        matrix_path = root / bundle_data["integrated_matrix"]["path"]
        matrix = json.loads(matrix_path.read_text())
        settings = {"mode": "model", "base_model": "local/base", "revision": "revision",
                    "temperature": 0.0, "max_new_tokens": 160, "start": 0, "count": 50}
        release = {
            "schema": "rlaif.local_app.matrix_release", "release_version": 1,
            "mode": "integrated", "layer": "integrated", "cohort_id": matrix["cohort_id"],
            "cohort": {**matrix["cohort"], "settings": settings},
            "generation_settings": settings, "adapters": matrix["adapters"],
            "identity": {"adapter_sha256": {name: "a" * 64 for name in matrix["adapters"]},
                         "base_model": "local/base", "base_model_revision": "revision",
                         "corpus_sha256": "b" * 64, "controller_sha256": "a" * 64,
                         "generation_settings": settings},
        }
        release["sha256"] = canonical_hash(release)
        matrix_path.write_text(json.dumps(release), encoding="utf-8")
        return matrix_path, release

    @staticmethod
    def _rehash_run_and_release(path, release, arm):
        record = arm["run_manifest"]["record"]
        arm["run_manifest"]["sha256"] = canonical_hash(record)
        release["sha256"] = canonical_hash({key: value for key, value in release.items() if key != "sha256"})
        path.write_text(json.dumps(release), encoding="utf-8")

    def test_default_report_is_pending_and_does_not_promote_controller_evidence(self):
        # Keep this negative-path test independent of whether a developer has
        # already generated the real default release artifact locally.
        with tempfile.TemporaryDirectory() as directory:
            report = render_release_report(
                integrated_path=Path(directory) / "missing-integrated.json"
            )
        self.assertIn("PENDING — no complete integrated release artifact", report)
        self.assertIn("Integrated metrics: unavailable", report)
        self.assertIn("Policy evidence level: `reported_summary`", report)
        self.assertIn("`sft` — summary-level", report)
        self.assertIn("not scenario-level records", report)
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

    def test_rehashed_integrated_release_is_pending_when_hardware_is_deleted_or_mps_is_changed_to_cuda(self):
        for mutation in ("delete_hardware", "mps_to_cuda"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, release = self._integrated_release(root)
                self.assertIn("Integrated artifact present", render_release_report(integrated_path=path))
                arm = release["adapters"]["sft"]
                record = arm["run_manifest"]["record"]
                provenance = arm["runtime_provenance"]
                if mutation == "delete_hardware":
                    del record["hardware"]
                else:
                    provenance["device"] = "mps"
                    provenance.pop("sha256")
                    provenance["sha256"] = canonical_hash(provenance)
                    record["hardware"]["device"] = "mps"
                    arm["run_manifest"]["sha256"] = canonical_hash(record)
                    record["hardware"]["device"] = "cuda"
                self._rehash_run_and_release(path, release, arm)
                report = render_release_report(integrated_path=path)
                self.assertIn("PENDING — no complete integrated release artifact", report)
                self.assertIn("Integrated metrics: unavailable", report)

    def test_release_with_a_removed_turn_manifest_remains_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_valid_release(Path(directory))
            release = json.loads(path.read_text(encoding="utf-8"))
            release["adapters"][ADAPTERS[0]]["turn_manifests"].pop()
            rehash_release(release)
            path.write_text(json.dumps(release), encoding="utf-8")
            report = render_release_report(integrated_path=path)
        self.assertIn("PENDING — no complete integrated release artifact", report)
        self.assertIn("Integrated metrics: unavailable", report)

    def test_zero_model_participation_on_every_arm_remains_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_valid_release(Path(directory))
            release = json.loads(path.read_text(encoding="utf-8"))
            for arm in release["adapters"].values():
                arm["model_participation"].update({
                    "turns": 0, "attempted_turns": 0, "generated_turns": 0,
                    "accepted_turns": 0, "fallback_turns": 0,
                })
                arm["metrics"]["model_inference"].update({
                    "attempted_turns": 0, "generated_turns": 0,
                    "accepted_turns": 0, "fallback_turns": 0,
                    "acceptance_rate": None, "fallback_reasons": [],
                })
                for turn in arm["turn_manifests"]:
                    evidence = turn["evidence"]
                    for key in ("model_attempted", "model_generated", "model_accepted", "model_fallback"):
                        evidence[key] = False
                    turn["record_sha256"] = hashlib.sha256(json.dumps(
                        turn["record"], sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()).hexdigest()
                    turn["sha256"] = hashlib.sha256(json.dumps(
                        {"record": turn["record"], "evidence": evidence},
                        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                    ).encode()).hexdigest()
            rehash_release(release)
            path.write_text(json.dumps(release), encoding="utf-8")
            report = render_release_report(integrated_path=path)
        self.assertIn("PENDING — no complete integrated release artifact", report)
        self.assertIn("Integrated metrics: unavailable", report)

    def test_rehashed_scenario_failure_under_unchanged_50_of_50_aggregate_remains_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_valid_release(Path(directory))
            release = json.loads(path.read_text(encoding="utf-8"))
            scenario = release["adapters"][ADAPTERS[0]]["scenarios"][0]
            scenario["metrics"]["task_success"] = False
            scenario["normalized_outcome"]["task_success"] = False
            rehash_release(release)
            path.write_text(json.dumps(release), encoding="utf-8")
            report = render_release_report(integrated_path=path)
        self.assertIn("PENDING — no complete integrated release artifact", report)
        self.assertIn("Integrated metrics: unavailable", report)

    def test_rehashed_fabricated_zero_token_totals_remain_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_valid_release(Path(directory))
            release = json.loads(path.read_text(encoding="utf-8"))
            release["adapters"][ADAPTERS[0]]["metrics"]["tokens"] = {
                "input": 0, "output": 0,
            }
            rehash_release(release)
            path.write_text(json.dumps(release), encoding="utf-8")
            report = render_release_report(integrated_path=path)
        self.assertIn("PENDING — no complete integrated release artifact", report)
        self.assertIn("Integrated metrics: unavailable", report)

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

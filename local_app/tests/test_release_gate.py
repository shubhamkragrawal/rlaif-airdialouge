import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_app.controller import ADAPTERS
from local_app.release_gate import PROTECTED_STAGING, evaluate, write_versioned


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.relative_to(path.parents[1])), "sha256": digest(path)}


class ReleaseGateTests(unittest.TestCase):
    def fixture(self, root):
        evidence = {"schema": "rlaif.local_app.release_evidence", "version": 1, "checks": {}}
        for check in ("local", "rebuild", "browser", "catalog", "clean_package", "ops"):
            p = root / "evidence" / f"{check}.json"
            put_json(p, {"command": f"run {check}", "status": "passed", "exit_code": 0})
            evidence["checks"][check] = {"path": f"evidence/{check}.json", "sha256": digest(p)}
        ids = list(range(50))
        adapters = {}
        for name in ADAPTERS:
            d = "a" * 64
            run_record = {"adapter": name}
            run_hash = hashlib.sha256(json.dumps(run_record, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            turn_record = {"scenario": 0}
            turn_evidence = {"unsupported_transaction_claim": False,
                             "malformed_output": False, "repetitive_output": False,
                             "validation_result": "valid", "transaction_claim": False,
                             "valid_search_transition": None,
                             "valid_booking_transition": None,
                             "expected_no_results": None, "presented_no_results": None}
            turn_hash = hashlib.sha256(json.dumps({"record": turn_record, "evidence": turn_evidence},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            provenance = {"complete": True, "adapter_sha256": d,
                          "base_model": "local/base", "base_model_revision": "revision",
                          "generation_settings": {"temperature": 0, "max_new_tokens": 160}, "device": "cpu",
                          "precision": "float32", "peak_memory_bytes": 1024}
            provenance["sha256"] = hashlib.sha256(json.dumps(provenance, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            metrics = {"scenario_count": 50, "task_completion": {"rate": 1, "successes": 50},
                       "unsupported_transaction_claim_rate": 0,
                       "malformed_output_rate": 0, "fallback_rate": 0,
                       "fallback_reason_distribution": {},
                       "valid_transitions": {"rate": 1}, "grounded_entity_precision": 1,
                       "no_results_presentation_accuracy": 1, "repetition_rate": 0,
                       "repetition_evidence": {"rate": 0}, "validation": {"rate": 1},
                       "turns": {}, "tokens": {}, "latency_ms": {},
                       "peak_memory_mb": 1, "judge_score": {"value": None, "basis": "judge-dependent"}}
            adapters[name] = {"layer": "integrated", "status": "complete", "cohort_id": "cohort",
                "scenario_ids": ids, "fingerprints": {"controller_sha256": d,
                "outcome_schema": {"name": "rlaif.local_app.turn_outcome", "version": 1,
                                   "sha256": d}, "runtime_provenance_sha256": provenance["sha256"]},
                "runtime_provenance": provenance,
                "runtime_status": {"lifecycle_status": "ready", "mode": "local_model", "available": True},
                "model_participation": {"required": True, "turns": 1},
                "run_manifest": {"record": run_record, "sha256": run_hash},
                "turn_manifests": [{"record": turn_record, "evidence": turn_evidence,
                                    "record_sha256": hashlib.sha256(json.dumps(turn_record,
                                        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                                    "sha256": turn_hash}],
                "scenarios": [{"scenario_id": scenario_id} for scenario_id in ids], "metrics": metrics}
        matrix = {"schema": "rlaif.local_app.adapter_matrix", "report_version": 1,
                  "mode": "integrated", "layer": "integrated", "cohort_id": "cohort",
                  "cohort": {"scenario_ids": ids, "scenario_count": 50,
                             "settings": {"mode": "model", "base_model": "local/base",
                                          "revision": "revision", "temperature": 0,
                                          "max_new_tokens": 160}}, "adapters": adapters}
        matrix_path = root / "evidence" / "integrated.json"
        put_json(matrix_path, matrix)
        evidence["integrated_matrix"] = {"path": "evidence/integrated.json", "sha256": digest(matrix_path)}
        bundle = root / "evidence-v1.json"
        put_json(bundle, evidence)
        return bundle

    def test_accepts_complete_hash_pinned_synthetic_evidence_and_writes_versioned_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            result = evaluate(bundle, root=root)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["adapters"], list(ADAPTERS))
            output = write_versioned(result, root / "artifacts/release")
            self.assertEqual(output.name, "release-gate-v1.json")
            self.assertEqual(json.loads(output.read_text())["status"], "pass")

    def test_rejects_tampered_evidence_and_incomplete_integrated_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            matrix_path = root / "evidence/integrated.json"
            matrix = json.loads(matrix_path.read_text())
            matrix["adapters"][ADAPTERS[0]]["model_participation"]["turns"] = 0
            put_json(matrix_path, matrix)
            result = evaluate(bundle, root=root)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("SHA-256 mismatch" in failure for failure in result["failures"]))

    def test_rejects_rehashed_invalid_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            matrix_path = root / "evidence/integrated.json"
            matrix = json.loads(matrix_path.read_text())
            matrix["adapters"][ADAPTERS[1]]["metrics"]["unsupported_transaction_claim_rate"] = 0.1
            put_json(matrix_path, matrix)
            bundle_data = json.loads(bundle.read_text())
            bundle_data["integrated_matrix"]["sha256"] = digest(matrix_path)
            put_json(bundle, bundle_data)
            result = evaluate(bundle, root=root)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("unsupported_transaction_claim_rate" in failure for failure in result["failures"]))

    def test_rejects_negative_and_incomplete_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            matrix_path = root / "evidence/integrated.json"
            matrix = json.loads(matrix_path.read_text())
            metrics = matrix["adapters"][ADAPTERS[0]]["metrics"]
            metrics["malformed_output_rate"] = -0.1
            metrics.pop("task_completion")
            put_json(matrix_path, matrix)
            bundle_data = json.loads(bundle.read_text())
            bundle_data["integrated_matrix"]["sha256"] = digest(matrix_path)
            put_json(bundle, bundle_data)
            result = evaluate(bundle, root=root)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("task_completion" in failure or "[0, 1]" in failure
                                for failure in result["failures"]))

    def test_rejects_complete_envelope_with_failed_quality_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            matrix_path = root / "evidence/integrated.json"
            matrix = json.loads(matrix_path.read_text())
            metrics = matrix["adapters"][ADAPTERS[0]]["metrics"]
            metrics["task_completion"] = {"rate": 0, "successes": 0}
            metrics["valid_transitions"] = {"rate": 0}
            metrics["repetition_rate"] = 1
            put_json(matrix_path, matrix)
            bundle_data = json.loads(bundle.read_text())
            bundle_data["integrated_matrix"]["sha256"] = digest(matrix_path)
            put_json(bundle, bundle_data)
            result = evaluate(bundle, root=root)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("release target" in failure for failure in result["failures"]))

    def test_rejects_unsafe_turn_even_when_aggregate_metrics_claim_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            matrix_path = root / "evidence/integrated.json"
            matrix = json.loads(matrix_path.read_text())
            turn = matrix["adapters"][ADAPTERS[0]]["turn_manifests"][0]
            turn["evidence"]["unsupported_transaction_claim"] = True
            turn["sha256"] = hashlib.sha256(json.dumps(
                {"record": turn["record"], "evidence": turn["evidence"]}, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            put_json(matrix_path, matrix)
            bundle_data = json.loads(bundle.read_text())
            bundle_data["integrated_matrix"]["sha256"] = digest(matrix_path)
            put_json(bundle, bundle_data)
            result = evaluate(bundle, root=root)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("per-turn" in failure for failure in result["failures"]))

    def test_rejects_failed_or_non_simulated_transaction_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            matrix_path = root / "evidence/integrated.json"
            matrix = json.loads(matrix_path.read_text())
            turn = matrix["adapters"][ADAPTERS[0]]["turn_manifests"][0]
            turn["evidence"].update({
                "transaction_claim": True,
                "authoritative_transaction_evidence": {
                    "current_turn": True, "booking_status": "failed", "simulated": False,
                    "hold_id": "not-a-hold", "booking_id": "not-a-booking",
                },
            })
            turn["sha256"] = hashlib.sha256(json.dumps(
                {"record": turn["record"], "evidence": turn["evidence"]}, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            put_json(matrix_path, matrix)
            bundle_data = json.loads(bundle.read_text())
            bundle_data["integrated_matrix"]["sha256"] = digest(matrix_path)
            put_json(bundle, bundle_data)
            result = evaluate(bundle, root=root)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("simulator evidence" in failure for failure in result["failures"]))

    def test_accepts_release_runner_schema_and_allows_packaged_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            matrix_path = root / "evidence/integrated.json"
            matrix = json.loads(matrix_path.read_text())
            release_ids = list(range(50))
            matrix["cohort"]["scenario_ids"] = release_ids
            matrix["cohort"]["scenario_count"] = 50
            for arm in matrix["adapters"].values():
                arm["scenario_ids"] = release_ids
                arm["scenarios"] = [{"scenario_id": scenario_id} for scenario_id in release_ids]
                arm["metrics"]["scenario_count"] = 50
            settings = {"mode": "model", "base_model": "local/base",
                        "revision": "revision", "temperature": 0.0,
                        "max_new_tokens": 160, "start": 0, "count": 50}
            release = {
                "schema": "rlaif.local_app.matrix_release", "release_version": 1,
                "mode": "integrated", "layer": "integrated",
                "cohort_id": matrix["cohort_id"],
                "cohort": {**matrix["cohort"], "settings": settings},
                "generation_settings": settings, "adapters": matrix["adapters"],
                "identity": {"adapter_sha256": {name: "a" * 64 for name in ADAPTERS},
                             "base_model": "local/base", "base_model_revision": "revision",
                             "corpus_sha256": "b" * 64, "controller_sha256": "a" * 64,
                             "generation_settings": settings},
            }
            release["sha256"] = hashlib.sha256(json.dumps(release, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            put_json(matrix_path, release)
            bundle_data = json.loads(bundle.read_text())
            bundle_data["integrated_matrix"]["sha256"] = digest(matrix_path)
            put_json(bundle, bundle_data)
            self.assertEqual(evaluate(bundle, root=root)["status"], "pass")
        self.assertIsNone(PROTECTED_STAGING.search("local_app/data/held_out_50.jsonl"))
        self.assertIsNotNone(PROTECTED_STAGING.search(".env.local"))
        self.assertIsNotNone(PROTECTED_STAGING.search("exports/model.safetensors"))
        self.assertIsNotNone(PROTECTED_STAGING.search("rebuild/cloud/pilot.json"))


if __name__ == "__main__":
    unittest.main()

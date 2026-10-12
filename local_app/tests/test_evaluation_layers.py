import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from local_app.evaluation import (
    EvaluationScenario,
    REPORT_VERSION,
    build_policy_only_report,
    evaluation_layer_contract,
    policy_only_contract,
    build_report,
    main,
    run_scenario,
)


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "airdialogue_sampled_5k.jsonl"


class StubModel:
    last_usage = {"input_tokens": 10, "output_tokens": 5}

    def generate(self, _messages, _retrieved):
        return "Flight 1027 is available for $100."


class EvaluationLayerContractTests(unittest.TestCase):
    def integrated_row(self):
        scenario = EvaluationScenario(0, ("Find a flight from IAD to LAS under $100",),
                                     "book", (1027,))
        row = run_scenario(scenario, corpus_path=CORPUS)
        row["model_turns"] = 1
        row["model_attempted_turns"] = 1
        row["model_generated_turns"] = 1
        row["model_accepted_turns"] = 0
        row["model_fallbacks"] = 2
        return scenario, row

    def test_controller_only_contract_forbids_model_loading(self):
        with self.assertRaisesRegex(ValueError, "forbid model loading"):
            evaluation_layer_contract(
                "controller_only", cohort_id="cohort", protocol_id="protocol",
                controller_usage={"executed": True},
                model_usage={"loading_permitted": True}, provenance=[{"path": "fixture"}],
            )

    def test_policy_only_references_existing_protocol_without_controller_rescue(self):
        contract = policy_only_contract(root=ROOT)
        self.assertEqual(contract["layer"], "policy_only")
        self.assertEqual(contract["protocol_id"], "held-out-policy-protocol-v3")
        self.assertEqual(contract["controller_usage"], {"executed": False, "rescue": False})
        self.assertFalse(contract["model_usage"]["runtime_invoked"])
        self.assertEqual(contract["status"], "incomplete")
        self.assertEqual(len(contract["provenance"]), 2)
        self.assertTrue(all(item.get("sha256") for item in contract["provenance"]))

    def test_policy_only_reference_without_imported_metrics_is_incomplete(self):
        report = build_policy_only_report(root=ROOT)
        self.assertEqual(report["report_version"], 2)
        self.assertEqual(REPORT_VERSION, 2)
        self.assertEqual(report["evaluation_layer"]["status"], "incomplete")
        self.assertFalse(report["results_imported"])
        self.assertIsNone(report["summary"])
        self.assertEqual(report["scenarios"], [])
        self.assertIn("not imported", report["evaluation_layer"]["status_reason"])
        self.assertFalse(report["evaluation_layer"]["fingerprint"]["generation_settings"]["cloud_jobs_launched"])
        fingerprint = report["evaluation_layer"]["fingerprint"]
        self.assertTrue(fingerprint["corpus"]["sha256"])
        self.assertEqual(fingerprint["controller"]["schema_version"], 1)
        self.assertEqual(fingerprint["adapter"]["name"], "all-held-out-arms")

    def test_missing_policy_evidence_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = policy_only_contract(root=directory)
        self.assertEqual(contract["status"], "unavailable")

    def test_current_report_is_explicit_controller_only_and_keeps_legacy_fields(self):
        scenario = EvaluationScenario(0, ("Find a flight from IAD to LAS under $100",),
                                     "book", (1027,))
        report = build_report([scenario], corpus_path=CORPUS)
        self.assertEqual(report["evaluation_layer"]["layer"], "controller_only")
        self.assertFalse(report["evaluation_layer"]["model_usage"]["loading_permitted"])
        self.assertIn("summary", report)
        self.assertIn("scenarios", report)
        self.assertIn("adapter", report)
        self.assertEqual(report["report_version"], 2)
        fingerprint = report["evaluation_layer"]["fingerprint"]
        self.assertEqual(fingerprint["case_ids"], [0])
        self.assertTrue(fingerprint["controller"]["version"])
        self.assertTrue(fingerprint["protocol"]["sha256"])
        self.assertTrue(fingerprint["corpus"]["sha256"])

    def test_integrated_layer_requires_controller_and_model_use_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "requires controller execution"):
            evaluation_layer_contract(
                "integrated", cohort_id="cohort", protocol_id="protocol",
                controller_usage={"executed": False},
                model_usage={"loading_permitted": True}, provenance=[{"path": "fixture"}],
            )
        report = build_report(
            [EvaluationScenario(0, ("Find a flight from IAD to LAS under $100",),
                                "book", (1027,))],
            corpus_path=CORPUS,
            model_loader=lambda _name: None,
        )
        self.assertEqual(report["evaluation_layer"]["layer"], "integrated")
        self.assertEqual(report["evaluation_layer"]["status"], "incomplete")
        self.assertEqual(report["evaluation_layer"]["model_usage"]["model_turns"], 0)
        self.assertEqual(report["evaluation_layer"]["model_usage"]["fallback_turns"], 1)
        self.assertIn("model_usage.model_generated_turns", report["evaluation_layer"]["status_reason"])

    def test_integrated_contract_arbitrary_paths_cannot_bypass_completeness(self):
        contract = evaluation_layer_contract(
            "integrated", cohort_id="cohort", protocol_id="protocol",
            controller_usage={"executed": True},
            model_usage={"loading_permitted": False, "loading_requested": False,
                         "model_turns": 0, "model_attempted_turns": 0,
                         "model_generated_turns": 0, "model_accepted_turns": 0},
            provenance=[{"path": "/arbitrary/evidence", "sha256": "f" * 64}],
            status="complete",
        )
        self.assertEqual(contract["status"], "incomplete")
        for field in ("model_usage.loading_permitted", "model_usage.loading_requested",
                      "model_usage.model_attempted_turns", "model_usage.model_generated_turns",
                      "adapter.artifact", "adapter.sha256",
                      "generation_settings.base_model"):
            self.assertIn(field, contract["status_reason"])

    def test_integrated_contract_complete_with_full_model_provenance(self):
        artifact = {"artifact": "gs://bucket/adapter", "sha256": "a" * 64}
        settings = {"mode": "model", "base_model": "Qwen/test", "revision": "rev-1",
                    "max_new_tokens": 160, "temperature": 0.2}
        contract = evaluation_layer_contract(
            "integrated", cohort_id="cohort", protocol_id="protocol",
            controller_usage={"executed": True},
            model_usage={"loading_permitted": True, "loading_requested": True,
                         "model_turns": 1, "model_attempted_turns": 3,
                         "model_generated_turns": 1, "model_accepted_turns": 0,
                         "fallback_turns": 2},
            provenance=[{"path": "evaluation.py", "sha256": "b" * 64}],
            adapter_artifact=artifact, generation_settings=settings,
            status="complete",
        )
        self.assertEqual(contract["status"], "complete")
        self.assertIsNone(contract["status_reason"])
        self.assertEqual(contract["adapter_provenance"], artifact)
        self.assertEqual(contract["generation_settings"], settings)

    def test_rejected_generated_candidate_counts_as_integrated_participation(self):
        artifact = {"artifact": "stub-adapter", "sha256": "a" * 64}
        settings = {"mode": "model", "base_model": "Qwen/test", "revision": "rev-1",
                    "max_new_tokens": 160, "temperature": 0.0}
        contract = evaluation_layer_contract(
            "integrated", cohort_id="cohort", protocol_id="protocol",
            controller_usage={"executed": True},
            model_usage={"loading_permitted": True, "loading_requested": True,
                         "model_turns": 1, "model_attempted_turns": 1,
                         "model_generated_turns": 1, "model_accepted_turns": 0,
                         "fallback_turns": 1},
            provenance=[{"path": "evaluation.py", "sha256": "b" * 64}],
            adapter_artifact=artifact, generation_settings=settings,
            status="complete",
        )
        self.assertEqual(contract["status"], "complete")
        self.assertEqual(contract["model_usage"]["model_accepted_turns"], 0)

    def test_integrated_model_participation_without_adapter_provenance_is_incomplete(self):
        scenario = EvaluationScenario(0, ("Find a flight from IAD to LAS under $100",),
                                     "book", (1027,))
        report = build_report([scenario], corpus_path=CORPUS,
                              model_loader=lambda _name: StubModel())
        contract = report["evaluation_layer"]
        self.assertEqual(contract["status"], "incomplete")
        self.assertEqual(contract["model_usage"]["model_turns"], 1)
        self.assertEqual(contract["fingerprint"]["generation_settings"]["mode"], "model")
        self.assertIn("adapter.artifact", contract["status_reason"])
        self.assertIn("adapter.sha256", contract["status_reason"])

    def test_integrated_missing_adapter_artifact_is_incomplete(self):
        scenario, row = self.integrated_row()
        settings = {"mode": "model", "base_model": "base", "revision": "rev",
                    "max_new_tokens": 160, "temperature": 0.2}
        with patch("local_app.evaluation.run_scenario", return_value=row):
            report = build_report(
                [scenario], corpus_path=CORPUS, model_loader=lambda _name: None,
                adapter_artifact={"sha256": "a" * 64}, generation_settings=settings,
            )
        self.assertEqual(report["evaluation_layer"]["status"], "incomplete")
        self.assertIn("adapter.artifact", report["evaluation_layer"]["status_reason"])

    def test_integrated_missing_adapter_hash_is_incomplete(self):
        scenario, row = self.integrated_row()
        settings = {"mode": "model", "base_model": "base", "revision": "rev",
                    "max_new_tokens": 160, "temperature": 0.2}
        with patch("local_app.evaluation.run_scenario", return_value=row):
            report = build_report(
                [scenario], corpus_path=CORPUS, model_loader=lambda _name: None,
                adapter_artifact={"artifact": "stub-adapter"},
                generation_settings=settings,
            )
        self.assertEqual(report["evaluation_layer"]["status"], "incomplete")
        self.assertIn("adapter.sha256", report["evaluation_layer"]["status_reason"])

    def test_integrated_missing_generation_settings_is_incomplete(self):
        scenario, row = self.integrated_row()
        with patch("local_app.evaluation.run_scenario", return_value=row):
            report = build_report(
                [scenario], corpus_path=CORPUS, model_loader=lambda _name: None,
                adapter_artifact={"artifact": "stub-adapter", "sha256": "a" * 64},
                generation_settings={"mode": "model"},
            )
        contract = report["evaluation_layer"]
        self.assertEqual(contract["status"], "incomplete")
        self.assertIn("generation_settings.base_model", contract["status_reason"])
        self.assertNotEqual(contract["fingerprint"]["generation_settings"]["mode"],
                            "deterministic_controller")

    def test_integrated_complete_with_adapter_and_generation_provenance(self):
        scenario, row = self.integrated_row()
        artifact = {"artifact": "stub-adapter", "sha256": "a" * 64}
        settings = {"mode": "model", "base_model": "Qwen/test", "revision": "rev-1",
                    "max_new_tokens": 160, "temperature": 0.2}
        with patch("local_app.evaluation.run_scenario", return_value=row):
            report = build_report(
                [scenario], corpus_path=CORPUS, model_loader=lambda _name: None,
                adapter_artifact=artifact, generation_settings=settings,
            )
        contract = report["evaluation_layer"]
        self.assertEqual(contract["status"], "complete")
        self.assertIsNone(contract["status_reason"])
        self.assertEqual(contract["fingerprint"]["adapter"]["sha256"], "a" * 64)
        self.assertEqual(contract["fingerprint"]["generation_settings"], settings)
        self.assertEqual(contract["model_usage"]["fallback_turns"], 2)

    def test_cli_policy_only_writes_evidence_report_without_loading_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "policy.json"
            self.assertEqual(main(["--layer", "policy_only", "--output", str(output)]), 0)
            import json
            report = json.loads(output.read_text())
        self.assertEqual(report["evaluation_layer"]["layer"], "policy_only")
        self.assertEqual(report["evaluation_layer"]["status"], "incomplete")

    def test_explicit_controller_only_rejects_model_flag(self):
        with self.assertRaises(SystemExit):
            main(["--layer", "controller_only", "--model", "--output", "unused.json"])

    def test_explicit_integrated_cli_records_unavailable_model_fallback_as_incomplete(self):
        from local_app.runtime import RuntimeConfig
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "integrated.json"
            with patch.object(RuntimeConfig, "local_defaults",
                              return_value=RuntimeConfig(adapter_paths={})):
                self.assertEqual(main(["--layer", "integrated", "--count", "1",
                                       "--output", str(output)]), 0)
            import json
            report = json.loads(output.read_text())
        contract = report["evaluation_layer"]
        self.assertEqual(contract["layer"], "integrated")
        self.assertEqual(contract["status"], "incomplete")
        self.assertGreater(contract["model_usage"]["fallback_turns"], 0)


if __name__ == "__main__":
    unittest.main()

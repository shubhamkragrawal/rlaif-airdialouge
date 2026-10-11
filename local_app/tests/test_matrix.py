import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_app.controller import ADAPTERS
from local_app.evaluation import EvaluationScenario
from local_app.matrix import (BASELINE_SCHEMA, compare_baseline, freeze_baseline,
                              run_matrix, run_release, _turn_evidence)
from local_app.evaluation_metrics import aggregate_evaluation_metrics
from local_app.observability import canonical_json


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "airdialogue_sampled_5k.jsonl"
SCENARIOS = [EvaluationScenario(0, ("Find a flight from IAD to LAS under $100",),
                               "book", (1027,))]


class _StubModel:
    last_usage = {"input_tokens": 8, "output_tokens": 7}

    def generate(self, _messages, _flights):
        return "I can help with simulated flights."


class _StubRuntimeManager:
    def __init__(self, paths, omit_provenance_field=None):
        self.config = SimpleNamespace(base_model="test/base", revision="rev-123",
                                      max_new_tokens=160, temperature=0.0,
                                      adapter_paths={key: str(value) for key, value in paths.items()})
        self.loaded = 0
        self.max_loaded = 0
        self.releases = 0
        self.current = None
        self.omit_provenance_field = omit_provenance_field

    def get(self, adapter, **_kwargs):
        if self.current is not None:
            raise AssertionError("previous adapter remained loaded")
        self.current = adapter
        self.loaded += 1
        self.max_loaded = max(self.max_loaded, self.loaded)
        return _StubModel()

    def status(self, adapter):
        lifecycle = "ready" if self.current == adapter else "available"
        return {"adapter": adapter, "lifecycle_status": lifecycle, "mode": "local_model",
                "available": True,
                "provenance_status": "complete" if lifecycle == "ready" else "partial"}

    def provenance(self, adapter):
        import hashlib
        lifecycle = "ready" if self.current == adapter else "available"
        result = {"adapter": adapter, "lifecycle_status": lifecycle, "mode": "local_model",
                  "available": True,
                  "provenance_status": "complete" if lifecycle == "ready" else "partial",
                  "adapter_sha256": hashlib.sha256(adapter.encode()).hexdigest(),
                  "base_model": self.config.base_model,
                  "base_model_revision": self.config.revision,
                  "generation_settings": {"max_new_tokens": self.config.max_new_tokens,
                                          "temperature": self.config.temperature},
                  "device": "cpu", "precision": "float32",
                  "peak_memory_bytes": 4096}
        if self.omit_provenance_field:
            result.pop(self.omit_provenance_field, None)
        return result

    def release(self):
        if self.current is not None:
            self.current = None
            self.loaded -= 1
        self.releases += 1


class MatrixTests(unittest.TestCase):
    def test_controller_only_matrix_runs_identical_cohort_and_records_all_arms(self):
        report = run_matrix(mode="controller_only", corpus_path=CORPUS, scenarios=SCENARIOS)
        self.assertEqual(tuple(report["adapters"]), ADAPTERS)
        self.assertEqual(report["layer"], "controller_only")
        self.assertEqual(report["cohort"]["scenario_ids"], [0])
        for adapter in ADAPTERS:
            arm = report["adapters"][adapter]
            self.assertEqual(arm["layer"], "controller_only")
            self.assertEqual(arm["scenario_ids"], [0])
            self.assertEqual(len(arm["fingerprint"]), 64)
            self.assertEqual(len(arm["scenarios"]), 1)
            self.assertIn("metrics", arm)
            self.assertEqual(arm["metrics"]["scenario_count"], 1)
            self.assertIn("validation", arm["metrics"])
            self.assertIn("repetition_evidence", arm["metrics"])
            self.assertEqual(len(arm["run_manifest"]["sha256"]), 64)
            self.assertTrue(all(len(turn["sha256"]) == 64 for turn in arm["turn_manifests"]))
            self.assertIsNone(arm["runtime_provenance"])
            self.assertIn("controller_sha256", arm["fingerprints"])
            self.assertIn("outcome_schema", arm["fingerprints"])
            self.assertIsNone(arm["run_manifest"]["record"]["adapter"]["runtime_provenance"])
            turn_evidence = arm["turn_manifests"][0]["evidence"]
            for field in ("validation_result", "validation_reason", "fallback",
                          "malformed_output", "repetitive_output",
                          "grounded_entities", "expected_no_results", "transaction_claim",
                          "total_latency_ms", "input_tokens", "output_tokens"):
                self.assertIn(field, turn_evidence)
            self.assertEqual(len(arm["turn_manifests"][0]["record_sha256"]), 64)
            manifest = arm["turn_manifests"][0]
            expected = hashlib.sha256(canonical_json({
                "record": manifest["record"], "evidence": manifest["evidence"]
            }).encode("utf-8")).hexdigest()
            self.assertEqual(manifest["sha256"], expected)
        self.assertTrue(report["scenario_comparison"]["sft"]["matches_reference"])

    def test_freeze_and_compare_versioned_controller_baseline(self):
        report = run_matrix(mode="controller_only", corpus_path=CORPUS, scenarios=SCENARIOS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline-v1.json"
            freeze_baseline(report, path)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["schema"], BASELINE_SCHEMA)
            self.assertEqual(loaded["baseline_version"], 1)
            self.assertTrue(all(value["matches"] for value in compare_baseline(report, path).values()))
            changed = run_matrix(mode="controller_only", corpus_path=CORPUS,
                                 scenarios=[EvaluationScenario(0, ("Find a flight from JFK to LAX",),
                                                               "no_flight")])
            with self.assertRaisesRegex(ValueError, "cohorts/settings"):
                compare_baseline(changed, path)

    def test_integrated_cannot_be_frozen(self):
        report = run_matrix(mode="controller_only", corpus_path=CORPUS, scenarios=SCENARIOS)
        report["mode"] = "integrated"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only controller_only"):
                freeze_baseline(report, Path(directory) / "no.json")

    def test_freeze_rejects_tampered_mixed_layer_arm(self):
        report = run_matrix(mode="controller_only", corpus_path=CORPUS, scenarios=SCENARIOS)
        report["adapters"][ADAPTERS[0]]["layer"] = "integrated"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "layer does not match"):
                freeze_baseline(report, Path(directory) / "tampered.json")

    def test_duplicate_accepted_responses_do_not_infer_repetition(self):
        scenario = EvaluationScenario(0, ("Tell me something", "Tell me something"), "no_flight")
        observed = [{"scenario_id": 0, "turn_index": index,
                     "result": {"text": "Hello", "results": [], "model_used": False},
                     "latency_ms": 1.0} for index in range(2)]
        rows, _ = _turn_evidence(
            observed, [scenario], run_id="r", adapter="sft", controller_version="v",
            generation_settings={}, adapter_metadata={}, scenario_rows={0: {}})
        self.assertEqual([row["repetitive_output"] for row in rows], [False, False])
        self.assertEqual(aggregate_evaluation_metrics([], rows)["repetition_rate"], 0.0)

    def test_show_flights_is_search_and_records_no_results_evidence(self):
        scenario = EvaluationScenario(0, ("Show flights from nowhere",), "no_flight")
        observed = [{"scenario_id": 0, "turn_index": 0,
                     "result": {"text": "No flights found", "results": [], "model_used": False},
                     "latency_ms": 1.0}]
        rows, _ = _turn_evidence(
            observed, [scenario], run_id="r", adapter="sft", controller_version="v",
            generation_settings={}, adapter_metadata={}, scenario_rows={0: {}})
        self.assertIs(rows[0]["valid_search_transition"], True)
        self.assertIs(rows[0]["expected_no_results"], True)
        self.assertIs(rows[0]["presented_no_results"], True)

    def test_integrated_requires_runtime_and_controller_mode_rejects_one(self):
        with self.assertRaisesRegex(ValueError, "requires a local runtime"):
            run_matrix(mode="integrated", scenarios=SCENARIOS)
        with self.assertRaisesRegex(ValueError, "forbids model loading"):
            run_matrix(mode="controller_only", scenarios=SCENARIOS, runtime_manager=object())

    def test_requested_layer_must_match_evaluator_contract(self):
        from local_app import matrix
        original = matrix.evaluation.build_report

        def wrong_layer(*args, **kwargs):
            report = original(*args, **kwargs)
            report["evaluation_layer"]["layer"] = "integrated"
            return report

        with patch.object(matrix.evaluation, "build_report", side_effect=wrong_layer):
            with self.assertRaisesRegex(ValueError, "does not match requested mode"):
                run_matrix(mode="controller_only", corpus_path=CORPUS, scenarios=SCENARIOS)

    def test_integrated_four_arm_matrix_has_complete_provenance_and_releases_each_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for adapter in ADAPTERS:
                path = Path(directory) / adapter
                path.mkdir()
                (path / "adapter_config.json").write_text('{"peft_type":"LORA"}')
                paths[adapter] = path
            manager = _StubRuntimeManager(paths)
            cohort = [EvaluationScenario(0, ("Tell me about flight booking",), "no_flight")]
            report = run_matrix(mode="integrated", corpus_path=CORPUS,
                                scenarios=cohort, runtime_manager=manager)
            self.assertEqual(tuple(report["adapters"]), ADAPTERS)
            self.assertEqual(manager.max_loaded, 1)
            self.assertEqual(manager.loaded, 0)
            self.assertEqual(manager.releases, 4)
            for adapter in ADAPTERS:
                arm = report["adapters"][adapter]
                self.assertEqual(arm["layer"], "integrated")
                self.assertEqual(arm["status"], "complete")
                self.assertGreater(arm["model_participation"]["turns"], 0)
                self.assertTrue(arm["runtime_provenance"]["complete"])
                self.assertEqual(arm["runtime_status"]["lifecycle_status"], "ready")
                hardware = arm["run_manifest"]["record"]["hardware"]
                self.assertEqual(hardware["device"], "cpu")
                self.assertEqual(hardware["peak_memory_bytes"], 4096)
                self.assertEqual(hardware["precision"], "float32")
                self.assertGreaterEqual(hardware["peak_process_rss_mb"], 0)
                self.assertGreaterEqual(arm["metrics"]["scenario_count"], 1)
                self.assertEqual(len(arm["turn_manifests"]), 1)
                self.assertEqual(arm["fingerprints"]["runtime_provenance_sha256"],
                                 arm["runtime_provenance"]["sha256"])

    def test_release_checkpoints_resume_and_assemble_in_temporary_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for adapter in ADAPTERS:
                path = Path(directory) / f"adapter-{adapter}"
                path.mkdir()
                (path / "adapter_config.json").write_text("{}")
                paths[adapter] = path
            manager = _StubRuntimeManager(paths)
            calls = []

            def arm_runner(**kwargs):
                adapter = kwargs["adapters"][0]
                calls.append(adapter)
                from local_app.evaluation import load_scenarios
                scenario_ids = [scenario.scenario_id for scenario in load_scenarios(CORPUS)]
                provenance = manager.provenance(adapter)
                provenance.update({"complete": True, "sha256": "a" * 64,
                                   "incomplete_fields": []})
                return {"adapters": {adapter: {
                    "layer": "integrated", "status": "complete",
                    "cohort_id": "release-cohort", "scenario_ids": scenario_ids,
                    "metrics": {"scenario_count": 50},
                    "runtime_provenance": provenance,
                    "model_participation": {"turns": 1, "required": True},
                }}}

            first = run_release(runtime_manager=manager, corpus_path=CORPUS,
                                output_dir=Path(directory) / "release", arm_runner=arm_runner)
            self.assertEqual(tuple(first["adapters"]), ADAPTERS)
            self.assertEqual(first["cohort"]["scenario_count"], 50)
            self.assertEqual(len(calls), 4)
            self.assertEqual(first["generation_settings"], {
                "mode": "model", "base_model": manager.config.base_model,
                "revision": manager.config.revision, "temperature": 0.0,
                "max_new_tokens": 160, "start": 0, "count": 50})
            self.assertTrue((Path(directory) / "release" / "manifest-v1.json").is_file())
            resumed = run_release(runtime_manager=manager, corpus_path=CORPUS,
                                  output_dir=Path(directory) / "release", arm_runner=arm_runner)
            self.assertEqual(resumed["sha256"], first["sha256"])
            self.assertEqual(len(calls), 4)

    def test_release_rejects_non_deterministic_settings_and_stale_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for adapter in ADAPTERS:
                path = Path(directory) / adapter
                path.mkdir()
                (path / "adapter_config.json").write_text("{}")
                paths[adapter] = path
            manager = _StubRuntimeManager(paths)
            with self.assertRaisesRegex(ValueError, "requires temperature=0"):
                run_release(runtime_manager=manager, output_dir=Path(directory) / "bad",
                            temperature=0.2)
            release_dir = Path(directory) / "release"
            checkpoint_dir = release_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / f"{ADAPTERS[0]}.json").write_text(json.dumps({
                "adapter": ADAPTERS[0], "arm": {"runtime_provenance": {
                    "adapter_sha256": "f" * 64}}, "corpus_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                run_release(runtime_manager=manager, corpus_path=CORPUS,
                            output_dir=release_dir, arm_runner=lambda **_kwargs: {})

    def test_idempotent_repeated_confirmation_uses_current_turn_simulator_evidence(self):
        scenario = EvaluationScenario(
            0, ("Find a flight from IAD to LAS under $100", "Passenger Amy",
                "Book flight 1027", "Book flight 1027 again"), "book", (1027,))
        report = run_matrix(mode="controller_only", corpus_path=CORPUS,
                            scenarios=[scenario])
        json.dumps(report, allow_nan=False)
        for arm in report["adapters"].values():
            confirmations = [manifest for manifest in arm["turn_manifests"]
                             if manifest["evidence"]["transaction_claim"]]
            self.assertEqual(len(confirmations), 2)
            first_ids = confirmations[0]["record"]["transaction_evidence_ids"]
            self.assertEqual(first_ids, confirmations[1]["record"]["transaction_evidence_ids"])
            self.assertTrue(first_ids[0].startswith("hold_"))
            self.assertTrue(first_ids[1].startswith("booking_"))
            for manifest in confirmations:
                evidence = manifest["evidence"]
                self.assertFalse(evidence["unsupported_transaction_claim"])
                self.assertTrue(evidence["valid_booking_transition"])
                self.assertEqual(manifest["record"]["final_outcome"], "booking")
                authoritative = evidence["authoritative_transaction_evidence"]
                self.assertTrue(authoritative["current_turn"])
                self.assertEqual(first_ids, [authoritative["hold_id"],
                                             authoritative["booking_id"]])
            self.assertNotIn("Amy", canonical_json(arm["turn_manifests"]))

    def test_continue_with_selected_flight_records_authoritative_booking_transition(self):
        scenario = EvaluationScenario(
            0, ("Find a flight from IAD to LAS under $100", "My name is Amy Gonzalez",
                "Continue with flight 1027"), "book", (1027,))
        report = run_matrix(mode="controller_only", corpus_path=CORPUS,
                            scenarios=[scenario])
        for arm in report["adapters"].values():
            final_turn = arm["turn_manifests"][-1]
            evidence = final_turn["evidence"]
            self.assertTrue(evidence["valid_booking_transition"])
            self.assertFalse(evidence["unsupported_transaction_claim"])
            transaction = evidence["authoritative_transaction_evidence"]
            self.assertTrue(transaction["hold_id"].startswith("hold_"))
            self.assertTrue(transaction["booking_id"].startswith("booking_"))
            self.assertEqual(final_turn["record"]["transaction_evidence_ids"],
                             [transaction["hold_id"], transaction["booking_id"]])

    def test_booking_metric_preserves_negation_and_non_booking_phrases(self):
        messages = ("Do not book flight 1027", "Is flight 1027 available to book?",
                    "Tell me about flight booking")
        scenario = EvaluationScenario(0, messages, "no_flight")
        observed = [{"scenario_id": 0, "turn_index": index,
                     "result": {"text": "I can help with that", "results": [],
                                "model_used": False}, "latency_ms": 1.0}
                    for index in range(len(messages))]
        rows, _ = _turn_evidence(
            observed, [scenario], run_id="r", adapter="sft", controller_version="v",
            generation_settings={}, adapter_metadata={}, scenario_rows={0: {}})
        self.assertEqual([row["valid_booking_transition"] for row in rows],
                         [False, False, None])

    def test_each_required_runtime_provenance_field_is_named_when_missing(self):
        required = ("adapter_sha256", "base_model", "base_model_revision",
                    "generation_settings", "device", "precision", "peak_memory_bytes")
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for adapter in ADAPTERS:
                path = Path(directory) / adapter
                path.mkdir()
                (path / "adapter_config.json").write_text("{}")
                paths[adapter] = path
            cohort = [EvaluationScenario(0, ("Tell me about flight booking",), "no_flight")]
            for field in required:
                with self.subTest(field=field):
                    manager = _StubRuntimeManager(paths, omit_provenance_field=field)
                    report = run_matrix(mode="integrated", corpus_path=CORPUS,
                                        scenarios=cohort, runtime_manager=manager)
                    self.assertEqual(manager.max_loaded, 1)
                    self.assertEqual(manager.loaded, 0)
                    self.assertEqual(manager.releases, 4)
                    self.assertFalse(report["baseline_freeze_allowed"])
                    for arm in report["adapters"].values():
                        self.assertEqual(arm["status"], "incomplete")
                        self.assertIn(f"runtime_provenance.{field}", arm["status_reason"])

    def test_matrix_schema_requires_manifest_metrics_and_layer_provenance_contracts(self):
        schema_path = ROOT / "artifacts" / "matrix" / "schema-v1.json"
        schema = json.loads(schema_path.read_text())
        definitions = schema["$defs"]
        required = definitions["arm"]["required"]
        for field in ("metrics", "run_manifest", "turn_manifests", "fingerprints",
                      "runtime_status", "runtime_provenance", "model_participation"):
            self.assertIn(field, required)
        self.assertEqual(len(definitions["arm"]["allOf"]), 2)
        self.assertIn("integrated", str(definitions["arm"]["allOf"][1]))
        self.assertIn("controller_only", str(definitions["arm"]["allOf"][0]))

    def test_integrated_missing_participation_and_provenance_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for adapter in ADAPTERS:
                path = Path(directory) / adapter
                path.mkdir()
                (path / "adapter_config.json").write_text("{}")
                paths[adapter] = path
            manager = _StubRuntimeManager(paths)
            manager.get = lambda _adapter, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
            manager.status = lambda adapter: {
                "adapter": adapter, "lifecycle_status": "unavailable",
                "mode": "deterministic_fallback", "available": False,
                "provenance_status": "unavailable"}
            manager.provenance = manager.status
            report = run_matrix(mode="integrated", corpus_path=CORPUS, scenarios=SCENARIOS,
                                runtime_manager=manager)
            self.assertTrue(all(arm["status"] == "incomplete" for arm in report["adapters"].values()))
            self.assertFalse(report["baseline_freeze_allowed"])
            with self.assertRaisesRegex(ValueError, "only controller_only"):
                freeze_baseline(report, Path(directory) / "integrated.json")


if __name__ == "__main__":
    unittest.main()

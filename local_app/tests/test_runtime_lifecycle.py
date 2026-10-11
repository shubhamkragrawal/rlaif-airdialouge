import json
import tempfile
import threading
import unittest
from pathlib import Path

from local_app.runtime import ModelAdapter, RuntimeConfig, RuntimeManager, RuntimeUnavailable

ENVELOPE_KEYS = {"adapter", "lifecycle_status", "mode", "available", "provenance_status"}


def assert_envelope(test, value, adapter, lifecycle_status):
    test.assertEqual({key: value[key] for key in ENVELOPE_KEYS}, {
        "adapter": adapter,
        "lifecycle_status": lifecycle_status,
        "mode": "local_model" if lifecycle_status in {"available", "loading", "ready"}
                else "deterministic_fallback",
        "available": lifecycle_status in {"available", "ready"},
        "provenance_status": {
            "ready": "complete", "available": "partial", "loading": "pending",
            "incompatible": "error", "configured": "unavailable", "unavailable": "unavailable",
        }[lifecycle_status],
    })

class StubAdapter(ModelAdapter):
    def __init__(self, name):
        self.name = name
        self.disposed = False

    def metadata(self):
        return {
            "adapter": self.name,
            "base_model": "fixture-model",
            "base_model_revision": "revision-123",
            "adapter_path": "/private/fixture/path",
            "device": "cpu",
            "precision": "float32",
            "generation_settings": {"max_new_tokens": 12, "temperature": 0},
            "peak_memory_bytes": 0,
            "api_token": "do-not-return",
        }


class RuntimeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {}
        for name in ("sft", "dpo"):
            path = self.root / name
            path.mkdir()
            (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
            (path / "weights.bin").write_bytes((name + "-weights").encode())
            self.paths[name] = str(path)

    def tearDown(self):
        self.temp.cleanup()

    def test_status_tracks_unavailable_configured_available_loading_and_ready(self):
        entered, finish = threading.Event(), threading.Event()

        def factory(name, _config):
            entered.set()
            finish.wait(1)
            return StubAdapter(name)

        manager = RuntimeManager(RuntimeConfig(adapter_paths={"sft": self.paths["sft"], "dpo": str(self.root / "missing")}), factory)
        assert_envelope(self, manager.status("ppo_checkpoint_19"), "ppo_checkpoint_19", "unavailable")
        assert_envelope(self, manager.status("dpo"), "dpo", "configured")
        assert_envelope(self, manager.status("sft"), "sft", "available")
        self.assertEqual(set(manager.status("sft")), ENVELOPE_KEYS)
        errors = []
        worker = threading.Thread(target=lambda: self._get(manager, errors))
        worker.start()
        self.assertTrue(entered.wait(1))
        assert_envelope(self, manager.adapter_status("sft"), "sft", "loading")
        finish.set()
        worker.join(1)
        self.assertFalse(errors)
        assert_envelope(self, manager.status("sft"), "sft", "ready")

    @staticmethod
    def _get(manager, errors):
        try:
            manager.get("sft")
        except Exception as exc:
            errors.append(exc)

    def test_switch_releases_old_adapter_before_constructing_new_one(self):
        instances = []
        manager = None

        def factory(name, _config):
            if name == "dpo":
                self.assertIsNone(manager._runtime)
                self.assertTrue(instances[0].disposed)
            instance = StubAdapter(name)
            instances.append(instance)
            return instance

        manager = RuntimeManager(RuntimeConfig(adapter_paths=self.paths), factory)
        first = manager.get("sft")
        original_dispose = manager._dispose

        def mark_disposed(runtime):
            if runtime is first:
                runtime.disposed = True
            original_dispose(runtime)

        manager._dispose = mark_disposed
        second = manager.get("dpo")
        self.assertIs(manager._runtime, second)
        assert_envelope(self, manager.status("sft"), "sft", "available")

    def test_provenance_includes_hash_and_runtime_facts_without_paths_or_secrets(self):
        manager = RuntimeManager(RuntimeConfig(adapter_paths=self.paths), lambda name, _config: StubAdapter(name))
        manager.get("sft")
        provenance = manager.provenance()
        assert_envelope(self, provenance, "sft", "ready")
        self.assertTrue(ENVELOPE_KEYS.issubset(provenance))
        self.assertEqual(provenance["base_model_revision"], "revision-123")
        self.assertEqual(len(provenance["adapter_sha256"]), 64)
        self.assertEqual(provenance["precision"], "float32")
        self.assertEqual(provenance["peak_memory_bytes"], 0)
        self.assertNotIn("adapter_path", provenance)
        self.assertNotIn("api_token", provenance)

    def test_incompatible_config_and_cancelled_load_are_reported(self):
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "adapter_config.json").write_text("not-json")
        manager = RuntimeManager(RuntimeConfig(adapter_paths={"sft": str(bad)}), lambda *_: self.fail("factory called"))
        assert_envelope(self, manager.status("sft"), "sft", "incompatible")
        with self.assertRaises(RuntimeUnavailable):
            manager.get("sft")

        cancelled = threading.Event()
        cancelled.set()
        manager = RuntimeManager(RuntimeConfig(adapter_paths=self.paths), lambda *_: self.fail("factory called"))
        with self.assertRaisesRegex(RuntimeUnavailable, "session_time_exceeded"):
            manager.get("sft", cancel_event=cancelled)
        assert_envelope(self, manager.status("sft"), "sft", "available")

    def test_load_errors_do_not_expose_backend_secrets(self):
        manager = RuntimeManager(
            RuntimeConfig(adapter_paths=self.paths),
            lambda *_: (_ for _ in ()).throw(RuntimeError("api_token=private-value")),
        )
        with self.assertRaises(RuntimeUnavailable):
            manager.get("sft")
        status = manager.status("sft")
        self.assertEqual(set(status), ENVELOPE_KEYS)
        self.assertEqual(status["provenance_status"], "error")
        provenance = manager.provenance("sft")
        self.assertNotIn("private-value", provenance["error"])
        self.assertEqual(status["adapter"], "sft")
        self.assertEqual(status["lifecycle_status"], "available")

    def test_requested_identity_is_preserved_across_status_and_provenance(self):
        manager = RuntimeManager(RuntimeConfig(adapter_paths=self.paths), lambda name, _config: StubAdapter(name))
        manager.get("sft")
        assert_envelope(self, manager.status("dpo"), "dpo", "available")
        requested = manager.provenance("dpo")
        assert_envelope(self, requested, "dpo", "available")
        self.assertNotEqual(requested["adapter"], manager.provenance()["adapter"])


if __name__ == "__main__":
    unittest.main()

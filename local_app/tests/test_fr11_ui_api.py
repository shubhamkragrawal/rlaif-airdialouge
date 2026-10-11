"""FR-11 runtime status API contract checks."""

import unittest

from local_app.api import SessionAPI, _BROWSER_HTML
from local_app.controller import ConversationController
from local_app.server import _runtime_status_provider


class FR11StatusAPITests(unittest.TestCase):
    def test_all_lifecycle_states_survive_the_api_contract(self):
        states = {
            "loading": ("local_model", False, "pending"),
            "ready": ("local_model", True, "complete"),
            "available": ("local_model", True, "partial"),
            "configured": ("deterministic_fallback", False, "unavailable"),
            "unavailable": ("deterministic_fallback", False, "unavailable"),
            "incompatible": ("deterministic_fallback", False, "error"),
        }
        for lifecycle, (mode, available, provenance) in states.items():
            with self.subTest(lifecycle=lifecycle):
                api = SessionAPI(ConversationController(), runtime_status_provider=lambda _adapter: {
                    "lifecycle_status": lifecycle, "mode": mode, "available": available,
                    "provenance_status": provenance,
                })
                self.assertEqual(api.runtime_status("dpo"), {
                    "adapter": "dpo", "lifecycle_status": lifecycle, "mode": mode,
                    "available": available, "provenance_status": provenance,
                })

    def test_configured_lifecycle_is_labeled_as_configured_in_ui(self):
        self.assertIn("configured:'Configured · deterministic fallback (model files unavailable)'", _BROWSER_HTML)
        self.assertIn("runtimeStatus.textContent=", _BROWSER_HTML)

    def test_runtime_status_returns_injectable_contract_for_selected_adapter(self):
        calls = []

        def runtime_status(adapter):
            calls.append(adapter)
            return {"lifecycle_status": "ready", "mode": "local_model", "available": True,
                    "provenance_status": "complete", "detail": "must not be returned"}

        api = SessionAPI(ConversationController(), runtime_status_provider=runtime_status)
        payload = api.runtime_status("ppo_adaptive_50")

        self.assertEqual(calls, ["ppo_adaptive_50"])
        self.assertEqual(payload, {
            "adapter": "ppo_adaptive_50", "lifecycle_status": "ready", "mode": "local_model",
            "available": True, "provenance_status": "complete",
        })

    def test_default_contract_reports_deterministic_fallback(self):
        api = SessionAPI(ConversationController())
        self.assertEqual(api.runtime_status("dpo"), {
            "adapter": "dpo", "lifecycle_status": "unavailable",
            "mode": "deterministic_fallback", "available": False,
            "provenance_status": "unavailable",
        })

    def test_invalid_provider_output_is_rejected(self):
        api = SessionAPI(ConversationController(), runtime_status_provider=lambda _adapter: None)
        with self.assertRaisesRegex(ValueError, "Runtime status is unavailable"):
            api.runtime_status("sft")

    def test_server_provider_normalizes_runtime_manager_envelope(self):
        class MockRuntimeManager:
            def status(self, adapter):
                return {"adapter": adapter, "lifecycle_status": "incompatible", "mode": "deterministic_fallback",
                        "available": False, "provenance_status": "error", "error": "private path /secret"}

            def provenance(self, adapter):
                return {"adapter": adapter, "provenance_status": "error", "adapter_path": "/secret"}

        status = _runtime_status_provider(MockRuntimeManager())("sft")
        self.assertEqual(status, {
            "adapter": "sft", "lifecycle_status": "incompatible",
            "mode": "deterministic_fallback", "available": False,
            "provenance_status": "error",
        })

    def test_server_provider_preserves_configured_lifecycle(self):
        class MockRuntimeManager:
            def status(self, adapter):
                return {"adapter": adapter, "lifecycle_status": "configured", "mode": "deterministic_fallback",
                        "available": False, "provenance_status": "unavailable"}

            def provenance(self, adapter):
                return {"adapter": adapter, "lifecycle_status": "configured", "provenance_status": "unavailable"}

        self.assertEqual(_runtime_status_provider(MockRuntimeManager())("dpo"), {
            "adapter": "dpo", "lifecycle_status": "configured",
            "mode": "deterministic_fallback", "available": False,
            "provenance_status": "unavailable",
        })


if __name__ == "__main__":
    unittest.main()

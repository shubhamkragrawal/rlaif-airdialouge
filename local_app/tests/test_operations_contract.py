"""Release-facing HTTP and bind contracts."""

import json
from contextlib import redirect_stdout
from io import BytesIO
import io
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_app.api import SessionAPI, make_server, normalize_loopback_host
from local_app.controller import ADAPTERS, ConversationController
from local_app.server import main
from local_app.runtime import RuntimeConfig, TransformersAdapter


class OperationsContractTests(unittest.TestCase):
    def test_only_documented_loopback_hosts_are_accepted(self):
        for host, normalized in (("127.0.0.1", "127.0.0.1"),
                                 ("localhost", "127.0.0.1"), ("::1", "::1")):
            with self.subTest(host=host):
                self.assertEqual(normalize_loopback_host(host), normalized)
        for host in ("0.0.0.0", "192.0.2.10", "example.com", "127.0.0.2", "::"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                make_server(host, 0)

    def test_inventory_and_default_status_are_safe_and_complete(self):
        self.assertEqual(tuple(ADAPTERS), (
            "sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50",
        ))
        api = SessionAPI(ConversationController())
        self.assertEqual(api.runtime_status("sft"), {
            "adapter": "sft", "lifecycle_status": "unavailable",
            "mode": "deterministic_fallback", "available": False,
            "provenance_status": "unavailable",
        })

    def test_make_server_normalizes_loopback_before_binding(self):
        with patch("local_app.api.ThreadingHTTPServer") as server_type:
            server_type.return_value = SimpleNamespace()
            make_server("localhost", 8000)
        self.assertEqual(server_type.call_args.args[0], ("127.0.0.1", 8000))

    def test_health_inventory_and_runtime_status_http_contracts(self):
        with patch("local_app.api.ThreadingHTTPServer") as server_type:
            server_type.return_value = SimpleNamespace()
            make_server("127.0.0.1", 0, SessionAPI(ConversationController()))
        handler_type = server_type.call_args.args[1]

        def get(path):
            handler = handler_type.__new__(handler_type)
            handler.path = path
            handler.wfile = BytesIO()
            handler.response_headers = {}
            handler.send_response = lambda status: setattr(handler, "response_status", status)
            handler.send_header = lambda name, value: handler.response_headers.__setitem__(name, value)
            handler.end_headers = lambda: None
            handler.do_GET()
            return handler.response_status, json.loads(handler.wfile.getvalue())

        self.assertEqual(get("/health"), (200, {"status": "ok", "simulator": True}))
        self.assertEqual(get("/adapters"), (200, {"adapters": list(ADAPTERS)}))
        self.assertEqual(get("/runtime-status?adapter=sft"), (200, {
            "adapter": "sft", "lifecycle_status": "unavailable",
            "mode": "deterministic_fallback", "available": False,
            "provenance_status": "unavailable",
        }))

    def test_deterministic_and_model_server_start_modes(self):
        fake_server = SimpleNamespace(serve_forever=lambda: None, server_close=lambda: None)
        with patch("local_app.server.make_server", return_value=fake_server) as make, \
                redirect_stdout(io.StringIO()):
            main(["--port", "0"])
        deterministic_api = make.call_args.args[2]
        self.assertIsNone(deterministic_api.controller.model_loader)
        self.assertEqual(deterministic_api.runtime_status("sft")["mode"], "deterministic_fallback")

        fake_get = lambda _adapter: None
        fake_manager = SimpleNamespace(get=fake_get, release=lambda: None,
                                       status=lambda _adapter: {}, provenance=lambda _adapter: {})
        with patch("local_app.runtime.RuntimeManager", return_value=fake_manager), \
                patch("local_app.server.make_server", return_value=fake_server) as make, \
                redirect_stdout(io.StringIO()):
            main(["--model", "--host", "::1", "--port", "0"])
        model_api = make.call_args.args[2]
        self.assertIs(model_api.controller.model_loader, fake_get)
        self.assertIsNotNone(model_api.runtime_status_provider)
        self.assertEqual(make.call_args.args[:2], ("::1", 0))

    def test_model_runtime_resolves_huggingface_files_locally_only(self):
        calls = []

        class FakeModel:
            def to(self, _device):
                return self

            def eval(self):
                return self

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls.append(("tokenizer", source, kwargs))
                return object()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(source, **kwargs):
                calls.append(("model", source, kwargs))
                return FakeModel()

        class FakePEFT:
            @staticmethod
            def from_pretrained(model, _path):
                return model

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = FakePEFT
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizer
        fake_transformers.AutoModelForCausalLM = FakeAutoModel

        with tempfile.TemporaryDirectory() as adapter_path, patch.dict(sys.modules, {
            "torch": fake_torch, "peft": fake_peft, "transformers": fake_transformers,
        }):
            TransformersAdapter("sft", RuntimeConfig(
                adapter_paths={"sft": adapter_path}, revision=None,
            ))

        self.assertEqual([call[0] for call in calls], ["tokenizer", "model"])
        self.assertTrue(all(call[2]["local_files_only"] is True for call in calls))


if __name__ == "__main__":
    unittest.main()

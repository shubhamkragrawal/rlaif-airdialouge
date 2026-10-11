"""Operational contracts for the local server and documented HTTP surface."""

import unittest
from unittest.mock import Mock, patch

from local_app.api import SessionAPI, normalize_loopback_host
from local_app.controller import ADAPTERS, ConversationController
from local_app import server as server_module


class OperationalContractsTests(unittest.TestCase):
    def test_only_documented_loopback_addresses_are_accepted(self):
        for host, expected in (("127.0.0.1", "127.0.0.1"), ("localhost", "127.0.0.1"), ("::1", "::1")):
            with self.subTest(host=host):
                self.assertEqual(normalize_loopback_host(host), expected)
        for host in ("0.0.0.0", "::", "192.168.1.10", "8.8.8.8", "example.com", ""):
            with self.subTest(host=host), self.assertRaises(ValueError):
                normalize_loopback_host(host)

    def test_default_inventory_and_status_are_safe_and_complete(self):
        api = SessionAPI(ConversationController())
        self.assertEqual(ADAPTERS, ("sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50"))
        status = api.runtime_status("sft")
        self.assertEqual(status, {
            "adapter": "sft", "lifecycle_status": "unavailable",
            "mode": "deterministic_fallback", "available": False,
            "provenance_status": "unavailable",
        })
        self.assertFalse(any("path" in key for key in status))

    def test_model_startup_is_lazy_and_does_not_load_or_download(self):
        manager = Mock()
        fake_server = Mock()
        with patch("local_app.runtime.RuntimeManager", return_value=manager), \
                patch.object(server_module, "make_server", return_value=fake_server):
            server_module.main(["--model"])
        manager.get.assert_not_called()
        fake_server.serve_forever.assert_called_once_with()
        fake_server.server_close.assert_called_once_with()
        manager.release.assert_called_once_with()

    def test_non_loopback_is_rejected_before_model_startup(self):
        with patch("local_app.runtime.RuntimeManager") as manager:
            with self.assertRaises(SystemExit) as error:
                server_module.main(["--host", "0.0.0.0", "--model"])
        self.assertEqual(error.exception.code, 2)
        manager.assert_not_called()


if __name__ == "__main__":
    unittest.main()

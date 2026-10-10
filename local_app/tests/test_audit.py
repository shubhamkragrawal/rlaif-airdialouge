import tempfile
import unittest
from pathlib import Path

from local_app.audit import AuditLog


class AuditTests(unittest.TestCase):
    def test_sensitive_values_are_redacted_and_payment_fields_are_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            audit.write("message", {
                "text": "My name is Amy Gonzalez, email amy@example.com, phone 212-555-0199, card 4111 1111 1111 1111.",
                "payment": "secret",
            })
            content = path.read_text()
        self.assertNotIn("Amy Gonzalez", content)
        self.assertNotIn("amy@example.com", content)
        self.assertNotIn("212-555-0199", content)
        self.assertNotIn("4111 1111 1111 1111", content)
        self.assertNotIn("secret", content)
        self.assertIn("[REDACTED]", content)


if __name__ == "__main__":
    unittest.main()

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

    def test_unicode_and_variable_whitespace_echoes_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            audit.write("message", {"text": "My name is José García"})
            audit.write("echo", {"text": "Passenger JOSE\u0301\t\tGARCÍA; Passenger José\t\tGarcía"})
            content = path.read_text(encoding="utf-8")
        self.assertNotIn("José García", content)
        self.assertNotIn("JOSE\u0301", content)
        self.assertNotIn("GARCÍA", content)

    def test_unicode_dash_variants_and_multiple_passengers_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            audit.write("message", {"text": "My name is Amy Gonzalez. Passenger José García"})
            audit.write("echo", {"text": "Amy\u2003Gonzalez; Jose\u0301   García"})
            audit.write("dash", {"text": "Passenger Jean‒Luc Picard"})
            audit.register_passenger("Jean-Luc Picard")
            audit.write("dash_echo", {"text": "Jean−Luc\tPicard"})
            content = path.read_text(encoding="utf-8")
        for candidate in ("Amy Gonzalez", "José García", "Jose\u0301", "Jean‒Luc", "Jean−Luc"):
            self.assertNotIn(candidate, content)

    def test_rejected_unicode_candidate_is_not_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            audit.write("message", {"text": "My name is अनन्या   शर्मा"})
            content = path.read_text(encoding="utf-8")
        self.assertNotIn("अनन्या", content)
        self.assertNotIn("शर्मा", content)
        self.assertIn("[REDACTED]", content)

    def test_all_passengers_in_mixed_free_text_fields_are_redacted(self):
        messages = (
            "My name is Amy Gonzalez, José García is traveling",
            "My name is Amy Gonzalez. José García will be traveling",
            "José García is travelling",
            "Passengers Amy Gonzalez and José García are traveling",
            "Search IAD to LAS for Amy Gonzalez. Book for José García",
            "Search IAD to LAS for Amy Gonzalez and for José García",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            for message in messages:
                audit.write("message", {"message": message, "text": f"Echo: {message}"})
            content = path.read_text(encoding="utf-8")
        for passenger in ("Amy Gonzalez", "José García"):
            self.assertNotIn(passenger, content)
        self.assertIn("IAD to LAS", content)
        self.assertIn("traveling", content)

    def test_traveling_passenger_redaction_preserves_airport_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            AuditLog(path).write("message", {
                "text": "Passengers Amy Gonzalez and José García are traveling from JFK to LAX"
            })
            content = path.read_text(encoding="utf-8")
        self.assertNotIn("Amy Gonzalez", content)
        self.assertNotIn("José García", content)
        self.assertIn("JFK to LAX", content)

    def test_unicode_traveling_expressions_are_fully_redacted(self):
        messages = (
            "Passengers Amy Gonzalez and Jean‑Luc Picard are traveling from JFK to LAX",
            "José García is travelling after Amy Gonzalez",
            "Passengers अमित कुमार and Jean‑Luc Picard are traveling through IAD",
            "Passengers Amy Gonzalez, Jean‑Luc Picard, and अमित कुमार are traveling",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            for message in messages:
                audit.write("message", {"text": message})
            content = path.read_text(encoding="utf-8")
        for fragment in ("Amy", "Gonzalez", "Jean", "Luc", "Picard",
                         "José", "García", "अमित", "कुमार"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, content)
        self.assertIn("JFK to LAX", content)
        self.assertIn("IAD", content)
        self.assertIn("traveling", content)
        self.assertIn("travelling", content)


if __name__ == "__main__":
    unittest.main()

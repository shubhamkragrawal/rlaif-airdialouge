import unittest

from local_app.guardrails import (
    TransactionClaimError,
    detect_transaction_claim,
    validate_transaction_claim,
)


class TransactionGuardrailTests(unittest.TestCase):
    def test_detects_transaction_wording_and_paraphrases(self):
        claims = (
            "Your flight has been booked.",
            "The reservation is confirmed.",
            "I've secured your seat.",
            "Your ticket is all set.",
            "I've issued your itinerary.",
            "You're all set for your trip.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertTrue(detect_transaction_claim(claim))

    def test_does_not_classify_search_or_instruction_as_completed_transaction(self):
        self.assertFalse(detect_transaction_claim("I found three available flights."))
        self.assertFalse(detect_transaction_claim("Would you like me to book this flight?"))

    def test_claim_without_authoritative_booking_evidence_has_machine_reason(self):
        with self.assertRaises(TransactionClaimError) as raised:
            validate_transaction_claim("Your booking is confirmed and simulated.")
        self.assertEqual(raised.exception.reason, "unsupported_transaction_claim")
        self.assertEqual(str(raised.exception), "unsupported_transaction_claim")

    def test_authoritative_evidence_still_requires_simulator_disclosure(self):
        evidence = self._evidence()
        with self.assertRaises(TransactionClaimError):
            validate_transaction_claim("Your booking is confirmed.", evidence)

    def test_authoritative_booking_and_simulator_disclosure_are_accepted(self):
        evidence = self._evidence()
        self.assertIsNone(
            validate_transaction_claim("Your simulated booking is confirmed.", evidence)
        )

    @staticmethod
    def _evidence():
        flight = {"flight_number": 1027}
        hold = {"hold_id": "hold-123", "flight": flight, "passenger": "Amy Gonzalez",
                "simulated": True, "status": "held"}
        return {"booking_id": "booking-123", "hold_id": "hold-123", "flight": flight,
                "flight_id": 1027, "passenger": "Amy Gonzalez", "simulated": True,
                "status": "confirmed", "hold": hold}

    def test_non_transactional_response_needs_no_booking_evidence(self):
        self.assertIsNone(validate_transaction_claim("I found three simulated flights."))


if __name__ == "__main__":
    unittest.main()

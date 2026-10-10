import unittest
from dataclasses import FrozenInstanceError

from local_app.knowledge import FlightKnowledgeBase
from local_app.outcomes import (
    OUTCOME_SCHEMA,
    OUTCOME_SCHEMA_VERSION,
    Intent,
    make_turn_outcome,
)
from local_app.tools import FlightSimulator


class TurnOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.simulator = FlightSimulator(FlightKnowledgeBase(scenario_id=0))

    def test_outcome_is_versioned_immutable_and_has_fallback_field(self):
        result = make_turn_outcome(
            intent=Intent.SEARCH,
            deterministic_response="I found flights.",
            search_results=self.simulator.search_flights(self._criteria()),
            fallback_reason="model_unavailable",
        )

        self.assertEqual(result.schema, OUTCOME_SCHEMA)
        self.assertEqual(result.schema_version, OUTCOME_SCHEMA_VERSION)
        self.assertEqual(result.fallback_reason, "model_unavailable")
        with self.assertRaises(FrozenInstanceError):
            result.intent = Intent.REFUSE

    def test_booking_claims_derive_from_simulator_result_and_require_disclosure(self):
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        booking = self.simulator.book_flight(hold["hold_id"])
        result = make_turn_outcome(
            intent=Intent.BOOK,
            deterministic_response="Your simulated booking is confirmed.",
            hold_result=hold,
            booking_result=booking,
        )

        self.assertEqual(result.authoritative_claims.booking_id, booking["booking_id"])
        self.assertEqual(result.authoritative_claims.booking_status, "confirmed")
        self.assertEqual(result.authoritative_claims.selected_flight_id, 1027)
        self.assertTrue(result.authoritative_claims.simulated)
        self.assertTrue(result.simulator_disclosure_required)

    def test_transaction_evidence_is_deeply_immutable_snapshot(self):
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        booking = self.simulator.book_flight(hold["hold_id"])
        result = make_turn_outcome(
            intent=Intent.BOOK,
            deterministic_response="Your simulated booking is confirmed.",
            hold_result=hold,
            booking_result=booking,
        )
        hold["flight"]["flight_number"] = 9999
        booking["flight"]["flight_number"] = 8888
        self.assertEqual(result.booking_result["flight"]["flight_number"], 1027)
        self.assertEqual(result.hold_result["flight"]["flight_number"], 1027)
        self.assertEqual(result.authoritative_claims.selected_flight_id, 1027)
        with self.assertRaises(TypeError):
            result.booking_result["flight"]["flight_number"] = 7777
        with self.assertRaises(TypeError):
            result.booking_result._data = {}
        with self.assertRaises(TypeError):
            del result.booking_result._data

    def test_hold_claims_are_checked_by_replace(self):
        from dataclasses import replace
        from local_app.outcomes import AuthoritativeClaims
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        result = make_turn_outcome(intent=Intent.HOLD,
                                   deterministic_response="A simulated hold is ready.",
                                   hold_result=hold)
        with self.assertRaises(ValueError):
            replace(result, authoritative_claims=AuthoritativeClaims(selected_flight_id=9999))
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        booking = self.simulator.book_flight(hold["hold_id"])
        booked = make_turn_outcome(intent=Intent.BOOK,
                                   deterministic_response="Your simulated booking is confirmed.",
                                   hold_result=hold, booking_result=booking)
        with self.assertRaises(ValueError):
            replace(booked, authoritative_claims=AuthoritativeClaims(selected_flight_id=9999))

    def test_rejects_transaction_result_with_wrong_intent_or_missing_disclosure(self):
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        with self.assertRaises(ValueError):
            make_turn_outcome(
                intent=Intent.SEARCH,
                deterministic_response="", hold_result=hold,
            )

    @staticmethod
    def _criteria():
        from local_app.tools import SearchCriteria

        return SearchCriteria()


if __name__ == "__main__":
    unittest.main()

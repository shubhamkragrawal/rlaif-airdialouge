import unittest

from local_app.controller import ConversationController
from local_app.guardrails import validate_grounding
from local_app.knowledge import FlightKnowledgeBase
from local_app.tools import FlightSimulator, SearchCriteria


class LocalAppTests(unittest.TestCase):
    def setUp(self):
        self.kb = FlightKnowledgeBase(scenario_id=0)
        self.sim = FlightSimulator(self.kb)

    def test_search_is_deterministic_and_filters_structured_fields(self):
        criteria = SearchCriteria(origin="IAD", destination="LAS", departure_month="Jan",
                                  departure_day="1", return_month="Jan", return_day="3",
                                  max_price=100, max_connections=1)
        first = self.sim.search_flights(criteria).flights
        second = self.sim.search_flights(criteria).flights
        self.assertEqual(first, second)
        self.assertEqual([f["flight_number"] for f in first], [1027])

    def test_booking_rejects_flight_outside_current_kb(self):
        with self.assertRaisesRegex(ValueError, "not in the local"):
            self.sim.hold_flight(9999, "Amy Gonzalez")

    def test_booking_is_idempotent(self):
        hold = self.sim.hold_flight(1027, "Amy Gonzalez")
        first = self.sim.book_flight(hold["hold_id"])
        second = self.sim.book_flight(hold["hold_id"])
        self.assertEqual(first, second)
        self.assertTrue(first["simulated"])

    def test_controller_grounded_search_then_booking(self):
        controller = ConversationController(self.kb)
        session = controller.new_session("dpo")
        result = controller.handle(session, "Find a flight from IAD to LAS under $100")
        self.assertIn("1027", result["text"])
        controller.handle(session, "My name is Amy Gonzalez")
        result = controller.handle(session, "Please book flight 1027")
        self.assertIn("confirmed", result["text"])
        self.assertEqual(session.booking["flight"]["flight_number"], 1027)

    def test_grounding_rejects_unknown_ids(self):
        with self.assertRaises(ValueError):
            validate_grounding("Flight 9999 is available", self.kb.all())


if __name__ == "__main__":
    unittest.main()

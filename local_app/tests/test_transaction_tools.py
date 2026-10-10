import unittest

from local_app.knowledge import FlightKnowledgeBase
from local_app.tools import BookingResult, FlightSimulator, HoldResult


class TransactionToolTests(unittest.TestCase):
    def setUp(self):
        self.simulator = FlightSimulator(FlightKnowledgeBase(scenario_id=0))

    def test_repeated_hold_is_stable_and_reports_held_status(self):
        first = self.simulator.hold_flight(1027, "Amy Gonzalez")
        repeated = self.simulator.hold_flight(1027, " Amy Gonzalez ")

        self.assertIsInstance(first, dict)
        self.assertEqual(first["hold_id"], repeated["hold_id"])
        self.assertEqual(first["status"], "held")
        self.assertEqual(repeated["status"], "held")
        self.assertEqual(len(self.simulator.holds), 1)
        self.assertEqual(self.simulator.holds[first["hold_id"]]["status"], "held")

    def test_hold_id_is_deterministic_across_simulators(self):
        another = FlightSimulator(FlightKnowledgeBase(scenario_id=0))
        first = self.simulator.hold_flight(1027, "Amy Gonzalez")
        second = another.hold_flight(1027, "Amy Gonzalez")

        self.assertEqual(first["hold_id"], second["hold_id"])
        self.assertEqual(first["flight"], second["flight"])

    def test_hold_result_mutation_does_not_change_stored_hold(self):
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        hold["passenger"] = "Different Passenger"
        hold["flight"]["flight_number"] = -1

        stored = self.simulator.holds[hold["hold_id"]]
        self.assertEqual(stored["passenger"], "Amy Gonzalez")
        self.assertEqual(stored["flight"]["flight_number"], 1027)
        repeated = self.simulator.hold_flight(1027, "Amy Gonzalez")
        self.assertEqual(repeated["status"], "held")
        self.simulator.holds[hold["hold_id"]]["passenger"] = "Conflicting Passenger"
        with self.assertRaises(RuntimeError):
            self.simulator.hold_flight(1027, "Amy Gonzalez")

    def test_repeated_booking_reuses_record_without_mutating_it(self):
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        first = self.simulator.book_flight(hold["hold_id"])
        persisted = self.simulator.bookings[first["booking_id"]].copy()
        repeated = self.simulator.book_flight(hold["hold_id"])

        self.assertEqual(first["booking_id"], repeated["booking_id"])
        self.assertEqual(first["status"], "confirmed")
        self.assertEqual(repeated["status"], "confirmed")
        self.assertEqual(self.simulator.bookings[first["booking_id"]], persisted)
        self.assertEqual(len(self.simulator.bookings), 1)

    def test_booking_rejects_invalid_persisted_hold(self):
        hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        self.simulator.holds[hold["hold_id"]]["status"] = "expired"
        with self.assertRaises(ValueError):
            self.simulator.book_flight(hold["hold_id"])
        self.assertEqual(self.simulator.bookings, {})
        self.simulator.holds[hold["hold_id"]] = {
            **hold, "status": "held", "flight": {"flight_number": 9999},
        }
        with self.assertRaises(ValueError):
            self.simulator.book_flight(hold["hold_id"])
        self.assertEqual(self.simulator.bookings, {})

    def test_transaction_results_have_explicit_typed_shapes(self):
        hold: HoldResult = self.simulator.hold_flight(1027, "Amy Gonzalez")
        booking: BookingResult = self.simulator.book_flight(hold["hold_id"])

        self.assertTrue(hold["simulated"])
        self.assertTrue(booking["simulated"])
        self.assertEqual(booking["hold_id"], hold["hold_id"])

    def test_different_passengers_do_not_share_a_hold_or_booking(self):
        first_hold = self.simulator.hold_flight(1027, "Amy Gonzalez")
        second_hold = self.simulator.hold_flight(1027, "Jordan Lee")
        first_booking = self.simulator.book_flight(first_hold["hold_id"])
        second_booking = self.simulator.book_flight(second_hold["hold_id"])

        self.assertNotEqual(first_hold["hold_id"], second_hold["hold_id"])
        self.assertNotEqual(first_booking["booking_id"], second_booking["booking_id"])
        self.assertEqual(len(self.simulator.bookings), 2)


if __name__ == "__main__":
    unittest.main()

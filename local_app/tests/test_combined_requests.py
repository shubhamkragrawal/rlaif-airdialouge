import unittest
from dataclasses import FrozenInstanceError

from local_app.request_parsing import FlightSelection, ParsedRequest, parse_request


class CombinedRequestParsingTests(unittest.TestCase):
    def test_combined_request_extracts_all_fields(self):
        result = parse_request(
            "Search IAD to LAS, return 1/5, nonstop, under $420, "
            "depart 1/2, passenger Amy Gonzalez, flight 1027"
        )
        self.assertEqual((result.origin, result.destination), ("IAD", "LAS"))
        self.assertEqual((result.departure_month, result.departure_day), ("Jan", "2"))
        self.assertEqual((result.return_month, result.return_day), ("Jan", "5"))
        self.assertEqual(result.max_connections, 0)
        self.assertEqual(result.max_price, 420)
        self.assertEqual(result.passenger.name, "Amy Gonzalez")
        self.assertEqual(result.flight_selections, (FlightSelection("1027", 1027),))
        self.assertTrue(result.search_intent)
        self.assertFalse(result.booking_intent)
        self.assertTrue({"origin", "destination", "departure_date", "return_date", "passenger"}
                        <= result.provided)

    def test_booking_and_search_may_be_requested_in_one_turn(self):
        result = parse_request("Please book flight 1027 for José O’Neil and find flights from IAD to LAS under $500")
        self.assertTrue(result.search_intent)
        self.assertTrue(result.booking_intent)
        self.assertEqual(result.passenger.name, "José O'Neil")
        self.assertEqual(result.flight_selections, (FlightSelection("1027", 1027),))
        self.assertEqual(result.max_price, 500)
        self.assertFalse(result.incomplete)

    def test_partial_requests_report_missing_route_without_inventing_values(self):
        result = parse_request("Find flights from IAD under $200")
        self.assertEqual(result.origin, "IAD")
        self.assertIsNone(result.destination)
        self.assertTrue(result.incomplete)
        self.assertIn("incomplete_route", result.ambiguities)
        self.assertEqual(result.provided, frozenset({"origin", "max_price", "search_intent"}))

    def test_reordered_route_and_date_clauses(self):
        result = parse_request("Return 2/10; for Amy Gonzalez, fly to LAS from IAD departing 2/3")
        self.assertEqual((result.origin, result.destination), ("IAD", "LAS"))
        self.assertEqual((result.departure_month, result.departure_day), ("Feb", "3"))
        self.assertEqual((result.return_month, result.return_day), ("Feb", "10"))

    def test_connection_and_price_conflicts_are_explicit(self):
        result = parse_request("Search IAD to LAS under $200 but below $100, at most 1 connection and nonstop")
        self.assertEqual(result.max_price, 200)
        self.assertEqual(result.max_connections, 1)
        self.assertIn("conflicting_max_price", result.conflicts)
        self.assertIn("conflicting_max_connections", result.conflicts)

    def test_airport_codes_are_not_inferred_from_passenger_names_or_lowercase_tokens(self):
        result = parse_request("Search from iad to las for Amy Gonzalez")
        self.assertIsNone(result.origin)
        self.assertIsNone(result.destination)
        self.assertEqual(result.passenger.name, "Amy Gonzalez")

    def test_explicit_same_airport_route_is_preserved_for_fixture_validation(self):
        result = parse_request("Find SFO to SFO flights")
        self.assertEqual((result.origin, result.destination), ("SFO", "SFO"))
        self.assertNotIn("same_origin_and_destination", result.conflicts)

    def test_booking_without_passenger_is_ambiguous(self):
        result = parse_request("Book flight 1027")
        self.assertTrue(result.booking_intent)
        self.assertIn("missing_passenger", result.ambiguities)

    def test_result_is_frozen(self):
        result = ParsedRequest(origin="IAD")
        with self.assertRaises(FrozenInstanceError):
            result.origin = "LAS"

    def test_later_never_mind_withdraws_booking_authorization(self):
        result = parse_request("Book flight 1027 for Amy Gonzalez. Never mind.")
        self.assertFalse(result.booking_intent)
        self.assertTrue(result.selection_intent)
        self.assertFalse(result.search_intent)

    def test_distinct_explicit_flight_selections_conflict(self):
        result = parse_request("Select flight 1027 and flight 2044")
        self.assertEqual(result.flight_selections, (
            FlightSelection("1027", 1027), FlightSelection("2044", 2044),
        ))
        self.assertIn("conflicting_flight_selections", result.conflicts)
        self.assertFalse(result.can_execute)

    def test_uppercase_passenger_is_not_misread_as_airport_code(self):
        result = parse_request("Book flight 1027 for AMY GONZALEZ")
        self.assertEqual(result.passenger.status, "found")
        self.assertEqual(result.passenger.name, "Amy Gonzalez")
        self.assertIsNone(result.origin)
        self.assertIsNone(result.destination)

    def test_booking_only_does_not_imply_search_or_incomplete_route(self):
        result = parse_request("Book flight 1027 for Amy Gonzalez")
        self.assertTrue(result.booking_intent)
        self.assertTrue(result.selection_intent)
        self.assertFalse(result.search_intent)
        self.assertFalse(result.incomplete)

    def test_repeated_identical_dates_are_not_a_conflict(self):
        result = parse_request("Find IAD to LAS departing 3/14 and returning 3/14")
        self.assertEqual((result.departure_month, result.departure_day), ("Mar", "14"))
        self.assertEqual((result.return_month, result.return_day), ("Mar", "14"))
        self.assertNotIn("conflicting_dates", result.conflicts)

    def test_departure_and_return_time_constraints_match_search_criteria(self):
        result = parse_request("Find IAD to LAS departing in the morning, return in the evening")
        self.assertEqual(result.departure_time, "morning")
        self.assertEqual(result.return_time, "evening")
        self.assertTrue({"departure_time", "return_time"} <= result.provided)

    def test_conflicting_time_constraints_are_not_actionable(self):
        result = parse_request("Find IAD to LAS departing morning then departing night")
        self.assertIn("conflicting_departure_time", result.conflicts)
        self.assertFalse(result.can_execute)

    def test_opposing_ordered_routes_conflict_but_repeated_route_does_not(self):
        opposing = parse_request("Find from IAD to LAS and from LAS to IAD")
        self.assertIn("conflicting_routes", opposing.conflicts)
        repeated = parse_request("Find from IAD to LAS and from IAD to LAS")
        self.assertNotIn("conflicting_routes", repeated.conflicts)

    def test_booking_authorization_is_independent_of_criteria_clause_order(self):
        parsed = parse_request("From IAD to LAS under $200, book flight 1027 for Amy Gonzalez")
        self.assertTrue(parsed.booking_intent)
        self.assertEqual((parsed.origin, parsed.destination, parsed.max_price), ("IAD", "LAS", 200))

    def test_withdrawal_after_booking_clause_revokes_authorization(self):
        parsed = parse_request("Book flight 1027 for Amy Gonzalez from IAD to LAS, but do not proceed")
        self.assertFalse(parsed.booking_intent)

    def test_terminal_cancel_clauses_revoke_booking_but_travel_reference_does_not(self):
        for ending in (", cancel that request", ", actually cancel"):
            with self.subTest(ending=ending):
                self.assertFalse(parse_request(
                    "Book flight 1027 for Amy Gonzalez from IAD to LAS" + ending
                ).booking_intent)
        unrelated = parse_request(
            "Book flight 1027 for Amy Gonzalez from IAD to LAS, with cancellation coverage"
        )
        self.assertTrue(unrelated.booking_intent)


if __name__ == "__main__":
    unittest.main()

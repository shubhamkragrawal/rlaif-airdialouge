import time
import unittest
import json

from local_app.controller import ConversationController
from local_app.evaluation import load_scenarios, run_scenario
from local_app.knowledge import FlightKnowledgeBase
from local_app.limits import ExecutionLimits
from local_app.request_parsing import parse_request
from local_app.tools import SearchCriteria


class BatchThreeIntegrationTests(unittest.TestCase):
    def controller(self, **kwargs):
        return ConversationController(FlightKnowledgeBase(scenario_id=0), **kwargs)

    def test_combined_route_constraints_passenger_selection_and_booking(self):
        controller = self.controller()
        state = controller.new_session()
        result = controller.handle(
            state,
            "Book flight 1027 from IAD to LAS departing 1/1 returning 1/3, "
            "at most 1 connection, under $500, passenger Amy Gonzalez",
        )
        self.assertEqual(state.selected_flight_id, 1027)
        self.assertEqual(state.booking["passenger"], "Amy Gonzalez")
        self.assertIn("simulated booking is confirmed", result["text"])

    def test_route_and_passenger_search_then_selection_clarification(self):
        controller = self.controller()
        state = controller.new_session()
        result = controller.handle(state, "Find IAD to LAS for Amy Gonzalez")
        self.assertEqual(state.passenger, "Amy Gonzalez")
        self.assertIsNotNone(state.last_search)
        self.assertIsNone(state.selected_flight_id)
        self.assertIn("Which flight", result["text"])

    def test_booking_only_uses_existing_search_without_searching_again(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        controller.handle(state, "Select flight 1027")
        calls = []
        original = controller.simulator.search_flights
        controller.simulator.search_flights = lambda criteria: (calls.append(criteria), original(criteria))[1]
        result = controller.handle(state, "Book flight 1027")
        self.assertTrue(state.booking)
        self.assertEqual(calls, [])
        self.assertIn("simulated booking is confirmed", result["text"])

    def test_conflicting_selections_and_passenger_names_do_not_mutate(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        before = (state.criteria, state.last_search, state.passenger,
                  state.selected_flight_id, state.hold_id, state.booking)
        for message, expected in (
            ("Book flight 1027 and flight 2044 for Amy Gonzalez", "conflicting_flight_selections"),
            ("Book flight 1027 for Amy Gonzalez and passenger Bob Jones", "conflicting_passenger_names"),
        ):
            parsed = parse_request(message)
            self.assertIn(expected, parsed.conflicts)
            result = controller.handle(state, message)
            self.assertEqual(result["fallback_reason"], "request_conflict")
            self.assertEqual((state.criteria, state.last_search, state.passenger,
                              state.selected_flight_id, state.hold_id, state.booking), before)

    def test_negated_booking_does_not_hold_or_book(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        result = controller.handle(state, "Do not book flight 1027")
        self.assertFalse(controller.simulator.holds)
        self.assertFalse(controller.simulator.bookings)
        self.assertIsNone(state.booking)
        self.assertIsNone(state.selected_flight_id)
        self.assertNotIn("simulated booking is confirmed", result["text"])

    def test_material_criteria_change_invalidates_active_choice_and_booking_archives(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        controller.handle(state, "Book flight 1027")
        active_booking = state.booking
        controller.handle(state, "Find IAD to LAS under $200")
        self.assertNotEqual(state.criteria.max_price, None)
        self.assertIsNone(state.selected_flight_id)
        self.assertIsNone(state.hold_id)
        self.assertIsNone(state.booking)
        self.assertEqual(len(state.booking_history), 1)
        self.assertEqual(state.booking_history[0].booking_id, active_booking["booking_id"])
        self.assertEqual(state.booking_history[0].passenger, "Amy Gonzalez")

    def test_failed_and_timed_out_searches_roll_back_but_empty_search_commits(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        controller.handle(state, "Select flight 1027")
        before = (state.criteria, state.last_search, state.selected_flight_id,
                  state.passenger, state.hold_id, state.booking, state.booking_history)
        original = controller.simulator.search_flights
        controller.simulator.search_flights = lambda _criteria: (_ for _ in ()).throw(RuntimeError("down"))
        failed = controller.handle(state, "Find BOS to LAS")
        self.assertEqual(failed["fallback_reason"], "tool_error:RuntimeError")
        self.assertEqual((state.criteria, state.last_search, state.selected_flight_id,
                          state.passenger, state.hold_id, state.booking, state.booking_history), before)
        controller.simulator.search_flights = original
        empty = controller.handle(state, "Find IAD to LAS under $1")
        self.assertIn("matching flight", empty["text"])
        self.assertEqual(empty["results"], [])
        self.assertEqual(state.criteria.max_price, 1)
        self.assertEqual(state.last_search.criteria, state.criteria)
        self.assertEqual(state.last_search.flights, ())
        self.assertIsNone(state.selected_flight_id)
        self.assertEqual(state.passenger, before[3])
        blocked = controller.handle(state, "Book flight 1027")
        self.assertEqual(blocked["fallback_reason"], "invalid_flight_selection")
        self.assertIsNone(state.booking)
        self.assertFalse(controller.simulator.bookings)

        slow_controller = self.controller(limits=ExecutionLimits(max_session_seconds=0.02))
        slow_state = slow_controller.new_session()
        slow_state.criteria = SearchCriteria(origin="IAD", destination="LAS")
        slow_state.last_search = slow_controller.simulator.search_flights(slow_state.criteria)
        slow_state.passenger = "Amy Gonzalez"
        slow_state.selected_flight_id = 1027
        timed_before = (slow_state.criteria, slow_state.last_search, slow_state.selected_flight_id)
        original_slow = slow_controller.simulator.search_flights
        def slow_search(criteria):
            time.sleep(0.05)
            return original_slow(criteria)
        slow_controller.simulator.search_flights = slow_search
        timed = slow_controller.handle(slow_state, "Find BOS to LAS")
        self.assertEqual(timed["fallback_reason"], "session_time_exceeded")
        self.assertEqual((slow_state.criteria, slow_state.last_search, slow_state.selected_flight_id), timed_before)

    def test_stale_selection_cannot_be_used_after_material_search(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        controller.handle(state, "Select flight 1027")
        result = controller.handle(state, "Book flight 1027 from BOS to LAS")
        self.assertEqual(result["fallback_reason"], "invalid_flight_selection")
        self.assertIsNone(state.selected_flight_id)
        self.assertIsNone(state.booking)
        self.assertFalse(controller.simulator.bookings)

    def test_passenger_change_archives_confirmed_evidence_immutably(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        controller.handle(state, "Book flight 1027")
        prior = state.booking
        controller.handle(state, "My name is Jordan Lee")
        self.assertEqual(state.passenger, "Jordan Lee")
        self.assertIsNone(state.booking)
        self.assertEqual(len(state.booking_history), 1)
        record = state.booking_history[0]
        self.assertEqual(record.passenger, "Amy Gonzalez")
        self.assertEqual(record.booking_id, prior["booking_id"])
        with self.assertRaises((AttributeError, TypeError)):
            record.passenger = "Jordan Lee"

    def test_empty_search_archives_booking_and_blocks_the_previous_flight(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        controller.handle(state, "Book flight 1027")
        prior = state.booking
        empty = controller.handle(state, "Find IAD to LAS under $1")
        self.assertEqual(empty["results"], [])
        self.assertEqual(state.criteria.max_price, 1)
        self.assertEqual(state.last_search.flights, ())
        self.assertIsNone(state.selected_flight_id)
        self.assertIsNone(state.hold_id)
        self.assertIsNone(state.booking)
        self.assertEqual(len(state.booking_history), 1)
        self.assertEqual(state.booking_history[0].booking_id, prior["booking_id"])
        blocked = controller.handle(state, "Book flight 1027")
        self.assertEqual(blocked["fallback_reason"], "invalid_flight_selection")
        self.assertEqual(controller.simulator.bookings[prior["booking_id"]], prior)
        self.assertIsNone(state.booking)

    def test_session_api_serializes_immutable_booking_history(self):
        from local_app.api import _session_payload

        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        controller.handle(state, "Book flight 1027")
        controller.handle(state, "Find IAD to LAS under $200")
        payload = _session_payload("history-test", state)
        encoded = json.dumps(payload, ensure_ascii=False)
        restored = json.loads(encoded)
        self.assertEqual(restored["booking_history"][0]["passenger"], "Amy Gonzalez")
        self.assertEqual(restored["booking_history"][0]["booking_id"],
                         state.booking_history[0].booking_id)

    def test_rejected_passenger_preserves_confirmed_session_and_repeat_is_idempotent(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "My name is Amy Gonzalez")
        first = controller.handle(state, "Book flight 1027")
        booking = state.booking
        hold_id = state.hold_id
        history = state.booking_history
        rejected = controller.handle(state, "Passenger 123")
        self.assertTrue(rejected["fallback_reason"].startswith("passenger_"))
        self.assertEqual(state.passenger, "Amy Gonzalez")
        self.assertEqual(state.hold_id, hold_id)
        self.assertIs(state.booking, booking)
        self.assertEqual(state.booking_history, history)
        repeated = controller.handle(state, "Book flight 1027")
        self.assertIs(state.booking, booking)
        self.assertEqual(repeated["text"], first["text"])
        self.assertEqual(len(controller.simulator.bookings), 1)

    def test_later_do_not_proceed_cancels_combined_booking_authority(self):
        controller = self.controller()
        state = controller.new_session()
        result = controller.handle(
            state, "Book flight 1027 for Amy Gonzalez from IAD to LAS, but do not proceed"
        )
        self.assertFalse(controller.simulator.holds)
        self.assertFalse(controller.simulator.bookings)
        self.assertIsNone(state.hold_id)
        self.assertIsNone(state.booking)
        self.assertNotIn("simulated booking is confirmed", result["text"])
        self.assertRegex(result["text"].lower(), r"(not|won't|will not|cannot).{0,30}(book|proceed|authorize)|cancel")

    def test_terminal_cancel_variants_leave_fresh_booking_state_untouched(self):
        for ending in (", cancel that request", ", actually cancel"):
            with self.subTest(ending=ending):
                controller = self.controller()
                state = controller.new_session()
                before = (state.criteria, state.last_search, state.selected_flight_id,
                          state.passenger, state.hold_id, state.booking, state.booking_history)
                searches = []
                original = controller.simulator.search_flights
                controller.simulator.search_flights = lambda criteria: (
                    searches.append(criteria), original(criteria)
                )[1]
                result = controller.handle(
                    state, "Book flight 1027 for Amy Gonzalez from IAD to LAS" + ending
                )
                self.assertFalse(searches)
                self.assertFalse(controller.simulator.holds)
                self.assertFalse(controller.simulator.bookings)
                self.assertNotIn("simulated booking is confirmed", result["text"])
                self.assertEqual((state.criteria, state.last_search, state.selected_flight_id,
                                  state.passenger, state.hold_id, state.booking,
                                  state.booking_history), before)

    def test_cancellation_coverage_is_not_a_withdrawal(self):
        controller = self.controller()
        state = controller.new_session()
        result = controller.handle(
            state, "Book flight 1027 for Amy Gonzalez from IAD to LAS, with cancellation coverage"
        )
        self.assertIn("simulated booking is confirmed", result["text"])
        self.assertEqual(state.booking["passenger"], "Amy Gonzalez")

    def test_opposing_routes_rejected_before_search_or_state_change(self):
        controller = self.controller()
        state = controller.new_session()
        before = (state.criteria, state.last_search, state.selected_flight_id,
                  state.passenger, state.hold_id, state.booking)
        calls = []
        original = controller.simulator.search_flights
        controller.simulator.search_flights = lambda criteria: (calls.append(criteria), original(criteria))[1]
        result = controller.handle(state, "Find from IAD to LAS and from LAS to IAD")
        self.assertEqual(result["fallback_reason"], "request_conflict")
        self.assertEqual(calls, [])
        self.assertEqual((state.criteria, state.last_search, state.selected_flight_id,
                          state.passenger, state.hold_id, state.booking), before)

    def test_criteria_before_booking_command_executes_combined_turn(self):
        controller = self.controller()
        state = controller.new_session()
        result = controller.handle(state, "From IAD to LAS under $200, book flight 1027 for Amy Gonzalez")
        self.assertEqual(state.selected_flight_id, 1027)
        self.assertEqual(state.booking["passenger"], "Amy Gonzalez")
        self.assertIn("simulated booking is confirmed", result["text"])

    def test_explicit_origin_without_destination_clarifies_without_mutation(self):
        controller = self.controller()
        state = controller.new_session()
        controller.handle(state, "Find IAD to LAS")
        controller.handle(state, "Select flight 1027")
        before = (state.criteria, state.last_search, state.selected_flight_id,
                  state.passenger, state.hold_id, state.booking)
        calls = []
        original = controller.simulator.search_flights
        controller.simulator.search_flights = lambda criteria: (calls.append(criteria), original(criteria))[1]
        result = controller.handle(state, "Find from BOS")
        self.assertIn("destination", result["text"].lower())
        self.assertEqual(calls, [])
        self.assertEqual((state.criteria, state.last_search, state.selected_flight_id,
                          state.passenger, state.hold_id, state.booking), before)

    def test_fresh_booking_asks_for_route_before_selection_membership(self):
        controller = self.controller()
        state = controller.new_session()
        result = controller.handle(state, "Book flight 1027 for Amy Gonzalez")
        self.assertIn("departure and destination", result["text"].lower())
        self.assertNotEqual(result["fallback_reason"], "invalid_flight_selection")
        self.assertIsNone(state.last_search)
        self.assertFalse(controller.simulator.holds)
        self.assertFalse(controller.simulator.bookings)

    def test_all_fields_passenger_extraction_and_booking(self):
        controller = self.controller()
        state = controller.new_session()
        message = ("Book flight 1027 for Amy Gonzalez from IAD to LAS departing 1/1 "
                   "returning 1/3 at most 1 connection under $500")
        parsed = parse_request(message)
        self.assertEqual(parsed.passenger.name, "Amy Gonzalez")
        self.assertNotIn("passenger_unusable", parsed.ambiguities)
        result = controller.handle(state, message)
        self.assertEqual(state.booking["passenger"], "Amy Gonzalez")
        self.assertIn("simulated booking is confirmed", result["text"])

    def test_no_flight_matrix_criteria_match_legacy_and_commit_empty_search(self):
        scenario = next(item for item in load_scenarios(count=5, start=0)
                        if item.scenario_id == 1)
        message = scenario.messages[0]
        controller = ConversationController(
            FlightKnowledgeBase(scenario_id=scenario.scenario_id)
        )
        parsed = parse_request(message)
        legacy = controller._criteria_from_text(message)
        parsed_criteria = SearchCriteria(
            origin=parsed.origin, destination=parsed.destination,
            departure_month=parsed.departure_month, departure_day=parsed.departure_day,
            return_month=parsed.return_month, return_day=parsed.return_day,
            departure_time=parsed.departure_time, return_time=parsed.return_time,
            max_price=parsed.max_price, max_connections=parsed.max_connections,
        )
        self.assertEqual(parsed_criteria, legacy)
        state = controller.new_session()
        result = controller.handle(state, message)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["results"], [])
        self.assertIsNotNone(state.last_search)
        self.assertEqual(state.last_search.criteria, legacy)
        self.assertEqual(state.last_search.flights, ())

    def test_same_airport_booking_matrix_cases_remain_search_and_selection_validated(self):
        scenarios = {item.scenario_id: item for item in load_scenarios(count=50, start=0)}
        for scenario_id in (44, 53):
            with self.subTest(scenario_id=scenario_id):
                scenario = scenarios[scenario_id]
                parsed = parse_request(scenario.messages[0])
                self.assertEqual(parsed.origin, parsed.destination)
                self.assertFalse(parsed.conflicts)
                result = run_scenario(scenario)
                self.assertTrue(result["task_success"])
                self.assertEqual(result["rejections"], 0)
                self.assertEqual(result["booked_flight_id"], scenario.expected_flight_ids[0])


if __name__ == "__main__":
    unittest.main()

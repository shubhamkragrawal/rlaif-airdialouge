import unittest

from local_app.controller import ConversationController
from local_app.knowledge import FlightKnowledgeBase


class Adapter:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, messages, retrieved_flights):
        self.calls += 1
        return self.response


class BatchOneIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.controller = ConversationController(FlightKnowledgeBase(scenario_id=0))
        self.state = self.controller.new_session("sft")

    def search_select(self):
        self.controller.handle(self.state, "Find a flight from IAD to LAS under $100")
        self.controller.handle(self.state, "My name is Amy Gonzalez")

    def test_unsupported_confirmation_adapter_is_rejected_with_reason(self):
        adapter = Adapter("Your booking is confirmed and simulated.")
        c = ConversationController(FlightKnowledgeBase(scenario_id=0), model_adapters={"sft": adapter})
        result = c.handle(c.new_session(), "IAD to LAS")
        self.assertFalse(result["model_used"])
        self.assertEqual(result["fallback_reason"], "unsupported_transaction_claim")

    def test_invalid_explicit_id_after_valid_selection_preserves_options(self):
        self.search_select()
        self.controller.handle(self.state, "Select flight 1027")
        result = self.controller.handle(self.state, "Book flight 9999")
        self.assertIn("not among the current simulator options", result["text"])
        self.assertEqual([f["flight_number"] for f in result["results"]],
                         [f["flight_number"] for f in self.state.last_search.flights[:5]])
        self.assertIsNone(self.state.booking)

    def test_route_change_and_book_checks_latest_search(self):
        self.search_select()
        result = self.controller.handle(self.state, "Change route to BOS to LAS and book flight 1027")
        self.assertIn("not among the current simulator options", result["text"])
        self.assertIsNone(self.state.booking)

    def test_empty_results_are_controller_authoritative(self):
        adapter = Adapter("I found another great option!")
        c = ConversationController(FlightKnowledgeBase(scenario_id=0), model_adapters={"sft": adapter})
        state = c.new_session()
        result = c.handle(state, "Find a flight from IAD to LAS under $1")
        self.assertEqual(result["text"], "I couldn't find a matching flight in the local simulator.")
        self.assertEqual(adapter.calls, 0)

    def test_simulated_confirmation_is_authoritative_and_idempotent(self):
        self.search_select()
        first = self.controller.handle(self.state, "Book flight 1027")
        booking = dict(self.state.booking)
        original_book = self.controller.simulator.book_flight
        calls = []
        def tracked(hold_id):
            calls.append(hold_id)
            return original_book(hold_id)
        self.controller.simulator.book_flight = tracked
        second = self.controller.handle(self.state, "Book flight 1027")
        self.assertEqual(len(calls), 1)
        self.assertIn("simulated booking is confirmed", first["text"])
        self.assertEqual(self.state.booking, booking)
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(first["fallback_reason"], None)

    def test_evidence_mismatches_reject_transaction_claim(self):
        from local_app.guardrails import TransactionClaimError, validate_transaction_claim
        hold = self.controller.simulator.hold_flight(1027, "Amy Gonzalez")
        booking = self.controller.simulator.book_flight(hold["hold_id"])
        evidence = {**booking, "flight_id": 1027, "hold": hold}
        for key, value in (("hold_id", "other"), ("flight_id", 9999), ("passenger", "Other Name"),
                           ("status", "pending"), ("simulated", False)):
            bad = dict(evidence, **{key: value})
            with self.subTest(key=key), self.assertRaises(TransactionClaimError):
                validate_transaction_claim("Your simulated booking is confirmed.", bad)

    def test_affirmative_claim_paraphrases_rejected_but_negations_and_questions_are_not_claims(self):
        from local_app.guardrails import detect_transaction_claim
        positives = ("Booked!", "Your seat is locked in.",
                     "Your booking has been successfully confirmed.",
                     "I have arranged your trip.", "I've arranged your flight.",
                     "Your trip is arranged.", "Your booking is confirmed! Anything else?",
                     "I have not charged you; I booked your flight.",
                     "You’re all set for your trip.", "I have arranged your trip.",
                     "Your e-ticket has been generated.", "Your reservation was made.",
                     "I completed your booking.",
                     "The booking is now complete.", "The booking was successful.",
                     "The reservation has already been made.",
                     "Not charged because booking is confirmed.",
                     "I have not charged you and your flight is booked.",
                     "Will do, your flight is confirmed.",
                     "Your booking is confirmed! Anything else?",
                     "Can I help you now that your booking is confirmed?",
                     "Would you like your itinerary now that I have booked your flight?",
                     "Would you like the flight I booked for you?",
                     "Would you like the ticket I issued for you?")
        for phrase in positives:
            with self.subTest(phrase=phrase):
                self.assertTrue(detect_transaction_claim(phrase))
        for phrase in ("Your booking is not confirmed.", "Is your booking confirmed?",
                       "Have I booked your flight?",
                       "Would you like me to book this flight?", "I haven't arranged your trip.",
                       "Have I arranged your trip?", "This option can be reserved after you choose it."):
            with self.subTest(phrase=phrase):
                self.assertFalse(detect_transaction_claim(phrase))

    def test_confirmation_claims_are_rejected_for_every_adapter(self):
        phrases = ("Booked!", "Your seat is locked in.",
                   "Your booking has been successfully confirmed.", "I have arranged your trip.",
                   "I've arranged your flight.", "The booking is now complete.",
                   "The booking was successful.", "The reservation has already been made.",
                   "Not charged because booking is confirmed.",
                   "Can I help you now that your booking is confirmed?",
                   "Would you like your itinerary now that I have booked your flight?",
                   "Would you like the flight I booked for you?",
                   "Would you like the ticket I issued for you?")
        for adapter_name in ("sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50"):
            for phrase in phrases:
                with self.subTest(adapter=adapter_name, phrase=phrase):
                    adapter = Adapter(phrase)
                    controller = ConversationController(FlightKnowledgeBase(scenario_id=0),
                                                        model_adapters={adapter_name: adapter})
                    result = controller.handle(controller.new_session(adapter_name), "IAD to LAS")
                    self.assertFalse(result["model_used"])
                    self.assertEqual(result["fallback_reason"], "unsupported_transaction_claim")

    def test_flight_ids_are_scanned_before_pronoun_fallback(self):
        for command in ("Book my flight 9999", "Book it, flight 9999",
                        "Confirm my reservation for flight 9999"):
            with self.subTest(command=command):
                self.search_select()
                self.controller.handle(self.state, "Select flight 1027")
                result = self.controller.handle(self.state, command)
                self.assertEqual(result["fallback_reason"], "invalid_flight_selection")
                self.assertIsNone(self.state.selected_flight_id)
                self.assertIsNone(self.state.booking)

    def test_unicode_apostrophe_withdrawal_overrides_prior_booking_command(self):
        self.search_select()
        for message in ("Book flight 1027? No, don’t book it.",
                        "Book flight 1027 — no, don’t reserve it.",
                        "Book flight 1027? No, do not actually book it.",
                        "Book flight 1027? Don’t ever confirm it.",
                        "Go ahead with flight 1027? No, do not immediately actually book it.",
                        "Go ahead with flight 1027? No."):
            with self.subTest(message=message):
                result = self.controller.handle(self.state, message)
                self.assertIsNone(self.state.booking)
                self.assertEqual(self.controller.simulator.bookings, {})
                self.assertNotIn("simulated booking is confirmed", result["text"])

    def test_booking_tool_schema_failure_does_not_commit_session(self):
        self.search_select()
        self.controller.simulator.book_flight = lambda _hold_id: {
            "booking_id": "bad", "hold_id": self.state.hold_id, "flight": {},
            "passenger": "Amy Gonzalez", "simulated": False, "status": "booked",
        }
        result = self.controller.handle(self.state, "Book flight 1027")
        self.assertIsNone(self.state.booking)
        self.assertIsNone(self.state.hold_id)
        self.assertNotIn("confirmed", result["text"].lower())

    def test_mismatched_booking_evidence_never_commits(self):
        from copy import deepcopy
        mutations = {
            "status": lambda value: value.update(status="booked"),
            "simulated": lambda value: value.update(simulated=False),
            "passenger": lambda value: value.update(passenger="Other Person"),
            "hold": lambda value: value.update(hold_id="other-hold"),
            "flight": lambda value: value.update(flight={"flight_number": 9999}),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                self.search_select()
                state = self.state
                original_book = self.controller.simulator.book_flight
                def malformed(hold_id, mutate=mutate, original_book=original_book):
                    result = deepcopy(original_book(hold_id))
                    mutate(result)
                    return result
                self.controller.simulator.book_flight = malformed
                result = self.controller.handle(state, "Book flight 1027")
                self.assertIsNone(state.booking)
                self.assertIsNone(state.hold_id)
                self.assertNotIn("confirmed", result["text"].lower())
                self.assertEqual(self.controller.simulator.holds, {})
                self.assertEqual(self.controller.simulator.bookings, {})
                self.controller.simulator.book_flight = original_book

    def test_invalid_turn_outcome_booking_evidence_is_rejected(self):
        from local_app.outcomes import Intent, make_turn_outcome
        hold = self.controller.simulator.hold_flight(1027, "Amy Gonzalez")
        malformed = {"booking_id": "b", "hold_id": hold["hold_id"], "flight": hold["flight"],
                     "passenger": hold["passenger"], "simulated": False, "status": "confirmed"}
        with self.assertRaises(ValueError):
            make_turn_outcome(intent=Intent.BOOK, deterministic_response="confirmed", hold_result=hold,
                              booking_result=malformed)

    def test_valid_but_wrong_tool_flight_is_rejected_before_booking(self):
        self.search_select()
        original_hold = self.controller.simulator.hold_flight
        original_book = self.controller.simulator.book_flight
        self.controller.simulator.hold_flight = lambda _flight_id, passenger: original_hold(1012, passenger)
        book_calls = []
        self.controller.simulator.book_flight = lambda hold_id: (book_calls.append(hold_id), original_book(hold_id))[1]
        result = self.controller.handle(self.state, "Book flight 1027")
        self.assertIsNone(self.state.booking)
        self.assertIsNone(self.state.hold_id)
        self.assertEqual(book_calls, [])
        self.assertNotIn("confirmed", result["text"].lower())
        self.assertEqual(self.controller.simulator.holds, {})
        self.assertEqual(self.controller.simulator.bookings, {})
        self.controller.simulator.hold_flight = original_hold
        self.controller.simulator.book_flight = original_book

    def test_do_not_book_and_availability_question_do_not_execute_booking(self):
        self.search_select()
        for command in ("Do not book flight 1027.", "Is flight 1027 available to book?"):
            with self.subTest(command=command):
                result = self.controller.handle(self.state, command)
                self.assertIsNone(self.state.booking)
                self.assertEqual(self.controller.simulator.bookings, {})
                self.assertNotIn("confirmed", result["text"].lower())

    def test_malformed_and_signed_booking_ids_never_reuse_selection(self):
        self.search_select()
        self.controller.handle(self.state, "Select flight 1027")
        for command in ("Book flight 999999", "Select flight 99", "Book flight -1027",
                        "Book flight 9999X", "Select flight 1027.5", "-9999",
                        "Flight 9999"):
            with self.subTest(command=command):
                result = self.controller.handle(self.state, command)
                self.assertEqual(result["fallback_reason"], "invalid_flight_selection")
                self.assertIsNone(self.state.selected_flight_id)
                self.assertIsNone(self.state.booking)

    def test_completed_claim_negation_and_question_are_clause_local(self):
        from local_app.guardrails import detect_transaction_claim
        for phrase in ("Your booking is confirmed! Anything else?",
                       "I have not charged you and flight is booked.",
                       "Will do, your flight is confirmed.",
                       "Your e-ticket was generated.", "Your reservation was made.",
                       "I completed your booking."):
            with self.subTest(phrase=phrase):
                self.assertTrue(detect_transaction_claim(phrase))
        for phrase in ("This option can be reserved later, and your flight is confirmed.",
                       "Your flight may be booked once selected; your reservation was made."):
            with self.subTest(phrase=phrase):
                self.assertTrue(detect_transaction_claim(phrase))
        for phrase in ("Your booking is not confirmed.",
                       "Is your booking confirmed?",
                       "This option can be reserved after you choose it.",
                       "Would you like me to book it?"):
            with self.subTest(phrase=phrase):
                self.assertFalse(detect_transaction_claim(phrase))

    def test_proceed_continue_and_go_ahead_selection_forms_validate_full_ids(self):
        self.search_select()
        for phrase in ("Proceed with flight 1027", "Continue with flight 1027",
                       "Go ahead with flight 1027"):
            with self.subTest(phrase=phrase):
                if self.state.booking is not None:
                    self.setUp()
                    self.search_select()
                result = self.controller.handle(self.state, phrase)
                self.assertEqual(self.state.selected_flight_id, 1027)
                self.assertEqual(result["fallback_reason"], None)
        for phrase in ("Proceed with flight 9999X", "Continue with flight -1027",
                       "Go ahead with flight 1027.5"):
            with self.subTest(phrase=phrase):
                result = self.controller.handle(self.state, phrase)
                self.assertEqual(result["fallback_reason"], "invalid_flight_selection")
                self.assertIsNone(self.state.selected_flight_id)

    def test_pronoun_and_article_booking_references_reuse_current_context(self):
        for phrase in ("Please confirm the booking", "Book it", "reserve it", "confirm it"):
            with self.subTest(phrase=phrase):
                controller = ConversationController(FlightKnowledgeBase(scenario_id=0))
                state = controller.new_session()
                controller.handle(state, "Find a flight from IAD to LAS under $100")
                controller.handle(state, "My name is Amy Gonzalez")
                controller.handle(state, "Select flight 1027")
                result = controller.handle(state, phrase)
                self.assertIsNotNone(state.booking)
                self.assertIn("simulated booking is confirmed", result["text"])
        controller = ConversationController(FlightKnowledgeBase(scenario_id=0))
        result = controller.handle(controller.new_session(), "Please confirm the booking")
        self.assertEqual(result["fallback_reason"], "selection_not_in_latest_search")

    def test_later_explicit_withdrawal_cancels_booking_command(self):
        self.search_select()
        result = self.controller.handle(self.state, "Book flight 1027? No, do not book it.")
        self.assertIsNone(self.state.booking)
        self.assertEqual(self.controller.simulator.bookings, {})
        self.assertNotIn("simulated booking is confirmed", result["text"])

    def test_post_book_store_mutations_fail_closed_and_rollback(self):
        for mutate in ("hold", "booking"):
            with self.subTest(mutate=mutate):
                controller = ConversationController(FlightKnowledgeBase(scenario_id=0))
                state = controller.new_session()
                controller.handle(state, "Find a flight from IAD to LAS under $100")
                controller.handle(state, "My name is Amy Gonzalez")
                original = controller.simulator.book_flight
                def sabotaged(hold_id, mutate=mutate, original=original):
                    returned = original(hold_id)
                    if mutate == "hold":
                        controller.simulator.holds[hold_id]["status"] = "cancelled"
                    else:
                        controller.simulator.bookings.pop(returned["booking_id"])
                    return returned
                controller.simulator.book_flight = sabotaged
                result = controller.handle(state, "Book flight 1027")
                self.assertIsNone(state.booking)
                self.assertIsNone(state.hold_id)
                self.assertNotIn("confirmed", result["text"].lower())
                self.assertEqual(controller.simulator.bookings, {})
                self.assertEqual(controller.simulator.holds, {})

    def test_repeat_booking_rechecks_stores_after_simulator_call(self):
        self.search_select()
        self.controller.handle(self.state, "Book flight 1027")
        original = self.controller.simulator.book_flight
        def sabotage(hold_id):
            returned = original(hold_id)
            self.controller.simulator.holds.pop(hold_id)
            return returned
        self.controller.simulator.book_flight = sabotage
        result = self.controller.handle(self.state, "Book it")
        self.assertEqual(result["fallback_reason"], "stale_booking_evidence")
        self.assertIsNone(self.state.booking)
        self.assertNotIn("confirmed", result["text"].lower())

    def test_schema_valid_fabricated_booking_not_in_storage_is_rolled_back(self):
        self.search_select()
        original_book = self.controller.simulator.book_flight
        def fabricate(hold_id):
            actual = original_book(hold_id)
            return dict(actual, booking_id="booking_fabricated")
        self.controller.simulator.book_flight = fabricate
        result = self.controller.handle(self.state, "Book flight 1027")
        self.assertEqual(result["fallback_reason"], "tool_error:ValueError")
        self.assertIsNone(self.state.booking)
        self.assertIsNone(self.state.hold_id)
        self.assertEqual(self.controller.simulator.bookings, {})
        self.assertEqual(self.controller.simulator.holds, {})

    def test_search_transition_is_atomic_for_malformed_results(self):
        for malformed in ({}, None, []):
            with self.subTest(malformed=malformed):
                self.search_select()
                state = self.state
                before = (state.criteria, state.last_search, state.selected_flight_id,
                          state.hold_id, state.booking)
                original = self.controller.simulator.search_flights
                self.controller.simulator.search_flights = lambda _criteria, value=malformed: value
                result = self.controller.handle(state, "Find a flight from BOS to LAS")
                self.assertTrue(result["fallback_reason"].startswith("tool_error:"))
                self.assertEqual((state.criteria, state.last_search, state.selected_flight_id,
                                  state.hold_id, state.booking), before)
                self.controller.simulator.search_flights = original

    def test_new_search_with_missing_explicit_selection_reports_invalid_selection(self):
        self.search_select()
        result = self.controller.handle(self.state,
            "Find a flight from BOS to LAS and select flight 1027")
        self.assertEqual(result["fallback_reason"], "invalid_flight_selection")
        self.assertIn("not among the current simulator options", result["text"])

    def test_model_loader_propagates_unsupported_claim_reason(self):
        adapter = Adapter("Your booking has been confirmed.")
        controller = ConversationController(FlightKnowledgeBase(scenario_id=0),
                                            model_loader=lambda _name: adapter)
        result = controller.handle(controller.new_session(), "Find a flight from IAD to LAS")
        self.assertEqual(result["fallback_reason"], "unsupported_transaction_claim")

    def test_malformed_search_tool_values_fail_closed_without_state_commit(self):
        for bad_result in ({}, None, []):
            with self.subTest(bad_result=bad_result):
                controller = ConversationController(FlightKnowledgeBase(scenario_id=0))
                state = controller.new_session()
                original = controller.simulator.search_flights
                controller.simulator.search_flights = lambda _criteria, value=bad_result: value
                result = controller.handle(state, "Find a flight from IAD to LAS")
                self.assertIsNone(state.last_search)
                self.assertIsNone(state.selected_flight_id)
                self.assertEqual(result["fallback_reason"], "tool_error:ValueError")
                controller.simulator.search_flights = original

    def test_malformed_hold_tool_values_are_cleaned_without_dereference(self):
        for bad_hold in ({}, None, []):
            with self.subTest(bad_hold=bad_hold):
                controller = ConversationController(FlightKnowledgeBase(scenario_id=0))
                state = controller.new_session()
                controller.handle(state, "Find a flight from IAD to LAS")
                controller.handle(state, "My name is Amy Gonzalez")
                controller.handle(state, "Select flight 1027")
                controller.simulator.hold_flight = lambda _flight, _passenger, value=bad_hold: value
                result = controller.handle(state, "Book flight 1027")
                self.assertIsNone(state.booking)
                self.assertIsNone(state.hold_id)
                self.assertNotIn("confirmed", result["text"].lower())

    def test_missing_booking_prerequisite_turns_never_call_model(self):
        adapter = Adapter("Booked!")
        controller = ConversationController(FlightKnowledgeBase(scenario_id=0),
                                            model_adapters={"sft": adapter})
        state = controller.new_session()
        result = controller.handle(state, "Please book a flight")
        self.assertEqual(adapter.calls, 0)
        self.assertNotIn("booked", result["text"].lower())

    def test_bare_invalid_numeric_selection_clears_prior_selection(self):
        self.search_select()
        self.controller.handle(self.state, "Select flight 1027")
        result = self.controller.handle(self.state, "9999")
        self.assertIn("not among the current simulator options", result["text"])
        self.assertIsNone(self.state.selected_flight_id)

    def test_stale_cached_booking_does_not_confirm(self):
        self.search_select()
        self.controller.handle(self.state, "Book flight 1027")
        booking_id = self.state.booking["booking_id"]
        del self.controller.simulator.bookings[booking_id]
        result = self.controller.handle(self.state, "Book flight 1027")
        self.assertIn("couldn't verify", result["text"])
        self.assertEqual(result["fallback_reason"], "stale_booking_evidence")

    def test_raw_state_and_empty_array_model_outputs_are_rejected(self):
        for model_text in ("[]", '{"selected_flight_id": 1027}', "SessionState(booking=None)", "state: booking=None"):
            with self.subTest(model_text=model_text):
                adapter = Adapter(model_text)
                controller = ConversationController(FlightKnowledgeBase(scenario_id=0),
                                                    model_adapters={"sft": adapter})
                result = controller.handle(controller.new_session(),
                                           "Find a flight from IAD to LAS under $100")
                self.assertFalse(result["model_used"])
                self.assertIn("1027", result["text"])

    def test_model_output_validator_takes_outcome_and_checks_claim_against_it(self):
        from local_app.outcomes import Intent, make_turn_outcome
        from local_app.tools import SearchCriteria
        result = self.controller.simulator.search_flights(SearchCriteria(origin="IAD", destination="LAS"))
        outcome = make_turn_outcome(intent=Intent.SEARCH,
                                    deterministic_response="I found simulated flights.",
                                    search_results=result)
        with self.assertRaisesRegex(ValueError, "unsupported_transaction_claim"):
            self.controller._validate_model_output("I have arranged your trip.", outcome)

    def test_secondary_search_error_fails_closed_without_state_change(self):
        state = self.controller.new_session()
        state.criteria = self.controller._criteria_from_text("IAD to LAS")
        before = (state.criteria, state.last_search, state.selected_flight_id, state.hold_id, state.booking)
        original = self.controller.simulator.search_flights
        self.controller.simulator.search_flights = lambda _criteria: (_ for _ in ()).throw(RuntimeError("search down"))
        result = self.controller.handle(state, "Show options")
        self.assertEqual(result["fallback_reason"], "tool_error:RuntimeError")
        self.assertEqual((state.criteria, state.last_search, state.selected_flight_id, state.hold_id, state.booking), before)
        self.controller.simulator.search_flights = original


if __name__ == "__main__":
    unittest.main()

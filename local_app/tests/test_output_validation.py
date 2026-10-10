import unittest
from local_app.output_validation import validate_output, validate_or_raise, OutputValidationException
from local_app.outcomes import Intent, make_turn_outcome
from local_app.tools import FlightSearchResult, SearchCriteria


def outcome(flights=(), *, booking=None, hold=None):
    results = FlightSearchResult(tuple(flights), SearchCriteria()) if flights is not None else None
    intent = Intent.BOOK if booking else Intent.HOLD if hold else Intent.SEARCH
    return make_turn_outcome(intent=intent, deterministic_response="", search_results=results,
                             booking_result=booking, hold_result=hold)


def codes(text, result):
    return {issue.code for issue in validate_output(text, result)}


class OutputValidationTests(unittest.TestCase):
    def setUp(self):
        self.flights = (
            {"flight_number": 1027, "price": 240, "airline": "Delta Airlines"},
            {"flight_number": 1031, "price": 310, "airline": "United Airlines"},
        )
        self.result = outcome(self.flights)

    def test_malformed_empty_json_fenced_json_state_control_and_limit(self):
        self.assertIn("malformed_output", codes(None, self.result))
        self.assertIn("empty_output", codes("  ", self.result))
        self.assertIn("raw_json", codes('{"flight": 1027}', self.result))
        self.assertIn("code_fenced_json", codes('```json\n{"flight": 1027}\n```', self.result))
        self.assertIn("internal_state", codes("selected_flight_id: 1027", self.result))
        self.assertIn("control_token", codes("Hello <|assistant|>", self.result))
        self.assertIn("output_too_long", codes("x" * 4001, self.result))

    def test_grounding_covers_ids_prices_airlines_and_availability(self):
        valid = "Delta Airlines flight 1027 costs $240; United Airlines flight 1031 costs $310."
        self.assertEqual(validate_output(valid, self.result), ())
        issues = codes("Flight 9999 on Alaska Airlines costs $999. No flights are available.", self.result)
        self.assertTrue({"unsupported_flight_id", "unsupported_price", "unsupported_airline", "availability_contradiction"} <= issues)
        self.assertIn("availability_contradiction", codes("I found two flights.", outcome(())))

    def test_repetition_checks_ignore_recurring_flight_ids(self):
        natural = "Flight 1027 is $240 on Delta. Flight 1027 leaves this evening from BOS."
        self.assertFalse({"repeated_sentence", "repeated_ngram", "closing_loop"} & codes(natural, self.result))
        repeated = "Please review these available flight options carefully. Another detail follows. Please review these available flight options carefully."
        self.assertIn("repeated_sentence", codes(repeated, self.result))
        self.assertIn("closing_loop", codes(repeated, self.result))
        self.assertIn("duplicate_lines", codes("This is a repeated line.\nThis is a repeated line.", self.result))
        loop = "Here are your flight options today. A second useful sentence appears. Here are your flight options today."
        self.assertIn("closing_loop", codes(loop, self.result))

    def test_transaction_claim_requires_booking_and_simulator_disclosure(self):
        self.assertIn("unsupported_transaction_claim", codes("Your booking is confirmed.", self.result))
        hold = {"hold_id": "hold_abc", "flight": dict(self.flights[0]), "passenger": "Amy Gonzalez", "simulated": True, "status": "held"}
        booking = {"booking_id": "booking_abc", "hold_id": "hold_abc", "flight": dict(self.flights[0]), "passenger": "Amy Gonzalez", "simulated": True, "status": "confirmed"}
        booked = outcome(self.flights, booking=booking, hold=hold)
        self.assertIn("simulator_disclosure_missing", codes("Your booking is confirmed.", booked))
        self.assertIn("simulator_disclosure_missing", codes("Your not simulated booking is confirmed.", booked))
        self.assertNotIn("unsupported_transaction_claim", codes("Your simulated booking is confirmed for flight 1027.", booked))
        self.assertIn("transaction_field_contradiction", codes(
            "Your simulated booking is confirmed. booking_id: booking_wrong", booked
        ))

    def test_semantic_claims_require_outcome_evidence(self):
        unsupported = (
            "Your reservation is now in place in the simulator.",
            "Your seat is held in the simulator. Hold reference HOLD-FAKE.",
            "Your payment has been processed.",
            "Your card has been charged $100.",
        )
        for text in unsupported:
            with self.subTest(text=text):
                self.assertTrue(codes(text, self.result) & {
                    "unsupported_transaction_claim", "unsupported_real_transaction"
                })

        flight = {
            "flight_number": 1027, "price": 240, "airline": "Delta Airlines",
            "departure_airport": "BOS", "return_airport": "LAX",
            "departure_month": "June", "departure_day": 12,
            "departure_time_num": 9, "num_connections": 1,
        }
        grounded = outcome((flight,))
        for text, expected in (
            ("Flight 1027 departs June 13 at 10 AM.", {"unsupported_date", "unsupported_departure_time"}),
            ("Flight 1027 is on Emirates.", {"unsupported_airline"}),
            ("Flight 1027 costs 999 dollars.", {"unsupported_price"}),
            ("Flight 1027 is sold out.", {"availability_contradiction"}),
            ("Flight 1027 goes from BOS to JFK.", {"unsupported_route"}),
            ("Flight 1027 has 2 connections.", {"unsupported_stops"}),
        ):
            with self.subTest(text=text):
                self.assertTrue(expected <= codes(text, grounded))
        self.assertFalse(validate_output("Flight 1027 has no hidden fees.", grounded))

    def test_affirmative_transaction_status_requires_evidence_but_negation_is_allowed(self):
        for text in (
            "Your ticket is ready for travel.",
            "Your reservation is active.",
            "Your seat is guaranteed.",
        ):
            with self.subTest(text=text):
                self.assertIn("unsupported_transaction_claim", codes(text, self.result))
        self.assertFalse(codes("Your card has not been charged.", self.result) & {
            "unsupported_transaction_claim", "unsupported_real_transaction"
        })
        for mixed in (
            "Your payment has not been processed, but your reservation is confirmed.",
            "Your card was not charged and your seat is allocated.",
            "No payment was charged, but your ticket is ready to fly.",
        ):
            with self.subTest(text=mixed):
                self.assertIn("unsupported_transaction_claim", codes(mixed, self.result))

    def test_claims_are_clause_local_and_evidence_driven(self):
        for text in (
            "Your reservation is confirmed.",
            "Your payment has been processed.",
            "Your ticket is ready to fly.",
            "Your seat is allocated.",
            "Flight 1027 is available.",
        ):
            with self.subTest(text=text):
                self.assertTrue(codes(text, outcome(())) & {
                    "unsupported_transaction_claim", "unsupported_real_transaction", "availability_contradiction"
                })
        self.assertFalse(codes("Your reservation is not confirmed, and your card has not been charged.", self.result)
                         & {"unsupported_transaction_claim", "unsupported_real_transaction"})

    def test_flight_fact_formats_and_roles(self):
        flight = {
            "flight_number": 1027, "price": 240, "airline": "Delta Airlines",
            "departure_airport": "JFK", "return_airport": "LAX",
            "departure_month": "January", "departure_day": 1,
            "departure_time_num": 12, "return_month": "January", "return_day": 3,
            "return_time_num": 14, "num_connections": 2,
        }
        grounded = outcome((flight,))
        good = (
            "Delta operates flight 1027 from jfk to lax for two hundred forty dollars, with a pair of connections.",
            "Flight 1027 runs on Delta Airlines, JFK/LAX, departs Jan. 1 at 12:00 and returns January 3 at 14:00.",
            "Flight 1027 is on Delta Airlines, departs January 1 at noon and returns Jan 3 at 2 PM.",
        )
        for text in good:
            with self.subTest(text=text):
                self.assertFalse(codes(text, grounded) & {
                    "unsupported_airline", "unsupported_price", "unsupported_route", "unsupported_stops",
                    "unsupported_date", "unsupported_departure_time", "unsupported_return_time",
                })
        bad = (
            "United operates flight 1027.",
            "Flight 1027 flies jfk/sfo.",
            "Flight 1027 costs two hundred fifty dollars.",
            "Flight 1027 has three connections.",
            "Flight 1027 departs Jan 3 and returns Jan 1.",
            "Flight 1027 departs at 14:00 and returns at 12:00.",
        )
        expected = ("unsupported_airline", "unsupported_route", "unsupported_price", "unsupported_stops",
                    "unsupported_date", "unsupported_departure_time")
        for text, code in zip(bad, expected):
            with self.subTest(text=text):
                self.assertIn(code, codes(text, grounded))

    def test_flight_facts_are_checked_against_the_matching_evidence_fields(self):
        flight = {
            "flight_number": 1027, "price": 240, "airline": "Delta Airlines",
            "departure_airport": "JFK", "return_airport": "LAX",
            "departure_month": "January", "departure_day": 1,
            "departure_time_num": 12, "return_month": "January", "return_day": 3,
            "return_time_num": 14, "num_connections": 1,
        }
        grounded = outcome((flight,))
        for text in (
            "Flight 1027 is on Emirates today.",
            "Emirates operates flight 1027.",
            "Flight 1027 flies jfk to sfo.",
            "Flight 1027 has two connections.",
            "Flight 1027 has 2 connections.",
            "Flight 1027 departs January 9 at 12:59 PM.",
        ):
            with self.subTest(text=text):
                self.assertTrue(codes(text, grounded) & {
                    "unsupported_airline", "unsupported_route", "unsupported_stops",
                    "unsupported_date", "unsupported_departure_time",
                })
        self.assertFalse(validate_output(
            "Flight 1027 departs January 1 at 12 PM and returns January 3 at 2 PM. It is not sold out.", grounded
        ))

    def test_trailing_thank_you_loop_is_detected(self):
        self.assertIn("repeated_sentence", codes(
            "Here are the options. Thank you Thank you Thank you", self.result
        ))
        self.assertIn("repeated_sentence", codes(
            "Please check this. Thank you, thank you, thank you. Goodbye, kindly.", self.result
        ))
        self.assertFalse(codes("Flight 1027 costs $240. Flight 1027 departs at noon.", self.result) & {
            "repeated_sentence", "repeated_ngram", "closing_loop"
        })

    def test_reservation_validity_and_negation_are_clause_local(self):
        for text in ("Your reservation is valid.", "Your ticket remains valid.", "Your ticket is valid."):
            with self.subTest(text=text):
                self.assertIn("unsupported_transaction_claim", codes(text, self.result))
        self.assertIn("unsupported_real_transaction", codes(
            "Your card was charged despite your ticket not being allocated.", self.result
        ))
        self.assertIn("availability_contradiction", codes("Flight 1027 is not available.", self.result))
        self.assertNotIn("unsupported_transaction_claim", codes("Your ticket is not allocated.", self.result))

    def test_pair_stop_words_and_mixed_date_time_roles(self):
        flight = {
            "flight_number": 1027, "price": 240, "airline": "Delta Airlines",
            "departure_airport": "JFK", "return_airport": "LAX",
            "departure_month": "January", "departure_day": 1,
            "departure_time_num": 0, "return_month": "January", "return_day": 3,
            "return_time_num": 12, "num_connections": 1,
        }
        grounded = outcome((flight,))
        for text in (
            "Flight 1027 has a pair of connections.",
            "Flight 1027 has a pair of stops.",
        ):
            with self.subTest(text=text):
                self.assertIn("unsupported_stops", codes(text, grounded))
        for text in (
            "Flight 1027 returns Jan. 3 at noon and departs January 1 at midnight.",
            "Flight 1027 departs Jan 1 at 00:00 and returns January 3 at 12:00.",
            "Flight 1027 returns January 3 at 12 PM, then departs Jan. 1 at 12 AM.",
        ):
            with self.subTest(text=text):
                self.assertFalse(codes(text, grounded) & {
                    "unsupported_date", "unsupported_departure_time", "unsupported_return_time"
                })

    def test_structured_flight_summaries_do_not_trigger_short_loop_fallback(self):
        summaries = (
            # Scenario 23: ATL to CLT, Sept 15–17, at most $500 and one connection.
            "Flight 1015: AA, ATL to CLT, Sept 15 to Sept 17, $200, one connection. "
            "Flight 1016: AA, ATL to CLT, Sept 15 to Sept 17, $400, one connection. "
            "Flight 1020: AA, ATL to CLT, Sept 15 to Sept 17, $100, one connection.",
            # Scenario 33: four matching MSP to IAD options with repeated fields.
            "Flight 1014: Delta, MSP to IAD, Mar 31 to Apr 2, $100, one connection. "
            "Flight 1017: UA, MSP to IAD, Mar 31 to Apr 2, $100, one connection. "
            "Flight 1020: UA, MSP to IAD, Mar 31 to Apr 2, $300, one connection. "
            "Flight 1027: UA, MSP to IAD, Mar 31 to Apr 2, $200, one connection.",
        )
        for summary in summaries:
            with self.subTest(summary=summary[:50]):
                self.assertNotIn("repeated_sentence", codes(summary, self.result))
        self.assertNotIn("repeated_sentence", codes(
            "Flight 1027 costs $240. Flight 1031 costs $310. Flight 1027 departs at noon.", self.result
        ))

    def test_machine_readable_exception_wrapper(self):
        with self.assertRaises(OutputValidationException) as caught:
            validate_or_raise("", self.result)
        self.assertEqual(caught.exception.errors[0].as_dict()["code"], "empty_output")


if __name__ == "__main__":
    unittest.main()

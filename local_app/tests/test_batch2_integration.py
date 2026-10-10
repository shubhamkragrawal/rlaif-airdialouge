import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from local_app.audit import AuditLog
from local_app.controller import ConversationController
from local_app.knowledge import FlightKnowledgeBase
from local_app.limits import ExecutionLimits
from local_app.output_validation import validate_output
from local_app.outcomes import Intent, make_turn_outcome
from local_app.parsing import parse_passenger_name
from local_app.tools import FlightSearchResult, SearchCriteria


class Adapter:
    def __init__(self, response): self.response, self.calls = response, 0
    def generate(self, messages, flights): self.calls += 1; return self.response


class BatchTwoIntegrationTests(unittest.TestCase):
    def controller(self, **kwargs):
        return ConversationController(FlightKnowledgeBase(scenario_id=0), **kwargs)

    def test_passenger_forms_unicode_combined_and_rejection(self):
        samples = {
            "My name is José García": "José García",
            "Search IAD to LAS for Anne-Marie O’Neill": "Anne-Marie O'Neill",
            "passenger: Anne-Marie O’Neill": "Anne-Marie O'Neill",
            "Book flight 1027 for José García, please": "José García",
            "José García is traveling tomorrow": "José García",
        }
        for text, expected in samples.items():
            with self.subTest(text=text): self.assertEqual(parse_passenger_name(text).name, expected)
        self.assertEqual(parse_passenger_name("passenger " + "A" * 101).reason, "name_too_long")
        self.assertEqual(parse_passenger_name("passenger Alex <script>").status, "rejected")
        c = self.controller()
        state = c.new_session()
        c.handle(state, "Search IAD to LAS for José García")
        self.assertEqual(state.passenger, "José García")

    def test_explicit_passenger_introducers_ignore_case_and_internal_whitespace(self):
        c = self.controller()
        state = c.new_session()
        c.handle(state, "Find a flight from IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        result = c.handle(state, "MY   NAME   IS   JOHN SMITH")
        self.assertEqual(state.passenger, "John Smith")
        self.assertIsNone(state.booking)
        self.assertNotEqual(result.get("fallback_reason"), "passenger_airport_code_or_acronym")
        c.handle(state, "Please select flight 1027 for Jane Doe")
        result = c.handle(state, "Book  for JANE DOE")
        self.assertEqual(state.passenger, "Jane Doe")
        self.assertEqual(state.booking["passenger"], "Jane Doe")

    def test_typed_output_probes(self):
        flights = ({"flight_number": 1027, "price": 240, "airline": "Delta Airlines"},)
        out = make_turn_outcome(intent=Intent.SEARCH, deterministic_response="", search_results=FlightSearchResult(flights, SearchCriteria()))
        probes = {
            '{"x": 1}': "raw_json", "```json\n{}\n```": "code_fenced_json",
            "Hi <|assistant|>": "control_token", "Loop this phrase carefully now. Another sentence. Loop this phrase carefully now.": "repeated_sentence",
            "Alaska Airlines operates flight 1027 for $240.": "unsupported_airline",
            "Delta Airlines operates flight 1027 for $240.50.": "unsupported_price",
        }
        for text, code in probes.items():
            with self.subTest(text=text): self.assertIn(code, {e.code for e in validate_output(text, out)})
        empty = make_turn_outcome(intent=Intent.SEARCH, deterministic_response="", search_results=FlightSearchResult((), SearchCriteria()))
        self.assertIn("availability_contradiction", {e.code for e in validate_output("I found two flights.", empty)})

    def test_short_loops_and_flight_specific_itinerary_facts(self):
        flights = (
            {"flight_number": 1027, "price": 240, "airline": "United Airlines",
             "departure_airport": "IAD", "return_airport": "LAS", "num_connections": 1},
            {"flight_number": 1031, "price": 310, "airline": "Delta Airlines",
             "departure_airport": "JFK", "return_airport": "LAX", "num_connections": 0},
        )
        out = make_turn_outcome(intent=Intent.SEARCH, deterministic_response="", search_results=FlightSearchResult(flights, SearchCriteria()))
        self.assertIn("repeated_sentence", {e.code for e in validate_output("Thank you. " * 4, out)})
        self.assertIn("duplicate_lines", {e.code for e in validate_output("OK\nOK\nOK", out)})
        ordinary = "Flight 1027 and flight 1031 are both options."
        self.assertFalse({e.code for e in validate_output(ordinary, out)} & {"repeated_sentence", "unsupported_airline"})
        samples = (
            ("Delta Airlines flight 1027 costs $240.", "unsupported_airline"),
            ("Delta for flight 1027", "unsupported_airline"),
            ("delta for flight 1027", "unsupported_airline"),
            ("Frontier for flight 1027", "unsupported_airline"),
            ("Flight 1027 travels JFK-LAX.", "unsupported_route"),
            ("Flight 1027 is nonstop.", "unsupported_stops"),
            ("Flight 1027 is $310 and flight 1031 is $240.", "unsupported_price"),
        )
        for text, code in samples:
            with self.subTest(text=text):
                self.assertIn(code, {e.code for e in validate_output(text, out)})
        empty = make_turn_outcome(intent=Intent.SEARCH, deterministic_response="", search_results=FlightSearchResult((), SearchCriteria()))
        for text in ("Nothing matches.", "All flights are unavailable."):
            self.assertIn("availability_contradiction", {e.code for e in validate_output(text, out)})
            self.assertIn("availability_contradiction", {e.code for e in validate_output("Flight 1027 is here.", empty)})

    def test_adapter_deadline_and_reported_token_budget(self):
        class Slow:
            def __init__(self): self.calls = 0
            def generate(self, *_args):
                self.calls += 1
                time.sleep(0.06)
                return "A late answer."
        slow = Slow()
        c = self.controller(model_adapters={"dpo": slow}, limits=ExecutionLimits(max_session_seconds=0.015))
        slow_state = c.new_session("dpo")
        start = time.monotonic()
        result = c.handle(slow_state, "IAD to LAS")
        elapsed = time.monotonic() - start
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertLess(elapsed, 0.05)
        self.assertNotEqual(result["text"], "A late answer.")
        self.assertFalse(any(turn["content"] == "A late answer." for turn in slow_state.turns))
        second = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertEqual(second["fallback_reason"], "session_time_exceeded")
        self.assertEqual(slow.calls, 2)

        class OverBudget:
            last_usage = {"output_tokens": 9}
            def generate(self, *_args): return "A plausible short answer."
        c = self.controller(model_adapters={"dpo": OverBudget()},
                            limits=ExecutionLimits(max_generation_tokens=4))
        result = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertEqual(result["fallback_reason"], "generation_length_exceeded")
        self.assertFalse(result["model_used"])

        class BudgetAware:
            def __init__(self): self.kwargs = None
            def generate(self, _messages, _flights, *, max_new_tokens, deadline_seconds):
                self.kwargs = (max_new_tokens, deadline_seconds)
                return "A short answer."
        adapter = BudgetAware()
        c = self.controller(model_adapters={"dpo": adapter},
                            limits=ExecutionLimits(max_generation_tokens=7, max_session_seconds=3))
        c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertEqual(adapter.kwargs[0], 7)
        self.assertGreater(adapter.kwargs[1], 0)

    def test_cooperative_adapter_cancels_and_controller_recovers(self):
        class Cooperative:
            def __init__(self): self.calls = 0; self.entered = threading.Event(); self.exited = threading.Event()
            def generate(self, _messages, _flights, *, cancel_event, deadline_seconds):
                self.calls += 1
                self.entered.set()
                if self.calls == 1:
                    cancel_event.wait(1)
                    self.exited.set()
                    raise TimeoutError("cancelled cooperatively")
                self.exited.set()
                return "I can help with the available simulator flights."
        adapter = Cooperative()
        c = self.controller(model_adapters={"dpo": adapter}, limits=ExecutionLimits(max_session_seconds=0.02))
        result = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertTrue(adapter.exited.wait(0.1))
        self.assertFalse(c.adapter_worker_active)
        next_result = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertTrue(next_result["model_used"])
        self.assertEqual(adapter.calls, 2)

    def test_model_loader_is_inside_deadline_and_late_load_is_discarded(self):
        finished = threading.Event()
        class Loaded:
            calls = 0
            def generate(self, *_args):
                self.calls += 1
                return "I can help with the available simulator flights."
        loaded = Loaded()
        attempts = [0]
        def loader(_name):
            attempts[0] += 1
            if attempts[0] == 1:
                time.sleep(0.08)
                finished.set()
            return loaded
        c = self.controller(model_loader=loader, limits=ExecutionLimits(max_session_seconds=0.02))
        start = time.monotonic()
        result = c.handle(c.new_session("dpo"), "IAD to LAS")
        elapsed = time.monotonic() - start
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertLess(elapsed, 0.06)
        self.assertEqual(loaded.calls, 0)
        self.assertTrue(finished.wait(0.2))
        self.assertFalse(c.adapter_worker_active)
        recovered = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertTrue(recovered["model_used"])
        self.assertEqual(loaded.calls, 1)

    def test_passenger_for_is_not_misread_as_flight_selection(self):
        c = self.controller()
        state = c.new_session()
        result = c.handle(state, "Book for Amy Gonzalez")
        self.assertNotEqual(result["fallback_reason"], "invalid_flight_selection")
        self.assertEqual(state.passenger, "Amy Gonzalez")

    def test_tool_deadline_and_tool_budget_roll_back_transactions(self):
        c = self.controller(limits=ExecutionLimits(max_tool_calls=2))
        state = c.new_session()
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        result = c.handle(state, "Book flight 1027")
        self.assertEqual(result["fallback_reason"], "tool_call_limit_exceeded")
        self.assertEqual(c.simulator.holds, {})
        self.assertEqual(c.simulator.bookings, {})
        self.assertIsNone(state.booking)

    def test_late_timed_out_hold_cleanup_preserves_another_sessions_booking(self):
        c = self.controller(limits=ExecutionLimits(max_session_seconds=5))
        amy = c.new_session()
        c.handle(amy, "Find a flight from IAD to LAS")
        c.handle(amy, "My name is Amy Gonzalez")
        c.handle(amy, "flight 1027")
        bob = c.new_session()
        c.handle(bob, "Find a flight from IAD to LAS")
        c.handle(bob, "My name is Bob Jones")
        c.handle(bob, "flight 1027")
        from local_app.limits import ExecutionBudget
        amy.budget = ExecutionBudget(ExecutionLimits(max_session_seconds=0.04))

        entered, release = threading.Event(), threading.Event()
        original = c.simulator.hold_flight
        def slow_live_wrapper(flight_id, passenger):
            entered.set()
            release.wait(1)
            return original(flight_id, passenger)
        c.simulator.hold_flight = slow_live_wrapper
        amy_result = c.handle(amy, "Book flight 1027")
        self.assertEqual(amy_result["fallback_reason"], "session_time_exceeded")
        self.assertTrue(entered.wait(0.5))

        del c.simulator.hold_flight
        bob_result = c.handle(bob, "Book flight 1027")
        self.assertIsNotNone(bob.booking)
        booking_id = bob.booking["booking_id"]
        release.set()
        time.sleep(0.05)
        self.assertTrue(bob_result["text"])
        self.assertIn(booking_id, c.simulator.bookings)
        self.assertEqual(c.simulator.bookings[booking_id], bob.booking)

        from local_app.limits import ExecutionBudget
        now = [0.0]
        c = self.controller(limits=ExecutionLimits(max_session_seconds=5))
        state = c.new_session()
        state.budget = ExecutionBudget(c.limits, clock=lambda: now[0])
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        original_hold = c.simulator.hold_flight
        def late_hold(*args):
            result = original_hold(*args)
            now[0] = 5.1
            return result
        c.simulator.hold_flight = late_hold
        result = c.handle(state, "Book flight 1027")
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertEqual(c.simulator.holds, {})
        self.assertEqual(c.simulator.bookings, {})
        self.assertIsNone(state.booking)

    def test_late_same_passenger_hold_cleanup_preserves_newer_booking_and_retry(self):
        c = self.controller(limits=ExecutionLimits(max_session_seconds=5))
        first = c.new_session()
        second = c.new_session()
        for state in (first, second):
            c.handle(state, "Find a flight from IAD to LAS")
            c.handle(state, "My name is Amy Gonzalez")
            c.handle(state, "flight 1027")
        from local_app.limits import ExecutionBudget
        first.budget = ExecutionBudget(ExecutionLimits(max_session_seconds=0.04))

        entered, release = threading.Event(), threading.Event()
        original = c.simulator.hold_flight
        def blocked_live_wrapper(flight_id, passenger):
            entered.set()
            release.wait(1)
            return original(flight_id, passenger)
        c.simulator.hold_flight = blocked_live_wrapper
        timed_out = c.handle(first, "Book flight 1027")
        self.assertEqual(timed_out["fallback_reason"], "session_time_exceeded")
        self.assertTrue(entered.wait(0.5))

        del c.simulator.hold_flight
        booked = c.handle(second, "Book flight 1027")
        self.assertIsNotNone(second.booking)
        hold_id = second.booking["hold_id"]
        booking_id = second.booking["booking_id"]
        self.assertIn(hold_id, c.simulator.holds)
        self.assertIn(booking_id, c.simulator.bookings)

        release.set()
        time.sleep(0.05)
        self.assertEqual(c.simulator.holds[hold_id]["passenger"], "Amy Gonzalez")
        self.assertEqual(c.simulator.bookings[booking_id], second.booking)
        repeated = c.handle(second, "Book flight 1027")
        self.assertIsNone(repeated["fallback_reason"])
        self.assertEqual(second.booking["booking_id"], booking_id)
        self.assertEqual(c.simulator.bookings[booking_id], second.booking)

    def test_audit_redacts_exact_passenger_without_learning_trailing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            c = self.controller(audit=audit)
            state = c.new_session()
            c.handle(state, "Search IAD to LAS for Amy Gonzalez and depart early")
            audit.write("assistant_response", {"text": "Amy Gonzalez can depart early."})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            serialized = json.dumps(records)
            self.assertNotIn("Amy Gonzalez", serialized)
            self.assertIn("depart early", serialized)
            self.assertIn("and depart early", records[0]["payload"]["message"])

    def test_model_rejection_reason_is_specific(self):
        for content, reason in (("{\"x\":1}", "raw_json"), ("<|assistant|>", "control_token"),
                                ("Delta Airlines flight 1027 costs $99.", "unsupported_price")):
            adapter = Adapter(content)
            c = self.controller(model_adapters={"dpo": adapter})
            result = c.handle(c.new_session("dpo"), "IAD to LAS")
            self.assertFalse(result["model_used"])
            self.assertEqual(result["validation_reason"], reason)

    def test_controller_rejects_airline_after_flight_id(self):
        for airline in ("Delta", "Frontier"):
            with self.subTest(airline=airline):
                adapter = Adapter(f"Flight 1027 is on {airline}.")
                c = self.controller(model_adapters={"dpo": adapter})
                result = c.handle(c.new_session("dpo"), "IAD to LAS")
                self.assertFalse(result["model_used"])
                self.assertEqual(result["fallback_reason"], "unsupported_airline")
                self.assertNotEqual(result["text"], f"Flight 1027 is on {airline}.")

    def test_audit_redacts_phrases_structured_fields_and_echoes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = AuditLog(path)
            for text in ("My name is José García", "passenger: Anne-Marie O’Neill",
                         "Book flight 1027 for José García", "Anne-Marie O’Neill is traveling",
                         "Search IAD to LAS for José García"):
                audit.write("user_message", {"message": text})
            audit.write("assistant_response", {"passenger": "José García", "text": "Booking for José García"})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            serialized = json.dumps(records, ensure_ascii=False)
            self.assertNotIn("José García", serialized)

    def test_turn_and_tool_exhaustion_preserve_machine_reasons(self):
        c = self.controller(limits=ExecutionLimits(max_turns=1, max_tool_calls=0))
        s = c.new_session()
        exhausted = c.handle(s, "IAD to LAS")
        self.assertEqual(exhausted["fallback_reason"], "tool_call_limit_exceeded")
        next_turn = c.handle(s, "hello")
        self.assertEqual(next_turn["fallback_reason"], "turn_limit_exceeded")

    def test_retry_exhaustion_and_time_boundary(self):
        a = Adapter("some response")
        c = self.controller(limits=ExecutionLimits(max_retries=0), model_adapters={"dpo": a})
        res = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertTrue(res["model_used"])
        self.assertEqual(c.new_session("dpo").budget.snapshot().retries, 0)
        now = [0.0]
        from local_app.limits import ExecutionBudget
        budget = ExecutionBudget(ExecutionLimits(max_session_seconds=2), clock=lambda: now[0])
        now[0] = 2
        self.assertEqual(budget.consume_turn(), ())
        now[0] = 2.01
        self.assertEqual(budget.consume_turn()[0].value, "session_time_exceeded")

    def test_latest_validation_and_privacy_probes(self):
        flight = {"flight_number": 1027, "price": 100, "airline": "Southwest",
                  "departure_airport": "IAD", "return_airport": "LAS", "num_connections": 1}
        out = make_turn_outcome(intent=Intent.SEARCH, deterministic_response="",
            search_results=FlightSearchResult((flight,), SearchCriteria()))
        for text, code in (("<tool_call>{\"name\":\"book_flight\"}</tool_call>", "control_token"),
                           ("\u200b\u200b", "empty_output"),
                           ("Thank you Thank you Thank you", "repeated_sentence"),
                           ("I made your reservation in the simulator.", "unsupported_transaction_claim"),
                           ("Your card was charged $100.", "unsupported_real_transaction"),
                           ("Flight 1027 costs USD 999.", "unsupported_price"),
                           ("Flight 1027 is unavailable.", "availability_contradiction"),
                           ("None of the flights are available.", "availability_contradiction")):
            with self.subTest(text=text):
                self.assertIn(code, {error.code for error in validate_output(text, out)})
        self.assertFalse(validate_output("Flight 1027 has no hidden fees.", out))
        with tempfile.TemporaryDirectory() as directory:
            audit = AuditLog(Path(directory) / "audit.jsonl")
            names = ("Anne-Marie O’Neill", "Jose\u0301 Garci\u0301a", "Ａｍｙ Ｇｏｎｚａｌｅｚ", "Jean‑Luc Picard")
            for name in names:
                audit.register_passenger(name)
            audit.write("echo", {"text": " | ".join(names)})
            raw = (Path(directory) / "audit.jsonl").read_text(encoding="utf-8")
            for name in names:
                self.assertNotIn(name, raw)

    def test_explicit_uppercase_name_replaces_previous_passenger(self):
        c = self.controller()
        state = c.new_session()
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Bob Smith")
        result = c.handle(state, "Book flight 1027 for JOHN SMITH")
        self.assertEqual(state.passenger, "John Smith")
        self.assertNotEqual(result.get("fallback_reason"), "invalid_flight_selection")
        self.assertEqual(state.booking["passenger"], "John Smith")

    def test_both_explicit_rejected_passenger_forms_block_booking(self):
        for request in ("Book flight 1027, passenger",):
            with self.subTest(request=request):
                c = self.controller()
                state = c.new_session()
                c.handle(state, "IAD to LAS")
                c.handle(state, "My name is Bob Smith")
                result = c.handle(state, request)
                self.assertFalse(c.simulator.bookings)
                self.assertFalse(c.simulator.holds)
                self.assertIsNone(state.booking)
                self.assertIsNone(state.passenger)
                self.assertTrue(str(result["fallback_reason"]).startswith("passenger_"))

    def test_cached_repeat_booking_is_bounded(self):
        c = self.controller(limits=ExecutionLimits(max_session_seconds=0.08))
        state = c.new_session()
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        first = c.handle(state, "Book flight 1027")
        self.assertTrue(state.booking)
        original = c.simulator.book_flight
        def slow(hold_id):
            time.sleep(0.2)
            return original(hold_id)
        c.simulator.book_flight = slow
        now = time.monotonic()
        repeated = c.handle(state, "Book flight 1027")
        self.assertLess(time.monotonic() - now, 0.15)
        self.assertNotEqual(repeated.get("text"), first.get("text"))
        time.sleep(0.22)
        self.assertEqual(len(c.simulator.bookings), 1)

    def test_cached_repeat_rechecks_deadline_after_booking_outcome(self):
        from local_app.limits import ExecutionBudget
        now = [0.0]
        c = self.controller(limits=ExecutionLimits(max_session_seconds=1))
        state = c.new_session()
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        c.handle(state, "Book flight 1027")
        state.budget = ExecutionBudget(c.limits, clock=lambda: now[0])
        original = c._booking_outcome
        def expire_after_outcome(booking, hold):
            result = original(booking, hold)
            now[0] = 1.1
            return result
        c._booking_outcome = expire_after_outcome
        result = c.handle(state, "Book flight 1027")
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertIsNotNone(state.booking)
        self.assertEqual(len(c.simulator.bookings), 1)

    def test_late_wrapper_around_original_hold_cannot_mutate_live_simulator(self):
        c = self.controller(limits=ExecutionLimits(max_session_seconds=0.02))
        state = c.new_session()
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        original_hold = c.simulator.hold_flight
        def slow_wrapper(flight_id, passenger):
            time.sleep(0.12)
            return original_hold(flight_id, passenger)
        c.simulator.hold_flight = slow_wrapper
        result = c.handle(state, "Book flight 1027")
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        time.sleep(0.16)
        self.assertFalse(c.simulator.holds)

    def test_malformed_usage_is_rejected_and_unreported_usage_is_estimated(self):
        for usage in ({}, {"input_tokens": 1}, {"output_tokens": None},
                      {"input_tokens": "1", "output_tokens": 1},
                      {"input_tokens": True, "output_tokens": 1},
                      {"input_tokens": 1, "output_tokens": 0}):
            class AdapterWithUsage:
                last_usage = usage
                def generate(self, *_args): return "ok"
            c = self.controller(model_adapters={"dpo": AdapterWithUsage()},
                                limits=ExecutionLimits(max_generation_tokens=1))
            result = c.handle(c.new_session("dpo"), "IAD to LAS")
            self.assertFalse(result["model_used"])
            self.assertEqual(result["fallback_reason"], "malformed_usage")

        class LongWord:
            def generate(self, *_args): return "supercalifragilisticexpialidocious"
        c = self.controller(model_adapters={"dpo": LongWord()},
                            limits=ExecutionLimits(max_generation_tokens=1))
        result = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertEqual(result["fallback_reason"], "generation_length_exceeded")

        class Unreported:
            def generate(self, *_args): return "Flight 1027 is the grounded choice at $100."
        c = self.controller(model_adapters={"dpo": Unreported()})
        result = c.handle(c.new_session("dpo"), "Find a flight from IAD to LAS under $100")
        self.assertTrue(result["model_used"])
        self.assertGreater(result["usage"]["input_tokens"], 0)
        self.assertEqual(result["usage"]["output_tokens"], len(result["text"].encode("utf-8")))
        self.assertTrue(result["usage"]["estimated"])

    def test_hold_and_book_calls_are_hard_bounded_and_isolated(self):
        for slow_method in ("hold_flight", "book_flight"):
            with self.subTest(method=slow_method):
                c = self.controller(limits=ExecutionLimits(max_session_seconds=0.02))
                state = c.new_session()
                c.handle(state, "IAD to LAS")
                c.handle(state, "My name is Amy Gonzalez")
                import time as walltime
                if slow_method == "hold_flight":
                    c.simulator.hold_flight = lambda *_args: (walltime.sleep(0.15), {})[1]
                else:
                    c.simulator.book_flight = lambda *_args: (walltime.sleep(0.15), {})[1]
                started = time.monotonic()
                result = c.handle(state, "Book flight 1027")
                elapsed = time.monotonic() - started
                self.assertEqual(result["fallback_reason"], "session_time_exceeded")
                self.assertLess(elapsed, 0.08)
                self.assertIsNone(state.booking)
                self.assertFalse(c.simulator.holds)
                self.assertFalse(c.simulator.bookings)

    def test_deadline_rechecked_after_booking_outcome_before_commit(self):
        from local_app.limits import ExecutionBudget
        now = [0.0]
        c = self.controller(limits=ExecutionLimits(max_session_seconds=1))
        state = c.new_session()
        state.budget = ExecutionBudget(c.limits, clock=lambda: now[0])
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        original = c._booking_outcome
        def late_outcome(booking, hold):
            result = original(booking, hold)
            now[0] = 1.1
            return result
        c._booking_outcome = late_outcome
        result = c.handle(state, "Book flight 1027")
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertIsNone(state.booking)
        self.assertIsNone(state.hold_id)
        self.assertFalse(c.simulator.holds)
        self.assertFalse(c.simulator.bookings)

    def test_deadline_after_commit_rolls_back_transaction_owned_writes(self):
        from local_app.limits import ExecutionBudget
        c = self.controller(limits=ExecutionLimits(max_session_seconds=1))
        state = c.new_session()
        c.handle(state, "IAD to LAS")
        c.handle(state, "My name is Amy Gonzalez")
        state.budget = ExecutionBudget(
            ExecutionLimits(max_session_seconds=1),
            clock=lambda: 1.1 if c.simulator.bookings else 0.0,
        )

        result = c.handle(state, "Book flight 1027")

        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertFalse(c.simulator.holds)
        self.assertFalse(c.simulator.bookings)
        self.assertIsNone(state.hold_id)
        self.assertIsNone(state.booking)

    def test_blocked_models_do_not_starve_fresh_session_searches(self):
        entered = threading.Event()
        release = threading.Event()
        class NonCooperative:
            def generate(self, *_args):
                entered.set()
                release.wait(1)
                return "late model result"
        c = self.controller(model_adapters={"dpo": NonCooperative()},
                            limits=ExecutionLimits(max_session_seconds=0.015))
        for _ in range(2):
            result = c.handle(c.new_session("dpo"), "IAD to LAS")
            self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertTrue(entered.is_set())
        fresh = c.new_session("dpo")
        result = c.handle(fresh, "IAD to LAS")
        self.assertIsNotNone(fresh.last_search)
        self.assertEqual(result["fallback_reason"], "model_capacity_exceeded")
        release.set()

    def test_runtime_manager_checks_cancellation_inside_commit_lock(self):
        from local_app.runtime import RuntimeConfig, RuntimeManager, RuntimeUnavailable
        with tempfile.TemporaryDirectory() as directory:
            adapter_path = Path(directory) / "adapter"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")
            factory_returned = threading.Event()
            factory_entered = threading.Event()
            allow_factory = threading.Event()
            class Candidate: pass
            def factory(_name, _config):
                factory_entered.set()
                allow_factory.wait(1)
                factory_returned.set()
                return Candidate()
            manager = RuntimeManager(RuntimeConfig(adapter_paths={"dpo": str(adapter_path)}), factory=factory)
            cancelled = threading.Event()
            worker_done = threading.Event()
            errors = []
            def load():
                try: manager.get("dpo", cancel_event=cancelled, deadline_seconds=2)
                except Exception as exc: errors.append(exc)
                finally: worker_done.set()
            worker = threading.Thread(target=load)
            worker.start()
            # Wait until factory is entered, then block its return/commit behind
            # the manager lock to exercise the critical cancellation window.
            self.assertTrue(factory_entered.wait(0.5))
            manager._lock.acquire()
            try:
                allow_factory.set()
                self.assertTrue(factory_returned.wait(0.5))
                cancelled.set()
            finally:
                manager._lock.release()
            self.assertTrue(worker_done.wait(1), "runtime get did not finish after cancellation")
            worker.join()
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeUnavailable)
            self.assertIsNone(manager._runtime)
            self.assertIsNone(manager._name)

    def test_unreported_generation_usage_is_bounded_and_normal_turns_do_not_spend_retries(self):
        class Unreported:
            def generate(self, *_args): return "one two three four five six seven eight"
        c = self.controller(model_adapters={"dpo": Unreported()},
                            limits=ExecutionLimits(max_generation_tokens=4))
        result = c.handle(c.new_session("dpo"), "IAD to LAS")
        self.assertEqual(result["fallback_reason"], "generation_length_exceeded")
        class Ordinary:
            def generate(self, *_args): return "I can help with these available simulator flights."
        c = self.controller(model_adapters={"dpo": Ordinary()}, limits=ExecutionLimits(max_retries=0))
        state = c.new_session("dpo")
        for _ in range(4):
            self.assertTrue(c.handle(state, "hello")["model_used"])
        self.assertEqual(state.budget.snapshot().retries, 0)

    def test_controller_session_deadline_prevents_tool_attempt(self):
        from local_app.limits import ExecutionBudget
        now = [0.0]
        c = self.controller(limits=ExecutionLimits(max_session_seconds=1))
        s = c.new_session()
        s.budget = ExecutionBudget(c.limits, clock=lambda: now[0])
        now[0] = 1.01
        result = c.handle(s, "IAD to LAS")
        self.assertEqual(result["fallback_reason"], "session_time_exceeded")
        self.assertIsNone(s.last_search)

    def test_session_payload_compatibility_excludes_internal_budget(self):
        from local_app.api import _session_payload
        c = self.controller()
        payload = _session_payload("session-test", c.new_session())
        self.assertEqual(payload["session_id"], "session-test")
        self.assertNotIn("budget", payload)

    def test_adapter_failure_fallback_is_machine_specific(self):
        class Broken:
            def __init__(self, error): self.error = error
            def generate(self, *_args): raise self.error
        for error, reason in ((TimeoutError(), "model_timeout"), (RuntimeError(), "model_runtime_error")):
            c = self.controller(model_adapters={"dpo": Broken(error)})
            result = c.handle(c.new_session("dpo"), "IAD to LAS")
            self.assertEqual(result["fallback_reason"], reason)


if __name__ == "__main__": unittest.main()

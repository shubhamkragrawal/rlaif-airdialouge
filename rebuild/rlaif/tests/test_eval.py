"""Run with python3 -m unittest discover -s RLAIF/rebuild -p test_eval.py."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from rlaif.evaluation.eval_core import (aggregate, check_outcome, make_manifest, matching_flights,
                       named_flights, paired_comparison, parse_marker)
from rlaif.evaluation.eval_qwen import (BudgetLimitError, EvalGeminiClient, parse_customer, run_scenario,
                       simulator_prompt, simulator_repair_prompt)
from rlaif.shared.data import load_dialogues, load_splits


def record():
    return {"dialogue_id": 1, "line_hash": "abc", "intent": {"goal": "book", "name": "Jane Smith",
            "departure_airport": "IAD", "return_airport": "LAS", "departure_day": "01",
            "return_day": "3", "departure_month": "Jan", "return_month": "Jan",
            "max_price": 200, "max_connections": 1}, "expected_action": {"status": "book", "flight": [1027]},
            "kb": [{"flight_number": 1027, "departure_airport": "IAD", "return_airport": "LAS",
                    "departure_day": "1", "return_day": "03", "departure_month": "Jan",
                    "return_month": "Jan", "price": 200, "num_connections": 1}]}


class StubPolicy:
    def __init__(self):
        self.calls = 0

    def reply(self, rec, transcript):
        self.calls += 1
        return "Flight number 1027 costs $200. May I book it?" if self.calls == 1 else "Booked.\nBOOKING_CONFIRMED 1027"


class StubSimulator:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def rank(self, prompt):
        self.calls += 1
        return None if self.fail else json.dumps({"text": "Yes, book flight 1027. My name is Jane Smith.", "accepted_flight": 1027})


class RepairingSimulator:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def rank(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return json.dumps({"text": "Yes, book flight 9999.", "accepted_flight": 9999})
        return json.dumps({"text": "Yes, book flight 1027. My name is Jane Smith.", "accepted_flight": 1027})


class EvalTests(unittest.TestCase):
    def test_constraints_and_unknown_fields(self):
        rec = record()
        self.assertEqual(matching_flights(rec), [1027])  # day leading zeros normalise
        rec["intent"]["max_price"] = 199
        self.assertEqual(matching_flights(rec), [])
        rec["intent"]["max_price"] = 200
        rec["intent"]["max_connections"] = 0
        self.assertEqual(matching_flights(rec), [])
        rec["intent"]["max_connections"] = -1
        self.assertEqual(matching_flights(rec), [1027])
        rec["intent"]["return_time"] = "evening"
        with self.assertRaises(ValueError):
            matching_flights(rec)

    def test_marker_requires_exact_one_last_line(self):
        self.assertEqual(parse_marker("Thank you.\nBOOKING_CONFIRMED 1027"), ("confirmed", 1027))
        self.assertEqual(parse_marker("BOOKING_FAILED\nThanks"), ("invalid", None))
        self.assertEqual(parse_marker("BOOKING_CONFIRMED 1027\nBOOKING_FAILED"), ("invalid", None))
        self.assertEqual(parse_marker("BOOKING_CONFIRMED"), ("invalid", None))

    def test_booking_requires_name_consent_and_valid_flight(self):
        state = {"transcript": [{"role": "user", "content": "I am Jane Smith. Book 1027."},
                               {"role": "assistant", "content": "BOOKING_CONFIRMED 1027"}], "accepted_flights": [1027]}
        self.assertTrue(check_outcome(record(), state)["task_success"])
        state["accepted_flights"] = []
        self.assertFalse(check_outcome(record(), state)["task_success"])
        state["accepted_flights"] = [9999]
        state["transcript"][-1]["content"] = "BOOKING_CONFIRMED 9999"
        metrics = check_outcome(record(), state)
        self.assertFalse(metrics["task_success"])
        self.assertTrue(metrics["unknown_flight_id_hallucination"])

    def test_no_flight_and_unjustified_decline(self):
        state = {"transcript": [{"role": "assistant", "content": "BOOKING_FAILED"}]}
        self.assertFalse(check_outcome(record(), state)["task_success"])
        rec = record()
        rec["intent"]["max_price"] = 10
        self.assertTrue(check_outcome(rec, state)["task_success"])

    def test_customer_cannot_see_oracle_or_invent_consent(self):
        rec = record()
        rec["kb"][0]["SECRET_KB_FIELD"] = "hidden-value"
        rec["expected_action"]["secret"] = "oracle-value"
        prompt = simulator_prompt(rec, [{"role": "user", "content": "Hi"}])
        self.assertNotIn("hidden-value", prompt)
        self.assertNotIn("oracle-value", prompt)
        self.assertIn("Jane Smith", prompt)
        with self.assertRaises(ValueError):
            parse_customer('{"text":"Yes","accepted_flight":9999}', [])
        with self.assertRaises(ValueError):
            parse_customer('{"text":"BOOKING_FAILED","accepted_flight":null}', [])
        self.assertEqual(named_flights("Flight no. 1027, $200, 12:00."), {1027})

    def test_customer_consent_accepts_known_bare_and_table_flight_offer(self):
        raw = '{"text":"Yes, book it","accepted_flight":1027}'
        for offer in ("The flight is 1027 and costs $200.",
                      "1027 | UA | economy | IAD Jan 1 12:00 | LAS Jan 3 19:00 | $200"):
            self.assertEqual(parse_customer(raw, [{"role": "assistant", "content": offer}], {1027}),
                             ("Yes, book it", 1027))
        with self.assertRaises(ValueError):
            parse_customer(raw, [{"role": "assistant", "content": "Price $1027"}], {9999})

    def test_resume_keeps_saved_policy_turn(self):
        state = {}
        saved = []
        policy = StubPolicy()
        with self.assertRaises(RuntimeError):
            run_scenario(record(), state, policy, StubSimulator(fail=True), 5,
                         lambda s: saved.append(copy.deepcopy(s)))
        self.assertEqual(policy.calls, 1)
        self.assertEqual(state["transcript"][-1]["role"], "assistant")
        result = run_scenario(record(), state, policy, StubSimulator(), 5, lambda s: None)
        self.assertEqual(policy.calls, 2)  # one offer and one confirmation, never a duplicate offer
        self.assertTrue(result["metrics"]["task_success"])
        self.assertEqual(result["status"], "complete")

    def test_invalid_simulator_consent_gets_bounded_semantic_repair(self):
        state = {}
        simulator = RepairingSimulator()
        result = run_scenario(record(), state, StubPolicy(), simulator, 5, lambda s: None)
        self.assertEqual(simulator.calls, 2)
        self.assertIn("unoffered flight", simulator.prompts[1])
        self.assertEqual(result["simulator_calls"][0]["phase"], "primary")
        self.assertEqual(result["simulator_calls"][1]["phase"], "semantic_repair")
        self.assertIn("validation_error", result["simulator_calls"][0])
        self.assertTrue(result["metrics"]["task_success"])

    def test_repair_prompt_never_exposes_oracle(self):
        rec = record()
        rec["expected_action"]["secret"] = "oracle-value"
        prompt = simulator_repair_prompt(rec, [{"role": "assistant", "content": "No flights."}],
                                         '{"accepted_flight":9999}', "unoffered flight")
        self.assertNotIn("oracle-value", prompt)
        self.assertIn("unoffered flight", prompt)

    def test_real_manifest_fixed50_disjoint_and_label_consistent(self):
        dialogues, splits = load_dialogues(), load_splits()
        manifest = make_manifest(dialogues, splits)
        self.assertEqual(manifest, make_manifest(dialogues, splits))
        self.assertEqual(manifest["eligible_count"], 105)
        self.assertEqual(manifest["exclusion_counts"]["structured_intent_does_not_recover_label"], 82)
        ids = {s["dialogue_id"] for s in manifest["scenarios"]}
        self.assertEqual(len(ids), 50)
        self.assertTrue(ids <= set(splits["held_out"]))
        self.assertFalse(ids & (set(splits["sft"]) | set(splits["pair_gen"])))
        by_id = {r["dialogue_id"]: r for r in dialogues}
        for scenario in manifest["scenarios"]:
            self.assertEqual(scenario["matching_flights"], sorted(by_id[scenario["dialogue_id"]]["expected_action"]["flight"]))

    def test_paired_stats_and_incomplete_summary(self):
        def row(i, success):
            return {"dialogue_id": i, "status": "complete", "metrics": {"task_success": success,
                    "booking_success": success, "requires_booking": True,
                    "unknown_flight_id_hallucination": False, "agent_turns": 4}}
        left = [row(1, False), row(2, True)]
        right = [row(1, True), row(2, True)]
        delta = paired_comparison(left, right, samples=100)
        self.assertEqual(delta["right_minus_left_success_rate"], .5)
        self.assertEqual(delta["right_only_successes"], 1)
        with self.assertRaises(ValueError):
            paired_comparison(left, right[:1])
        summary = aggregate(left, 50)
        self.assertTrue(summary["provisional"])
        self.assertEqual(summary["incomplete_scenarios"], 48)
        self.assertEqual(summary["judge_scored_scenarios"], 0)
        self.assertIsNone(summary["mean_judge_score"])

    def test_api_caps_stop_before_network_request(self):
        fake_types = types.SimpleNamespace(GenerateContentConfig=lambda **kw: kw,
                                          ThinkingConfig=lambda **kw: kw)
        fake_genai = types.ModuleType("google.genai")
        fake_genai.types = fake_types
        client = EvalGeminiClient.__new__(EvalGeminiClient)
        client.model = "test"
        client.max_calls, client.max_tokens, client.max_output_tokens, client.max_retries = 1, 1000, 10, 1
        client.usage = {"calls_attempted": 1, "calls_successful": 0, "errors": 0,
                        "reported_total_tokens": 0, "reserved_unreported_tokens": 0}
        with tempfile.TemporaryDirectory() as directory:
            client.path = Path(directory) / "usage.json"
            with patch.dict("sys.modules", {"google.genai": fake_genai}):
                with self.assertRaises(BudgetLimitError):
                    client.rank("hello")  # client.client deliberately does not exist
                client.usage["calls_attempted"] = 0
                client.max_tokens = 10  # 5 UTF-8 bytes + 10 max response tokens exceeds cap
                with self.assertRaises(BudgetLimitError):
                    client.rank("hello")
            self.assertEqual(json.loads(client.path.read_text())["calls_attempted"], 0)


if __name__ == "__main__":
    unittest.main()

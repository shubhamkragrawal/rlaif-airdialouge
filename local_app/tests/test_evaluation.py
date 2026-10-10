import json
import tempfile
import unittest
from pathlib import Path

from local_app.evaluation import (
    EvaluationScenario,
    aggregate_results,
    build_report,
    load_scenarios,
    run_scenario,
    wilson_interval,
    write_report,
)


CORPUS = Path(__file__).resolve().parents[2] / "airdialogue_sampled_5k.jsonl"


class StepClock:
    def __init__(self, step=0.001):
        self.value = 0.0
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


class EvaluationTests(unittest.TestCase):
    def test_load_and_run_book_scenario(self):
        scenario = load_scenarios(CORPUS, count=1)[0]
        self.assertEqual(scenario.scenario_id, 0)
        row = run_scenario(scenario, corpus_path=CORPUS, clock=StepClock())
        self.assertTrue(row["task_success"])
        self.assertTrue(row["completed"])
        self.assertEqual(row["booked_flight_id"], 1027)
        self.assertEqual(row["turn_count"], 3)
        self.assertEqual(row["search_validity"], 1.0)
        self.assertEqual(row["tool_validity"], 1.0)
        self.assertEqual(row["latency_ms"]["first_response"], 2.0)
        self.assertEqual(row["latency_ms"]["completed_action"], 6.0)

    def test_guardrail_rejection_and_recovery_are_measured(self):
        scenario = EvaluationScenario(
            0,
            ("", "Find a flight from IAD to LAS under $100", "My name is Amy Gonzalez",
             "Please book flight 1027"),
            "book",
            (1027,),
        )
        row = run_scenario(scenario, corpus_path=CORPUS, clock=StepClock())
        self.assertEqual(row["rejections"], 1)
        self.assertEqual(row["rejection_rate"], 0.25)
        self.assertTrue(row["recovered_after_rejection"])
        self.assertTrue(row["task_success"])

    def test_unavailable_metrics_are_null(self):
        rows = [{
            "task_success": False, "completed": False, "turn_count": 0,
            "turn_attempts": 0, "search_calls": 0, "search_valid_calls": 0,
            "tool_calls": 0, "tool_valid_calls": 0, "rejections": 0,
            "recovered_after_rejection": None,
            "latency_ms": {"first_response": None, "completed_action": None},
        }]
        summary = aggregate_results(rows)
        self.assertIsNone(summary["search_validity"])
        self.assertIsNone(summary["tool_validity"])
        self.assertIsNone(summary["recovery_rate"])
        self.assertIsNone(summary["mean_latency_ms"]["first_response"])

    def test_report_is_sorted_and_canonical(self):
        scenarios = [
            EvaluationScenario(1, ("Find a flight from SFO to PHL under $1",), "no_flight"),
            EvaluationScenario(0, ("Find a flight from IAD to LAS under $100",
                                   "My name is Amy Gonzalez", "Book flight 1027"),
                               "book", (1027,)),
        ]
        report = build_report(scenarios, corpus_path=CORPUS, clock=StepClock())
        self.assertEqual(report["scenario_ids"], [0, 1])
        self.assertEqual(report["summary"]["scenario_count"], 2)
        with tempfile.TemporaryDirectory() as directory:
            first = write_report(report, Path(directory) / "report.json")
            original = first.read_bytes()
            write_report(report, first)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(json.loads(original), report)
            self.assertTrue(original.endswith(b"\n"))

    def test_wilson_interval(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.236593, places=5)
        self.assertAlmostEqual(high, 0.763407, places=5)


if __name__ == "__main__":
    unittest.main()

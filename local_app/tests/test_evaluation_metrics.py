import unittest

from local_app.evaluation_metrics import aggregate_evaluation_metrics, wilson_interval


class EvaluationMetricsTests(unittest.TestCase):
    def test_scenario_metrics_and_available_per_turn_totals(self):
        result = aggregate_evaluation_metrics([
            {"task_success": True, "turn_count": 2, "search_calls": 1,
             "search_valid_calls": 1, "booking_calls": 1, "booking_valid_calls": 1,
             "tokens": {"input": 12, "output": 5},
             "latency_ms": {"per_turn": [10, 20]}, "peak_process_rss_mb": 30},
            {"task_success": False, "turn_count": 4, "search_calls": 1,
             "search_valid_calls": 0, "booking_calls": 0, "booking_valid_calls": 0,
             "tokens": {"input": 7, "output": 3},
             "latency_ms": {"per_turn": [15]}, "peak_process_rss_mb": 25},
        ])
        self.assertEqual(result["task_completion"]["rate"], 0.5)
        self.assertEqual(result["task_completion"]["wilson_95"], wilson_interval(1, 2))
        self.assertEqual(result["valid_transitions"]["search_rate"], 0.5)
        self.assertEqual(result["valid_transitions"]["booking_rate"], 1.0)
        self.assertEqual(result["turns"], {"average": 3, "p95": 4})
        self.assertEqual(result["tokens"], {"input": 19, "output": 8})
        self.assertEqual(result["latency_ms"]["total"], 22.5)
        self.assertEqual(result["peak_memory_mb"], 30)
        self.assertIsNone(result["latency_ms"]["time_to_first_token"])

    def test_turn_metrics_use_explicit_evidence(self):
        turns = [
            {"scenario_id": "a", "transaction_claim": True,
             "unsupported_transaction_claim": True, "claimed_entities": ["IAD", "Flight 8"],
             "grounded_entities": ["iad"], "expected_no_results": True,
             "presented_no_results": True, "malformed_output": False,
             "response": "No flights found", "validation_reason": "repeated_sentence",
             "fallback": True,
             "fallback_reason": "unsafe_claim", "input_tokens": 4,
             "time_to_first_token_ms": 12, "total_latency_ms": 40,
             "peak_memory_mb": 80},
            {"scenario_id": "a", "transaction_claim": True,
             "unsupported_transaction_claim": False, "claimed_entities": ["IAD"],
             "grounded_entities": ["IAD"], "expected_no_results": False,
             "presented_no_results": True, "malformed_output": True,
             "response": "No flights found", "repetitive_output": False,
             "fallback": False,
             "output_tokens": 6, "time_to_first_token_ms": 20,
             "total_latency_ms": 60, "peak_memory_mb": 70},
        ]
        result = aggregate_evaluation_metrics([], turns, judge_scores=[0.7, 0.9])
        self.assertEqual(result["unsupported_transaction_claim_rate"], 0.5)
        self.assertEqual(result["grounded_entity_precision"], 2 / 3)
        self.assertEqual(result["no_results_presentation_accuracy"], 0.5)
        self.assertEqual(result["malformed_output_rate"], 0.5)
        self.assertEqual(result["repetition_rate"], 0.5)
        self.assertEqual(result["fallback_rate"], 0.5)
        self.assertEqual(result["fallback_reason_distribution"], {"unsafe_claim": 1})
        self.assertEqual(result["latency_ms"]["time_to_first_token"], 16)
        self.assertEqual(result["latency_ms"]["total"], 50)
        self.assertEqual(result["peak_memory_mb"], 80)
        self.assertEqual(result["judge_score"], {"value": 0.8, "basis": "judge-dependent"})

    def test_repetition_uses_validator_evidence_for_single_response(self):
        result = aggregate_evaluation_metrics([], [{
            "response": "Repeat this. Repeat this. Repeat this.",
            "validation_reason": "repeated_sentence",
        }])
        self.assertEqual(result["repetition_rate"], 1.0)

        explicit = aggregate_evaluation_metrics([], [{"repetitive_output": True}])
        self.assertEqual(explicit["repetition_rate"], 1.0)

    def test_repetition_is_unavailable_without_validator_evidence(self):
        result = aggregate_evaluation_metrics([], [{
            "response": "Repeat this. Repeat this. Repeat this.",
        }])
        self.assertIsNone(result["repetition_rate"])

    def test_empty_or_missing_evidence_is_not_fabricated(self):
        result = aggregate_evaluation_metrics([])
        self.assertIsNone(result["task_completion"]["rate"])
        self.assertIsNone(result["task_completion"]["wilson_95"])
        self.assertIsNone(result["unsupported_transaction_claim_rate"])
        self.assertIsNone(result["grounded_entity_precision"])
        self.assertIsNone(result["fallback_rate"])
        self.assertIsNone(result["fallback_reason_distribution"])
        self.assertIsNone(result["tokens"]["input"])
        self.assertIsNone(result["peak_memory_mb"])
        self.assertEqual(result["judge_score"], {"value": None, "basis": "judge-dependent"})


if __name__ == "__main__":
    unittest.main()

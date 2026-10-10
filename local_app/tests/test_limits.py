import unittest
from dataclasses import FrozenInstanceError

from local_app.limits import BudgetSnapshot, ExecutionBudget, ExecutionLimits, LimitReason


class ExecutionLimitsTests(unittest.TestCase):
    def test_limits_are_immutable_and_validate_configuration(self):
        limits = ExecutionLimits(max_turns=2, max_generation_tokens=40)
        with self.assertRaises(FrozenInstanceError):
            limits.max_turns = 3
        for kwargs in (
            {"max_turns": -1},
            {"max_retries": 1.5},
            {"max_tool_calls": True},
            {"max_session_seconds": -0.1},
            {"max_session_seconds": 0},
            {"max_session_seconds": float("nan")},
            {"max_session_seconds": float("inf")},
            {"max_generation_tokens": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExecutionLimits(**kwargs)

    def test_turn_and_generation_limits_are_independent(self):
        budget = ExecutionBudget(ExecutionLimits(max_turns=1, max_generation_tokens=12))
        self.assertEqual(budget.consume_turn(), ())
        self.assertEqual(budget.check_generation_length(12), ())
        self.assertEqual(
            budget.check_generation_length(13), (LimitReason.GENERATION_LENGTH,)
        )
        self.assertEqual(budget.consume_turn(), (LimitReason.TURN_LIMIT,))
        self.assertEqual(budget.snapshot().turns, 1)

    def test_retry_and_tool_call_accounting_rejects_without_increment(self):
        budget = ExecutionBudget(ExecutionLimits(max_retries=1, max_tool_calls=1))
        self.assertEqual(budget.consume_retry(), ())
        self.assertEqual(budget.consume_retry(), (LimitReason.RETRY_LIMIT,))
        self.assertEqual(budget.consume_tool_call(), ())
        self.assertEqual(budget.consume_tool_call(), (LimitReason.TOOL_CALL_LIMIT,))
        self.assertEqual(budget.snapshot().retries, 1)
        self.assertEqual(budget.snapshot().tool_calls, 1)

    def test_injected_monotonic_clock_enforces_total_session_time(self):
        now = [100.0]
        budget = ExecutionBudget(
            ExecutionLimits(max_session_seconds=5), clock=lambda: now[0]
        )
        self.assertEqual(budget.snapshot(), BudgetSnapshot(0, 0, 0, 0.0))
        now[0] += 5
        self.assertEqual(budget.exceeded_reasons(), ())
        now[0] += 0.01
        self.assertEqual(budget.exceeded_reasons(), (LimitReason.SESSION_TIME,))
        self.assertEqual(budget.consume_turn(), (LimitReason.SESSION_TIME,))
        self.assertEqual(budget.snapshot().turns, 0)

    def test_expired_session_reason_is_combined_with_request_specific_reason(self):
        now = [0.0]
        budget = ExecutionBudget(
            ExecutionLimits(max_generation_tokens=3, max_session_seconds=0.1),
            clock=lambda: now[0],
        )
        now[0] = 0.11
        self.assertEqual(
            budget.check_generation_length(4),
            (LimitReason.SESSION_TIME, LimitReason.GENERATION_LENGTH),
        )

    def test_invalid_generation_length_is_rejected(self):
        budget = ExecutionBudget()
        for value in (-1, 1.2, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.check_generation_length(value)


if __name__ == "__main__":
    unittest.main()

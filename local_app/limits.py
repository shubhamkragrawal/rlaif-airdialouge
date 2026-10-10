"""Bounded execution budgets for a local conversation session.

The limits are policy configuration; :class:`ExecutionBudget` is the mutable,
per-session accounting object intended to be shared by a controller/runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from threading import RLock
from time import monotonic
from typing import Callable


class LimitReason(StrEnum):
    """Stable machine-readable reasons an execution budget can be exceeded."""

    TURN_LIMIT = "turn_limit_exceeded"
    GENERATION_LENGTH = "generation_length_exceeded"
    RETRY_LIMIT = "retry_limit_exceeded"
    TOOL_CALL_LIMIT = "tool_call_limit_exceeded"
    SESSION_TIME = "session_time_exceeded"


@dataclass(frozen=True)
class ExecutionLimits:
    """Immutable upper bounds for one session.

    ``max_generation_tokens`` applies to each generation request. The other
    count limits apply cumulatively across the session.
    """

    max_turns: int = 20
    max_generation_tokens: int = 2048
    max_retries: int = 3
    max_tool_calls: int = 20
    max_session_seconds: float = 900.0

    def __post_init__(self) -> None:
        for name in ("max_turns", "max_generation_tokens", "max_retries", "max_tool_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        value = self.max_session_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("max_session_seconds must be a finite positive number")
        if self.max_generation_tokens == 0:
            raise ValueError("max_generation_tokens must be positive")


@dataclass(frozen=True)
class BudgetSnapshot:
    """Immutable view of the current counters and elapsed session time."""

    turns: int
    retries: int
    tool_calls: int
    elapsed_seconds: float


class ExecutionBudget:
    """Thread-safe accounting for one session, with an injectable clock."""

    def __init__(
        self,
        limits: ExecutionLimits | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.limits = limits if limits is not None else ExecutionLimits()
        if not isinstance(self.limits, ExecutionLimits):
            raise TypeError("limits must be an ExecutionLimits instance")
        self._clock = clock
        self._started_at = clock()
        self._turns = 0
        self._retries = 0
        self._tool_calls = 0
        self._lock = RLock()

    def snapshot(self) -> BudgetSnapshot:
        """Return a consistent immutable snapshot of session accounting."""
        with self._lock:
            return BudgetSnapshot(
                turns=self._turns,
                retries=self._retries,
                tool_calls=self._tool_calls,
                elapsed_seconds=max(0.0, self._clock() - self._started_at),
            )

    def remaining_seconds(self) -> float:
        with self._lock:
            return max(0.0, self.limits.max_session_seconds - (self._clock() - self._started_at))

    def exceeded_reasons(self) -> tuple[LimitReason, ...]:
        """Return all currently exceeded cumulative limits in stable order."""
        with self._lock:
            reasons = []
            if self._turns > self.limits.max_turns:
                reasons.append(LimitReason.TURN_LIMIT)
            if self._retries > self.limits.max_retries:
                reasons.append(LimitReason.RETRY_LIMIT)
            if self._tool_calls > self.limits.max_tool_calls:
                reasons.append(LimitReason.TOOL_CALL_LIMIT)
            if self._clock() - self._started_at > self.limits.max_session_seconds:
                reasons.append(LimitReason.SESSION_TIME)
            return tuple(reasons)

    def can_start_turn(self) -> tuple[LimitReason, ...]:
        """Check whether another turn may start without changing accounting."""
        with self._lock:
            reasons = list(self.exceeded_reasons())
            if self._turns >= self.limits.max_turns and LimitReason.TURN_LIMIT not in reasons:
                reasons.append(LimitReason.TURN_LIMIT)
            return tuple(reasons)

    def consume_turn(self) -> tuple[LimitReason, ...]:
        """Account for a turn, or return reasons and leave counts unchanged."""
        with self._lock:
            reasons = self.can_start_turn()
            if reasons:
                return reasons
            self._turns += 1
            return ()

    def check_generation_length(self, token_count: int) -> tuple[LimitReason, ...]:
        """Validate one generation length; generation requests are not cumulative."""
        _validate_count(token_count, "token_count")
        with self._lock:
            reasons = list(self.exceeded_reasons())
            if token_count > self.limits.max_generation_tokens:
                reasons.append(LimitReason.GENERATION_LENGTH)
            return tuple(dict.fromkeys(reasons))

    def consume_retry(self) -> tuple[LimitReason, ...]:
        """Account for a retry, rejecting it if the session budget is exhausted."""
        return self._consume_counter("retries", LimitReason.RETRY_LIMIT)

    def consume_tool_call(self) -> tuple[LimitReason, ...]:
        """Account for a tool call, rejecting it if the session budget is exhausted."""
        return self._consume_counter("tool_calls", LimitReason.TOOL_CALL_LIMIT)

    def _consume_counter(self, counter: str, reason: LimitReason) -> tuple[LimitReason, ...]:
        with self._lock:
            reasons = list(self.exceeded_reasons())
            current = getattr(self, f"_{counter}")
            maximum = getattr(self.limits, f"max_{counter}")
            if current >= maximum and reason not in reasons:
                reasons.append(reason)
            if reasons:
                return tuple(dict.fromkeys(reasons))
            setattr(self, f"_{counter}", current + 1)
            return ()


def _validate_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")

"""Deterministic, dependency-free regression evaluation for the local controller.

The runner deliberately evaluates the controller through its public ``handle``
method.  It does not call a model, a network service, or the simulator directly.
Latency is measured around each controller turn; callers may inject a clock for
fully reproducible tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

from .controller import ADAPTERS, ConversationController
from .guardrails import safe_fallback
from .knowledge import DEFAULT_CORPUS, FlightKnowledgeBase


REPORT_VERSION = 1
SAFE_FALLBACK_PREFIX = safe_fallback("").split("complete", 1)[0]


def peak_process_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return raw / divisor


@dataclass(frozen=True)
class EvaluationScenario:
    """A stable conversation and its expected terminal simulator outcome."""

    scenario_id: int
    messages: tuple[str, ...]
    expected_status: str
    expected_flight_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_status not in {"book", "no_flight"}:
            raise ValueError("expected_status must be 'book' or 'no_flight'")
        if not self.messages:
            raise ValueError("an evaluation scenario must contain at least one message")
        if self.expected_status == "book" and not self.expected_flight_ids:
            raise ValueError("book scenarios require at least one expected flight ID")


def _search_message(intent: dict[str, Any], dialogue: Sequence[str] = ()) -> str:
    origin = intent.get("departure_airport")
    destination = intent.get("return_airport")
    if not origin or not destination:
        raise ValueError("scenario intent is missing departure or destination airport")
    message = f"Find a flight from {origin} to {destination}"
    month_numbers = {
        month: index for index, month in enumerate(
            "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1
        )
    }
    month_numbers.update({
        "January": 1, "February": 2, "March": 3, "April": 4, "June": 6,
        "July": 7, "August": 8, "September": 9, "Sept": 9, "October": 10,
        "November": 11, "December": 12,
    })
    dep_month = month_numbers.get(intent.get("departure_month"))
    dep_day = intent.get("departure_day")
    ret_month = month_numbers.get(intent.get("return_month"))
    ret_day = intent.get("return_day")
    if dep_month and dep_day:
        message += f" on {dep_month:02d}/{int(dep_day):02d}"
    if ret_month and ret_day:
        message += f" returning {ret_month:02d}/{int(ret_day):02d}"
    max_price = intent.get("max_price")
    if isinstance(max_price, int) and max_price > 0:
        message += f" under ${max_price}"
    if intent.get("max_connections") == 0:
        message += " nonstop"
    elif isinstance(intent.get("max_connections"), int) and intent["max_connections"] > 0:
        message += f" with at most {intent['max_connections']} connection"
    customer_text = " ".join(
        turn.split(":", 1)[1] for turn in dialogue
        if turn.lower().startswith("customer:") and ":" in turn
    )
    for bucket in ("morning", "afternoon", "evening", "night"):
        if re.search(rf"\breturn\w*\b[^.]*\b{bucket}\b", customer_text, re.I):
            message += f" and return in the {bucket}"
        if re.search(rf"\b(?:depart\w*|leave)\b[^.]*\b{bucket}\b", customer_text, re.I):
            message += f" and depart in the {bucket}"
    return message


def scenario_from_record(scenario_id: int, record: dict[str, Any]) -> EvaluationScenario:
    """Convert one AirDialogue record into controller-supported deterministic turns."""

    intent = record.get("intent") or {}
    expected = record.get("expected_action") or record.get("action") or {}
    status = expected.get("status")
    flight_ids = tuple(sorted({int(value) for value in expected.get("flight", ())}))
    messages = [_search_message(intent, record.get("dialogue") or ())]
    if status == "book":
        name = expected.get("name") or intent.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("book scenario is missing a passenger name")
        messages.extend((f"My name is {name.strip()}", f"Please book flight {flight_ids[0]}"))
    return EvaluationScenario(scenario_id, tuple(messages), str(status), flight_ids)


def load_scenarios(
    corpus_path: str | Path = DEFAULT_CORPUS,
    *,
    count: int = 50,
    start: int = 0,
) -> list[EvaluationScenario]:
    """Load a contiguous, stable set of eligible scenarios from a JSONL corpus."""

    if count <= 0:
        raise ValueError("count must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    scenarios: list[EvaluationScenario] = []
    with Path(corpus_path).open(encoding="utf-8") as handle:
        for scenario_id, line in enumerate(handle):
            if scenario_id < start:
                continue
            record = json.loads(line)
            expected = record.get("expected_action") or record.get("action") or {}
            if expected.get("status") not in {"book", "no_flight"}:
                continue
            scenarios.append(scenario_from_record(scenario_id, record))
            if len(scenarios) == count:
                break
    if len(scenarios) != count:
        raise ValueError(f"requested {count} scenarios but found {len(scenarios)} eligible records")
    return scenarios


def _flight_matches(flight: dict[str, Any], criteria: Any) -> bool:
    pairs = (
        ("departure_airport", criteria.origin),
        ("return_airport", criteria.destination),
        ("departure_month", criteria.departure_month),
        ("departure_day", criteria.departure_day),
        ("return_month", criteria.return_month),
        ("return_day", criteria.return_day),
        ("airline", criteria.airline),
        ("class", criteria.cabin_class),
    )
    def same(field, actual, expected):
        if field.endswith("_month"):
            return str(actual).lower()[:3] == str(expected).lower()[:3]
        return str(actual).lower() == str(expected).lower()

    if any(value is not None and not same(field, flight.get(field), value)
           for field, value in pairs):
        return False
    if criteria.max_price is not None and int(flight["price"]) > criteria.max_price:
        return False
    if criteria.max_connections is not None and int(flight["num_connections"]) > criteria.max_connections:
        return False
    buckets = {
        "morning": range(5, 12), "afternoon": range(12, 20),
        "evening": range(18, 24), "night": range(0, 5),
    }
    if criteria.departure_time and int(flight["departure_time_num"]) not in buckets[criteria.departure_time]:
        return False
    if criteria.return_time and int(flight["return_time_num"]) not in buckets[criteria.return_time]:
        return False
    return True


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _safe_count(value: Any) -> int:
    """Keep malformed model metadata from breaking evaluation aggregation."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a 95% Wilson score interval, or ``None`` for an empty sample."""

    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def run_scenario(
    scenario: EvaluationScenario,
    *,
    adapter: str = "sft",
    corpus_path: str | Path = DEFAULT_CORPUS,
    clock: Callable[[], float] = time.perf_counter,
    model_loader=None,
) -> dict[str, Any]:
    """Run and score one scenario without exposing prompts or random tool IDs."""

    kb = FlightKnowledgeBase(corpus_path, scenario.scenario_id)
    controller = ConversationController(kb, model_loader=model_loader)
    state = controller.new_session(adapter)
    turn_latencies: list[float] = []
    searches = search_valid = bookings = booking_valid = 0
    rejections = 0
    fallback_reasons: list[str] = []
    had_rejection = False
    recovered = False
    error_types: list[str] = []
    model_turns = model_fallbacks = input_tokens = output_tokens = 0
    started = clock()
    first_response_ms: float | None = None
    completion_ms: float | None = None

    for message in scenario.messages:
        turn_started = clock()
        try:
            result = controller.handle(state, message)
        except (TypeError, ValueError) as exc:
            turn_ended = clock()
            turn_latencies.append((turn_ended - turn_started) * 1000)
            rejections += 1
            had_rejection = True
            error_types.append(type(exc).__name__)
            continue
        turn_ended = clock()
        turn_latencies.append((turn_ended - turn_started) * 1000)
        if first_response_ms is None:
            first_response_ms = (turn_ended - started) * 1000
        if result.get("model_used"):
            model_turns += 1
            usage = result.get("usage")
            if isinstance(usage, Mapping):
                input_tokens += _safe_count(usage.get("input_tokens"))
                output_tokens += _safe_count(usage.get("output_tokens"))
        elif model_loader is not None:
            model_fallbacks += 1

        reason = result.get("fallback_reason") or result.get("validation_reason")
        if isinstance(reason, str) and reason and reason not in fallback_reasons:
            fallback_reasons.append(reason)
        fallback = bool(reason) or str(result.get("text", "")).startswith(SAFE_FALLBACK_PREFIX)
        if fallback:
            rejections += 1
            had_rejection = True
        elif had_rejection:
            recovered = True

        criteria = controller._criteria_from_text(message)
        is_search = any(value is not None for value in criteria.__dict__.values())
        if is_search:
            searches += 1
            results = result.get("results", ())
            known_ids = {int(flight["flight_number"]) for flight in kb.all()}
            if all(int(flight.get("flight_number", -1)) in known_ids
                   and _flight_matches(flight, criteria) for flight in results):
                search_valid += 1

        if state.booking is not None and bookings == 0:
            bookings = 1
            flight_id = int(state.booking["flight"]["flight_number"])
            if kb.get(flight_id) is not None:
                booking_valid = 1
            completion_ms = (turn_ended - started) * 1000
        elif scenario.expected_status == "no_flight" and is_search and not result.get("results"):
            completion_ms = (turn_ended - started) * 1000

    booked_id = (int(state.booking["flight"]["flight_number"])
                 if state.booking is not None else None)
    if scenario.expected_status == "book":
        task_success = booked_id in scenario.expected_flight_ids
    else:
        task_success = bool(state.last_search is not None and not state.last_search.flights)
    completed = completion_ms is not None
    attempts = len(scenario.messages)
    valid_tools = search_valid + booking_valid
    tool_calls = searches + bookings
    return {
        "scenario_id": scenario.scenario_id,
        "adapter": adapter,
        "expected_status": scenario.expected_status,
        "expected_flight_ids": list(scenario.expected_flight_ids),
        "booked_flight_id": booked_id,
        "task_success": task_success,
        "completed": completed,
        "turn_count": state.turn_count,
        "turn_attempts": attempts,
        "search_calls": searches,
        "search_valid_calls": search_valid,
        "search_validity": _rate(search_valid, searches),
        "booking_calls": bookings,
        "booking_valid_calls": booking_valid,
        "tool_calls": tool_calls,
        "tool_valid_calls": valid_tools,
        "tool_validity": _rate(valid_tools, tool_calls),
        "rejections": rejections,
        "fallback_reasons": fallback_reasons,
        "rejection_rate": _rate(rejections, attempts),
        "recovered_after_rejection": recovered if had_rejection else None,
        "latency_ms": {
            "first_response": first_response_ms,
            "completed_action": completion_ms,
            "per_turn": turn_latencies,
        },
        "error_types": error_types,
        "model_turns": model_turns,
        "model_fallbacks": model_fallbacks,
        "tokens": {"input": input_tokens, "output": output_tokens},
        "peak_process_rss_mb": peak_process_rss_mb(),
    }


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scenario rows with explicit unavailable (``None``) rates."""

    total = len(results)
    successes = sum(bool(row["task_success"]) for row in results)
    completed = sum(bool(row["completed"]) for row in results)
    searches = sum(int(row["search_calls"]) for row in results)
    search_valid = sum(int(row["search_valid_calls"]) for row in results)
    tools = sum(int(row["tool_calls"]) for row in results)
    tool_valid = sum(int(row["tool_valid_calls"]) for row in results)
    attempts = sum(int(row["turn_attempts"]) for row in results)
    rejections = sum(int(row["rejections"]) for row in results)
    rejected_rows = [row for row in results if row["rejections"]]
    recovered = sum(row["recovered_after_rejection"] is True for row in rejected_rows)
    first = [row["latency_ms"]["first_response"] for row in results
             if row["latency_ms"]["first_response"] is not None]
    action = [row["latency_ms"]["completed_action"] for row in results
              if row["latency_ms"]["completed_action"] is not None]
    turns = [int(row["turn_count"]) for row in results]
    model_turns = sum(int(row.get("model_turns", 0)) for row in results)
    model_fallbacks = sum(int(row.get("model_fallbacks", 0)) for row in results)
    input_tokens = sum(_safe_count((row.get("tokens") or {}).get("input"))
                       for row in results if isinstance(row.get("tokens") or {}, dict))
    output_tokens = sum(_safe_count((row.get("tokens") or {}).get("output"))
                        for row in results if isinstance(row.get("tokens") or {}, dict))
    rss_values = []
    for row in results:
        value = row.get("peak_process_rss_mb", 0)
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed) and parsed >= 0:
            rss_values.append(parsed)
    peak_rss = max(rss_values, default=0.0)
    return {
        "scenario_count": total,
        "task_successes": successes,
        "task_success_rate": _rate(successes, total),
        "task_success_wilson_95": wilson_interval(successes, total),
        "completed_scenarios": completed,
        "completion_rate": _rate(completed, total),
        "search_calls": searches,
        "search_validity": _rate(search_valid, searches),
        "tool_calls": tools,
        "tool_validity": _rate(tool_valid, tools),
        "mean_turn_count": mean(turns) if turns else None,
        "rejections": rejections,
        "rejection_rate": _rate(rejections, attempts),
        "recovery_rate": _rate(recovered, len(rejected_rows)),
        "mean_latency_ms": {
            "first_response": mean(first) if first else None,
            "completed_action": mean(action) if action else None,
        },
        "model_turns": model_turns,
        "model_fallbacks": model_fallbacks,
        "tokens": {"input": input_tokens, "output": output_tokens},
        "peak_process_rss_mb": peak_rss,
    }


def build_report(
    scenarios: Iterable[EvaluationScenario],
    *,
    adapter: str = "sft",
    corpus_path: str | Path = DEFAULT_CORPUS,
    clock: Callable[[], float] = time.perf_counter,
    model_loader=None,
) -> dict[str, Any]:
    scenario_list = sorted(scenarios, key=lambda item: item.scenario_id)
    if len({item.scenario_id for item in scenario_list}) != len(scenario_list):
        raise ValueError("scenario IDs must be unique")
    results = [run_scenario(item, adapter=adapter, corpus_path=corpus_path, clock=clock,
                            model_loader=model_loader)
               for item in scenario_list]
    return {
        "report_version": REPORT_VERSION,
        "adapter": adapter,
        "corpus": Path(corpus_path).name,
        "scenario_ids": [item.scenario_id for item in scenario_list],
        "summary": aggregate_results(results),
        "scenarios": results,
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically write canonical JSON (stable keys, ordering, and whitespace)."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--adapter", choices=ADAPTERS, default="sft")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="store_true",
                        help="run the selected local Qwen/PEFT adapter")
    args = parser.parse_args(argv)
    scenarios = load_scenarios(args.corpus, count=args.count, start=args.start)
    manager = None
    if args.model:
        from .runtime import RuntimeManager
        manager = RuntimeManager()
    try:
        report = build_report(
            scenarios, adapter=args.adapter, corpus_path=args.corpus,
            model_loader=manager.get if manager else None,
        )
    finally:
        if manager:
            manager.release()
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

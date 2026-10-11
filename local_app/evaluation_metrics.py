"""Evidence-aware aggregation of scenario and turn evaluation records.

Scenario records use the fields emitted by :mod:`local_app.evaluation`. Optional
turn records enable presentation and response-quality measures. Missing evidence
stays ``None``; counts and rates with a known empty denominator are zero and
``None`` respectively.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """95% Wilson score interval; no trials has no defined interval."""
    if total <= 0:
        return None
    p = successes / total
    z2 = z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return max(0.0, center - radius), min(1.0, center + radius)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank percentile (rank ceil(.95*n)).
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _entity_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def aggregate_evaluation_metrics(
    scenarios: Sequence[Mapping[str, Any]],
    turns: Sequence[Mapping[str, Any]] | None = None,
    *,
    judge_scores: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Aggregate scenario rows and optional per-turn evidence.

    Turn fields are explicit booleans/values: ``valid_search_transition`` and
    ``valid_booking_transition``; ``transaction_claim`` and
    ``unsupported_transaction_claim``; ``claimed_entities`` and
    ``grounded_entities``; ``expected_no_results`` and
    ``presented_no_results``; ``malformed_output``; ``response``;
    ``repetitive_output`` or ``validation_reason`` (where
    ``repeated_sentence`` is a positive validator result);
    ``fallback``/``fallback_reason``; token counts, ``time_to_first_token_ms``,
    ``total_latency_ms``, and ``peak_memory_mb``. Scenario IDs associate repeated
    responses; absent IDs share no repetition group. A fallback reason itself
    indicates a fallback. Unsupported-claim rate is over turns with an explicit
    transaction claim. Entity precision is micro precision over supplied claims.
    """
    turns = list(turns or [])
    scenario_count = len(scenarios)
    successes = sum(row.get("task_success") is True for row in scenarios)
    search_calls = sum(max(0, int(_number(row.get("search_calls")) or 0)) for row in scenarios)
    valid_searches = sum(max(0, int(_number(row.get("search_valid_calls")) or 0)) for row in scenarios)
    booking_calls = sum(max(0, int(_number(row.get("booking_calls")) or 0)) for row in scenarios)
    valid_bookings = sum(max(0, int(_number(row.get("booking_valid_calls")) or 0)) for row in scenarios)

    turn_counts = [_number(row.get("turn_count")) for row in scenarios]
    turn_counts = [x for x in turn_counts if x is not None]
    input_tokens = [_number((row.get("tokens") or {}).get("input")) for row in scenarios]
    output_tokens = [_number((row.get("tokens") or {}).get("output")) for row in scenarios]
    input_tokens = [x for x in input_tokens if x is not None]
    output_tokens = [x for x in output_tokens if x is not None]
    first_token, total_latency, memory = [], [], []
    for row in scenarios:
        latency = row.get("latency_ms") or {}
        if isinstance(latency, Mapping):
            value = _number(latency.get("time_to_first_token", latency.get("first_token")))
            if value is not None: first_token.append(value)
            value = _number(latency.get("total"))
            if value is None and isinstance(latency.get("per_turn"), Sequence):
                samples = [_number(x) for x in latency["per_turn"]]
                if samples and all(x is not None for x in samples): value = sum(samples)
            if value is not None: total_latency.append(value)
        value = _number(row.get("peak_memory_mb", row.get("peak_process_rss_mb")))
        if value is not None: memory.append(value)
    turn_input_tokens, turn_output_tokens = [], []
    for turn in turns:
        for field, target in (("time_to_first_token_ms", first_token), ("total_latency_ms", total_latency),
                              ("peak_memory_mb", memory)):
            value = _number(turn.get(field))
            if value is not None: target.append(value)
        for field, target in (("input_tokens", turn_input_tokens), ("output_tokens", turn_output_tokens)):
            value = _number(turn.get(field))
            if value is not None: target.append(value)
    if not input_tokens: input_tokens = turn_input_tokens
    if not output_tokens: output_tokens = turn_output_tokens

    fallback_turns = [t for t in turns if bool(t.get("fallback")) or bool(t.get("fallback_reason"))]
    fallback_reasons = Counter(str(t["fallback_reason"]) for t in fallback_turns if t.get("fallback_reason"))
    unsupported_denominator = [t for t in turns if t.get("transaction_claim") is True]
    unsupported = sum(t.get("unsupported_transaction_claim") is True for t in unsupported_denominator)
    entity_claims = entity_true = 0
    for turn in turns:
        claimed, grounded = turn.get("claimed_entities"), turn.get("grounded_entities")
        if isinstance(claimed, Sequence) and not isinstance(claimed, (str, bytes)) and isinstance(grounded, Sequence) and not isinstance(grounded, (str, bytes)):
            grounded_set = {_entity_key(x) for x in grounded}
            entity_claims += len(claimed)
            entity_true += sum(_entity_key(x) in grounded_set for x in claimed)
    no_result_records = [t for t in turns if isinstance(t.get("expected_no_results"), bool) and isinstance(t.get("presented_no_results"), bool)]
    no_result_correct = sum(t["expected_no_results"] == t["presented_no_results"] for t in no_result_records)
    malformed_records = [t for t in turns if isinstance(t.get("malformed_output"), bool)]
    malformed = sum(t["malformed_output"] for t in malformed_records)

    repetition_records = [
        turn for turn in turns
        if isinstance(turn.get("repetitive_output"), bool)
        or isinstance(turn.get("validation_reason"), str)
    ]
    repeated_outputs = sum(
        turn.get("repetitive_output") is True
        or turn.get("validation_reason") == "repeated_sentence"
        for turn in repetition_records
    )
    transition_records = [t for t in turns if "valid_search_transition" in t or "valid_booking_transition" in t]
    valid_transitions = sum(t.get("valid_search_transition") is True for t in transition_records) + sum(t.get("valid_booking_transition") is True for t in transition_records)
    transition_count = sum(isinstance(t.get("valid_search_transition"), bool) for t in transition_records) + sum(isinstance(t.get("valid_booking_transition"), bool) for t in transition_records)

    scored = [_number(x) for x in (judge_scores or [])]
    scored = [x for x in scored if x is not None]
    return {
        "scenario_count": scenario_count,
        "task_completion": {"successes": successes, "rate": _rate(successes, scenario_count),
                            "wilson_95": wilson_interval(successes, scenario_count)},
        "valid_transitions": {"valid": valid_transitions, "observed": transition_count,
                              "rate": _rate(valid_transitions, transition_count),
                              "search_rate": _rate(valid_searches, search_calls),
                              "booking_rate": _rate(valid_bookings, booking_calls)},
        "unsupported_transaction_claim_rate": _rate(unsupported, len(unsupported_denominator)),
        "grounded_entity_precision": _rate(entity_true, entity_claims),
        "no_results_presentation_accuracy": _rate(no_result_correct, len(no_result_records)),
        "malformed_output_rate": _rate(malformed, len(malformed_records)),
        "repetition_rate": _rate(repeated_outputs, len(repetition_records)),
        "fallback_rate": _rate(len(fallback_turns), len(turns)) if turns else None,
        "fallback_reason_distribution": dict(sorted(fallback_reasons.items())) if turns else None,
        "turns": {"average": mean(turn_counts) if turn_counts else None, "p95": _p95(turn_counts)},
        "tokens": {"input": sum(input_tokens) if input_tokens else None,
                   "output": sum(output_tokens) if output_tokens else None},
        "latency_ms": {"time_to_first_token": mean(first_token) if first_token else None,
                       "total": mean(total_latency) if total_latency else None},
        "peak_memory_mb": max(memory) if memory else None,
        "judge_score": {"value": mean(scored) if scored else None, "basis": "judge-dependent"},
    }

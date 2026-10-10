"""Independent evaluation checks. No model, simulator, or reward judge dependencies."""
import hashlib
import json
import math
import random
import re
from collections import Counter

from rlaif.shared.prompts import CONFIRM_MARKER, FAIL_MARKER


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matching_flights(record):
    """Flights satisfying every structured intent field present in this corpus.

    -1 max_connections means unrestricted. Unsupported extra constraint fields fail
    closed: an evaluator must not silently ignore a new constraint.
    """
    intent = record["intent"]
    supported = {"goal", "name", "departure_airport", "return_airport", "departure_month",
                 "return_month", "departure_day", "return_day", "max_price", "max_connections"}
    unknown = set(intent) - supported
    if unknown:
        raise ValueError("Unsupported intent fields: %s" % sorted(unknown))
    if intent["goal"] != "book":
        raise ValueError("Reservation changes/cancellations cannot be evaluated from this KB")
    matches = []
    for flight in record["kb"]:
        if any(str(flight[k]).casefold() != str(intent[k]).casefold()
               for k in ("departure_airport", "return_airport", "departure_month", "return_month")):
            continue
        if any(int(flight[k]) != int(intent[k]) for k in ("departure_day", "return_day")):
            continue
        if float(flight["price"]) > float(intent["max_price"]):
            continue
        cap = int(intent["max_connections"])
        if cap >= 0 and int(flight["num_connections"]) > cap:
            continue
        matches.append(int(flight["flight_number"]))
    return sorted(matches)


def make_manifest(dialogues, splits, count=50, seed=20260922):
    eligible, excluded = [], []
    by_id = {r["dialogue_id"]: r for r in dialogues}
    for dialogue_id in splits["held_out"]:
        record = by_id[dialogue_id]
        reason = None
        if record["intent"].get("goal") != "book":
            reason = "reservation_state_unavailable"
        else:
            actual = matching_flights(record)
            action = record["expected_action"]
            labelled = sorted(int(f) for f in action.get("flight", []))
            status = "book" if actual else "no_flight"
            if actual != labelled or action["status"] != status:
                reason = "structured_intent_does_not_recover_label"
        if reason:
            excluded.append({"dialogue_id": dialogue_id, "reason": reason})
        else:
            eligible.append(dialogue_id)
    if count > len(eligible) or count < 1:
        raise ValueError("Requested %d scenarios; eligible pool has %d" % (count, len(eligible)))
    ids = random.Random(seed).sample(sorted(eligible), count)
    manifest = {"schema_version": 1, "selection_seed": seed, "source_sha256": splits["source_sha256"],
                "scope": "booking_only_structured_intent_agrees_with_label",
                "eligible_count": len(eligible), "heldout_count": len(splits["held_out"]),
                "exclusion_counts": dict(Counter(x["reason"] for x in excluded)), "excluded": excluded,
                "scenarios": [{"dialogue_id": i, "line_hash": by_id[i]["line_hash"],
                               "matching_flights": matching_flights(by_id[i])} for i in ids]}
    manifest["manifest_sha256"] = fingerprint(manifest)
    return manifest


def named_flights(text):
    """Only explicit flight-ID mentions; not a general factual hallucination detector."""
    found = re.findall(r"\bflight\s*(?:number|no\.?|#)?\s*[:#]?\s*(\d+)\b", text, re.I)
    found += re.findall(r"\bBOOKING_CONFIRMED\s+(\d+)\b", text)
    return {int(x) for x in found}


def parse_marker(text):
    occurrences = re.findall(r"\b(?:BOOKING_CONFIRMED|BOOKING_FAILED)\b", text)
    if not occurrences:
        return None, None
    if len(occurrences) != 1:
        return "invalid", None
    last = text.strip().splitlines()[-1].strip()
    if last == FAIL_MARKER:
        return "failed", None
    match = re.fullmatch(re.escape(CONFIRM_MARKER) + r"\s+(\d+)", last)
    if match:
        return "confirmed", int(match.group(1))
    return "invalid", None


def check_outcome(record, state):
    transcript = state["transcript"]
    assistant = [t for t in transcript if t["role"] == "assistant"]
    expected = matching_flights(record)
    marker, flight = parse_marker(assistant[-1]["content"]) if assistant else (None, None)
    previous_markers = any(parse_marker(t["content"])[0] is not None for t in assistant[:-1])
    known = {int(f["flight_number"]) for f in record["kb"]}
    invented = sorted(set().union(*(named_flights(t["content"]) for t in assistant)) - known)
    name = " ".join(record["intent"]["name"].casefold().split())
    name_given = any(name in " ".join(t["content"].casefold().split())
                     for t in transcript if t["role"] == "user")
    consent = flight is not None and flight in state.get("accepted_flights", [])
    valid_marker = not previous_markers and marker in ("confirmed", "failed")
    booked = valid_marker and marker == "confirmed" and flight in expected and name_given and consent
    declined = valid_marker and marker == "failed" and not expected
    return {"task_success": bool(booked or declined), "booking_success": bool(booked),
            "correct_no_flight": bool(declined), "requires_booking": bool(expected),
            "marker": marker, "confirmed_flight": flight, "valid_terminal_marker": valid_marker,
            "name_given": name_given, "consent_given": consent,
            "unknown_flight_ids": invented, "unknown_flight_id_hallucination": bool(invented),
            "agent_turns": len(assistant), "matching_flights": expected}


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [mid - half, mid + half]


def aggregate(results, planned_count):
    complete = [r for r in results if r.get("status") == "complete"]
    n = len(complete)
    success = sum(r["metrics"]["task_success"] for r in complete)
    booking = [r for r in complete if r["metrics"]["requires_booking"]]
    scores = [r["judge_score"] for r in complete if r.get("judge_score") is not None]
    return {"planned_scenarios": planned_count, "completed_scenarios": n,
            "incomplete_scenarios": planned_count - n,
            "task_success_count": success, "task_success_rate": success / n if n else None,
            "task_success_wilson95": wilson(success, n),
            "booking_scenarios": len(booking),
            "booking_success_rate": sum(r["metrics"]["booking_success"] for r in booking) / len(booking) if booking else None,
            "unknown_flight_id_hallucination_rate": sum(r["metrics"]["unknown_flight_id_hallucination"] for r in complete) / n if n else None,
            "mean_agent_turns": sum(r["metrics"]["agent_turns"] for r in complete) / n if n else None,
            "mean_agent_turns_to_success": sum(r["metrics"]["agent_turns"] for r in complete if r["metrics"]["task_success"]) / success if success else None,
            "judge_scored_scenarios": len(scores), "mean_judge_score": sum(scores) / len(scores) if scores else None,
            "provisional": n != planned_count}


def paired_comparison(left, right, samples=10000, seed=20260922):
    a = {r["dialogue_id"]: int(r["metrics"]["task_success"]) for r in left if r.get("status") == "complete"}
    b = {r["dialogue_id"]: int(r["metrics"]["task_success"]) for r in right if r.get("status") == "complete"}
    if not a or set(a) != set(b):
        raise ValueError("Paired comparison requires the same nonempty completed scenario IDs")
    deltas = [b[i] - a[i] for i in sorted(a)]
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
    return {"n": len(deltas), "right_minus_left_success_rate": sum(deltas) / len(deltas),
            "paired_bootstrap95": [bootstrap[int(.025 * samples)], bootstrap[min(samples - 1, int(.975 * samples))]],
            "right_only_successes": sum(x == 1 for x in deltas), "left_only_successes": sum(x == -1 for x in deltas),
            "bootstrap_samples": samples, "bootstrap_seed": seed}

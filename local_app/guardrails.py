"""Fail-closed checks around transactional actions and grounded responses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# The local corpus uses four-digit flight IDs. Keeping prices and stop counts out of
# this detector prevents valid grounded summaries such as "$100, 1 stop" from failing.
FLIGHT_ID_RE = re.compile(r"\b\d{4,5}\b")
MAX_MESSAGE_LENGTH = 4000

# Covers direct wording and common conversational paraphrases of a completed
# transaction. This is deliberately deterministic so it can guard model output.
TRANSACTION_CLAIM_RE = re.compile(
    r"\b(?:"
    r"(?:booked|reserved|confirmed|ticketed|secured|issued|completed)\b"
    r"|(?:i\s+(?:have|had)|i['’]ve|we\s+have|we['’]ve)\s+arranged\s+(?:your\s+)?(?:trip|travel|flight|journey|itinerary)\b"
    r"|arranged\s+(?:your\s+)?(?:trip|travel|flight|journey|itinerary)\b"
    r"|(?:your\s+)?(?:trip|travel|flight|journey|itinerary)\s+(?:(?:has|is|was)\s+)?arranged\b"
    r"|(?:e[- ]?ticket|booking|reservation|ticket|itinerary)\s+(?:(?:is|was)\s+(?:now\s+|already\s+)?|(?:has|have)\s+(?:already\s+)?been\s+)?(?:successfully\s+)?(?:issued|sent|generated|made|created|complete(?:d)?|done|processed|confirmed|secured|booked|reserved|ticketed|successful)\b"
    r"|completed\s+(?:your\s+)?(?:booking|reservation|trip|flight)\b"
    r"|(?:your\s+)?(?:flight|trip|seat|ticket)\s+(?:(?:is|was|has\s+been|have\s+been)\s+)?(?:successfully\s+)?(?:locked\s+in|(?:all\s+)?set|secured|confirmed|booked|reserved|ticketed)\b"
    r"|you(?:'re| are|’re)\s+all\s+set(?:\s+for\s+(?:your\s+)?(?:trip|flight|journey))?|you(?:'re| are|’re)\s+good\s+to\s+go"
    r")\b",
    re.IGNORECASE,
)


class TransactionClaimError(ValueError):
    """A transaction claim failed validation with a stable machine reason."""

    reason = "unsupported_transaction_claim"

    def __init__(self) -> None:
        super().__init__(self.reason)


def detect_transaction_claim(text: str) -> bool:
    """Return whether text asserts or strongly implies a completed transaction."""
    if not isinstance(text, str):
        return False
    normalized = text.translate(str.maketrans({
        "’": "'", "‘": "'", "ʼ": "'", "–": "-", "—": "-", "−": "-",
    }))
    for match in TRANSACTION_CLAIM_RE.finditer(normalized):
        clause_start = max(normalized.rfind(mark, 0, match.start()) for mark in ".!?;") + 1
        clause_end_candidates = [normalized.find(mark, match.end()) for mark in ".!?;" if normalized.find(mark, match.end()) >= 0]
        clause_end = min(clause_end_candidates) if clause_end_candidates else len(normalized)
        prefix = normalized[clause_start:match.start()]
        clause = normalized[clause_start:clause_end].strip()
        next_mark = normalized[clause_end:clause_end + 1]
        # Negation only scopes within the local coordinated clause and must
        # syntactically govern the matched predicate. In particular, "not
        # charged ... and flight is booked" still contains an affirmative claim.
        local_prefix = re.split(r"\b(?:and|but|however)\b", prefix, flags=re.I)[-1]
        if re.search(r"\b(?:not|never)\s+(?:(?:yet|been|actually)\s+)?$|\b(?:isn't|isnt|wasn't|wasnt|hasn't|hasnt|haven't|havent|didn't|didnt|don't|doesn't)\s*$", local_prefix, re.I):
            continue
        # Exempt only direct questions whose matched transaction predicate is
        # itself queried ("Is your booking confirmed?", "Have I booked?"). A
        # leading question may still contain an asserted relative clause,
        # e.g. "Would you like the flight I booked for you?".
        direct_question_prefix = re.fullmatch(
            r"\s*(?:is|are|was|were|has|have|had|do|does|did|can|could|would)\s+"
            r"(?:(?:i|you|we|they|he|she|it|your|the|this|that)\s+)?\s*",
            prefix, re.I,
        )
        if (next_mark == "?" and direct_question_prefix
                and re.match(r"^(?:is|are|was|were|has|have|had|do|does|did|can|could|would)\b", clause, re.I)):
            continue
        if re.search(r"\b(?:can|could|may|might|will|would|should)\s+be\s*$", prefix, re.I):
            continue
        return True
    return False


def validate_transaction_claim(
    text: str,
    booking_evidence: dict[str, Any] | None = None,
) -> None:
    """Require authoritative booking evidence and simulator disclosure for claims.

    Evidence must explicitly identify itself as authoritative and include a
    booking ID. Simulator-originated evidence must also disclose ``simulated``.
    """
    if not isinstance(text, str):
        raise TransactionClaimError()
    if not detect_transaction_claim(text):
        return
    hold = booking_evidence.get("hold") if isinstance(booking_evidence, Mapping) else None
    evidence_is_authoritative = (
        isinstance(booking_evidence, Mapping)
        and isinstance(booking_evidence.get("booking_id"), str)
        and bool(booking_evidence["booking_id"].strip())
        and isinstance(booking_evidence.get("hold_id"), str)
        and bool(booking_evidence["hold_id"].strip())
        and booking_evidence.get("status") == "confirmed"
        and booking_evidence.get("simulated") is True
        and isinstance(booking_evidence.get("flight_id"), int)
        and isinstance(booking_evidence.get("passenger"), str)
        and bool(booking_evidence["passenger"].strip())
        and isinstance(hold, Mapping)
        and hold.get("status") == "held" and hold.get("simulated") is True
        and hold.get("hold_id") == booking_evidence.get("hold_id")
        and hold.get("flight", {}).get("flight_number") == booking_evidence.get("flight_id")
        and hold.get("flight") == booking_evidence.get("flight")
        and hold.get("passenger") == booking_evidence.get("passenger")
    )
    if not evidence_is_authoritative:
        raise TransactionClaimError()
    if not re.search(r"\bsimulat\w*\b", text, re.I):
        raise TransactionClaimError()


def validate_input(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"message exceeds {MAX_MESSAGE_LENGTH} characters")
    return message.strip()


def validate_action(action: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    if not isinstance(action, dict) or action.get("type") not in allowed:
        raise ValueError("unsupported or malformed action")
    if action["type"] in {"hold_flight", "get_flight"}:
        if not isinstance(action.get("flight_id"), int):
            raise ValueError("flight_id must be an integer")
    if action["type"] == "hold_flight" and not isinstance(action.get("passenger"), str):
        raise ValueError("passenger is required for a hold")
    return action


def validate_grounding(text: str, retrieved_flights: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    known_ids = {str(f["flight_number"]) for f in retrieved_flights}
    mentioned = set(FLIGHT_ID_RE.findall(text))
    unknown = mentioned - known_ids
    if unknown:
        raise ValueError("response mentions flight IDs outside the retrieved context")


def safe_fallback(reason: str = "that request") -> str:
    return f"I couldn't safely complete {reason}. Please restate the flight details."

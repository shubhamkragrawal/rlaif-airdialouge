"""Fail-closed checks around transactional actions and grounded responses."""

from __future__ import annotations

import re
from typing import Any

# The local corpus uses four-digit flight IDs. Keeping prices and stop counts out of
# this detector prevents valid grounded summaries such as "$100, 1 stop" from failing.
FLIGHT_ID_RE = re.compile(r"\b\d{4,5}\b")
MAX_MESSAGE_LENGTH = 4000


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

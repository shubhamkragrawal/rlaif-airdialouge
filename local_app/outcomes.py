"""Immutable typed outcomes at the local controller boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .tools import BookingResult, FlightSearchResult, HoldResult

OUTCOME_SCHEMA = "rlaif.local_app.turn_outcome"
OUTCOME_SCHEMA_VERSION = 1


class Intent(StrEnum):
    SEARCH = "search"
    HOLD = "hold"
    BOOK = "book"
    CANCEL = "cancel"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    OTHER = "other"


class FrozenMapping(Mapping):
    """Recursively immutable, mapping-compatible snapshot of simulator evidence."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping):
        object.__setattr__(self, "_data", MappingProxyType({
            key: _freeze(item_value) for key, item_value in value.items()
        }))

    def __setattr__(self, name, value):
        raise TypeError("FrozenMapping is immutable")

    def __delattr__(self, name):
        raise TypeError("FrozenMapping is immutable")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def _freeze(value: Any):
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class AuthoritativeClaims:
    """Facts that can be stated only when backed by the controller's tool result."""

    selected_flight_id: int | None = None
    hold_id: str | None = None
    booking_id: str | None = None
    booking_status: str | None = None
    passenger: str | None = None
    simulated: bool | None = None


@dataclass(frozen=True)
class TurnOutcome:
    schema: str
    schema_version: int
    intent: Intent
    deterministic_response: str
    search_results: FlightSearchResult | None = None
    hold_result: Mapping[str, Any] | None = None
    booking_result: Mapping[str, Any] | None = None
    authoritative_claims: AuthoritativeClaims = AuthoritativeClaims()
    fallback_reason: str | None = None
    simulator_disclosure_required: bool = False

    def __post_init__(self) -> None:
        if self.schema != OUTCOME_SCHEMA:
            raise ValueError(f"schema must be {OUTCOME_SCHEMA!r}")
        if self.schema_version != OUTCOME_SCHEMA_VERSION:
            raise ValueError(f"unsupported outcome schema version: {self.schema_version}")
        if self.hold_result is not None and self.intent not in (Intent.HOLD, Intent.BOOK):
            raise ValueError("hold_result requires hold or book intent")
        if self.booking_result is not None and self.intent is not Intent.BOOK:
            raise ValueError("booking_result requires book intent")
        if (self.hold_result is not None or self.booking_result is not None) and not (
            self.simulator_disclosure_required
        ):
            raise ValueError("transaction outcomes require simulator disclosure")
        if self.search_results is not None:
            frozen_search = FlightSearchResult(
                tuple(_freeze(flight) for flight in self.search_results.flights),
                self.search_results.criteria,
            )
            object.__setattr__(self, "search_results", frozen_search)
        if self.booking_result is not None:
            booking = _freeze(self.booking_result)
            hold = _freeze(self.hold_result)
            validate_booking_evidence(booking, hold)
            expected = AuthoritativeClaims(
                selected_flight_id=booking["flight"]["flight_number"],
                hold_id=booking["hold_id"], booking_id=booking["booking_id"],
                booking_status="confirmed", passenger=booking["passenger"], simulated=True,
            )
            if self.authoritative_claims != expected:
                raise ValueError("authoritative claims do not match booking evidence")
            object.__setattr__(self, "booking_result", booking)
            object.__setattr__(self, "hold_result", hold)
        elif self.hold_result is not None:
            hold = _freeze(self.hold_result)
            validate_hold_result(hold)
            expected = AuthoritativeClaims(
                selected_flight_id=hold["flight"]["flight_number"],
                hold_id=hold["hold_id"], passenger=hold["passenger"], simulated=True,
            )
            if self.authoritative_claims != expected:
                raise ValueError("authoritative claims do not match hold evidence")
            object.__setattr__(self, "hold_result", hold)


def validate_hold_result(hold: HoldResult | Mapping[str, Any]) -> None:
    if not isinstance(hold, Mapping) or set(hold) != {"hold_id", "flight", "passenger", "simulated", "status"}:
        raise ValueError("malformed hold result")
    flight = hold.get("flight")
    if (hold.get("status") != "held" or hold.get("simulated") is not True
            or not isinstance(hold.get("hold_id"), str) or not hold["hold_id"].strip()
            or not isinstance(hold.get("passenger"), str) or not hold["passenger"].strip()
            or not isinstance(flight, Mapping) or not isinstance(flight.get("flight_number"), int)):
        raise ValueError("invalid hold evidence")


def validate_booking_evidence(booking: BookingResult | Mapping[str, Any], hold: HoldResult | Mapping[str, Any] | None) -> None:
    validate_hold_result(hold)
    if not isinstance(booking, Mapping) or set(booking) != {"booking_id", "hold_id", "flight", "passenger", "simulated", "status"}:
        raise ValueError("malformed booking result")
    if (booking.get("status") != "confirmed" or booking.get("simulated") is not True
            or not isinstance(booking.get("booking_id"), str) or not booking["booking_id"].strip()
            or booking.get("hold_id") != hold.get("hold_id")
            or booking.get("flight") != hold.get("flight")
            or booking.get("passenger") != hold.get("passenger")):
        raise ValueError("booking result does not match current hold")


def make_turn_outcome(
    *,
    intent: Intent,
    deterministic_response: str,
    search_results: FlightSearchResult | None = None,
    hold_result: HoldResult | None = None,
    booking_result: BookingResult | None = None,
    fallback_reason: str | None = None,
) -> TurnOutcome:
    """Build a versioned outcome and derive claims from its authoritative tool result."""
    if booking_result is not None:
        validate_booking_evidence(booking_result, hold_result)
    elif hold_result is not None:
        validate_hold_result(hold_result)
    claims = AuthoritativeClaims()
    if booking_result is not None:
        claims = AuthoritativeClaims(
            selected_flight_id=int(booking_result["flight"]["flight_number"]),
            hold_id=booking_result["hold_id"],
            booking_id=booking_result["booking_id"],
            booking_status="confirmed",
            passenger=booking_result["passenger"],
            simulated=booking_result["simulated"],
        )
    elif hold_result is not None:
        claims = AuthoritativeClaims(
            selected_flight_id=int(hold_result["flight"]["flight_number"]),
            hold_id=hold_result["hold_id"],
            passenger=hold_result["passenger"],
            simulated=hold_result["simulated"],
        )
    return TurnOutcome(
        schema=OUTCOME_SCHEMA,
        schema_version=OUTCOME_SCHEMA_VERSION,
        intent=intent,
        deterministic_response=deterministic_response,
        search_results=search_results,
        hold_result=hold_result,
        booking_result=booking_result,
        authoritative_claims=claims,
        fallback_reason=fallback_reason,
        simulator_disclosure_required=hold_result is not None or booking_result is not None,
    )

"""Typed deterministic simulator tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from .knowledge import FlightKnowledgeBase


@dataclass(frozen=True)
class SearchCriteria:
    origin: str | None = None
    destination: str | None = None
    departure_month: str | None = None
    departure_day: str | None = None
    return_month: str | None = None
    return_day: str | None = None
    departure_time: str | None = None
    return_time: str | None = None
    airline: str | None = None
    cabin_class: str | None = None
    max_price: int | None = None
    max_connections: int | None = None


@dataclass(frozen=True)
class FlightSearchResult:
    flights: tuple[dict[str, Any], ...]
    criteria: SearchCriteria


class FlightSimulator:
    def __init__(self, kb: FlightKnowledgeBase):
        self.kb = kb
        self.holds: dict[str, dict[str, Any]] = {}
        self.bookings: dict[str, dict[str, Any]] = {}

    def search_flights(self, criteria: SearchCriteria) -> FlightSearchResult:
        def matches(f: dict[str, Any]) -> bool:
            pairs = (("departure_airport", criteria.origin), ("return_airport", criteria.destination),
                     ("departure_month", criteria.departure_month), ("departure_day", criteria.departure_day),
                     ("return_month", criteria.return_month), ("return_day", criteria.return_day),
                     ("airline", criteria.airline), ("class", criteria.cabin_class))
            def same(field, actual, expected):
                if field.endswith("_month"):
                    return str(actual).lower()[:3] == str(expected).lower()[:3]
                return str(actual).lower() == str(expected).lower()

            if any(value is not None and not same(field, f.get(field), value)
                   for field, value in pairs):
                return False
            if criteria.max_price is not None and int(f["price"]) > criteria.max_price:
                return False
            if (criteria.max_connections is not None
                    and int(f["num_connections"]) > criteria.max_connections):
                return False
            buckets = {
                "morning": range(5, 12), "afternoon": range(12, 20),
                "evening": range(18, 24), "night": range(0, 5),
            }
            if (criteria.departure_time is not None
                    and int(f["departure_time_num"]) not in buckets[criteria.departure_time]):
                return False
            if (criteria.return_time is not None
                    and int(f["return_time_num"]) not in buckets[criteria.return_time]):
                return False
            return True

        return FlightSearchResult(tuple(f for f in self.kb.all() if matches(f)), criteria)

    def get_flight(self, flight_id: int) -> dict[str, Any] | None:
        return self.kb.get(flight_id)

    def hold_flight(self, flight_id: int, passenger: str) -> dict[str, Any]:
        flight = self.get_flight(flight_id)
        if not flight:
            raise ValueError("flight_id is not in the local knowledge base")
        if not passenger or len(passenger.strip()) < 2:
            raise ValueError("passenger must be a non-empty name")
        hold_id = "hold_" + uuid4().hex[:12]
        self.holds[hold_id] = {"hold_id": hold_id, "flight": flight, "passenger": passenger.strip()}
        return self.holds[hold_id].copy()

    def book_flight(self, hold_id: str) -> dict[str, Any]:
        hold = self.holds.get(hold_id)
        if not hold:
            raise ValueError("hold_id is unknown or expired")
        existing = next((b for b in self.bookings.values() if b["hold_id"] == hold_id), None)
        if existing:
            return existing.copy()
        booking_id = "booking_" + uuid4().hex[:12]
        result = {"booking_id": booking_id, **hold, "simulated": True}
        self.bookings[booking_id] = result
        return result.copy()

    def cancel_booking(self, booking_id: str) -> dict[str, Any]:
        booking = self.bookings.pop(booking_id, None)
        if not booking:
            raise ValueError("booking_id is unknown")
        return {"booking_id": booking_id, "cancelled": True, "simulated": True}


def flight_dict(flight: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy without exposing mutable simulator state."""
    return dict(flight)

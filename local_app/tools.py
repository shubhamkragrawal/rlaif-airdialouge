"""Typed deterministic simulator tools."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, TypedDict

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


class HoldResult(TypedDict):
    hold_id: str
    flight: dict[str, Any]
    passenger: str
    simulated: bool
    status: Literal["held"]


class BookingResult(TypedDict):
    booking_id: str
    hold_id: str
    flight: dict[str, Any]
    passenger: str
    simulated: bool
    status: Literal["confirmed"]


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

    def hold_flight(self, flight_id: int, passenger: str) -> HoldResult:
        flight = self.get_flight(flight_id)
        if not flight:
            raise ValueError("flight_id is not in the local knowledge base")
        if not passenger or len(passenger.strip()) < 2:
            raise ValueError("passenger must be a non-empty name")
        passenger = passenger.strip()
        fingerprint = sha256(f"{int(flight_id)}\0{passenger}".encode("utf-8")).hexdigest()
        hold_id = "hold_" + fingerprint[:24]
        existing = self.holds.get(hold_id)
        if existing is not None:
            expected = {
                "hold_id": hold_id, "flight": flight, "passenger": passenger,
                "simulated": True, "status": "held",
            }
            if existing != expected:
                raise RuntimeError("deterministic hold ID conflicts with an existing hold")
            return deepcopy(existing)
        result: HoldResult = {
            "hold_id": hold_id, "flight": flight, "passenger": passenger,
            "simulated": True, "status": "held",
        }
        self.holds[hold_id] = deepcopy(result)
        return deepcopy(result)

    def book_flight(self, hold_id: str) -> BookingResult:
        hold = self.holds.get(hold_id)
        if not isinstance(hold, dict):
            raise ValueError("hold_id is unknown or expired")
        required = {"hold_id", "flight", "passenger", "simulated", "status"}
        flight = hold.get("flight")
        if (set(hold) != required or hold.get("hold_id") != hold_id
                or hold.get("status") != "held" or hold.get("simulated") is not True
                or not isinstance(hold.get("passenger"), str)
                or len(hold["passenger"].strip()) < 2 or hold["passenger"] != hold["passenger"].strip()
                or not isinstance(flight, dict)
                or isinstance(flight.get("flight_number"), bool)
                or not isinstance(flight.get("flight_number"), int)
                or self.kb.get(flight["flight_number"]) != flight):
            raise ValueError("persisted hold failed schema, passenger, status, or inventory validation")
        existing = next((b for b in self.bookings.values()
                         if isinstance(b, dict) and b.get("hold_id") == hold_id), None)
        if existing:
            expected_id = "booking_" + sha256(hold_id.encode("utf-8")).hexdigest()[:24]
            expected = {
                "booking_id": expected_id, "hold_id": hold["hold_id"],
                "flight": hold["flight"], "passenger": hold["passenger"],
                "simulated": True, "status": "confirmed",
            }
            if (set(existing) != set(expected) or existing != expected
                    or existing.get("status") != "confirmed"
                    or existing.get("simulated") is not True):
                raise RuntimeError("existing booking conflicts with its hold")
            return deepcopy(existing)
        booking_id = "booking_" + sha256(hold_id.encode("utf-8")).hexdigest()[:24]
        result: BookingResult = {
            "booking_id": booking_id, "hold_id": hold["hold_id"],
            "flight": hold["flight"], "passenger": hold["passenger"],
            "simulated": True, "status": "confirmed",
        }
        conflicting = self.bookings.get(booking_id)
        if conflicting is not None and conflicting != result:
            raise RuntimeError("deterministic booking ID conflicts with an existing booking")
        self.bookings[booking_id] = deepcopy(result)
        return deepcopy(result)

    def cancel_booking(self, booking_id: str) -> dict[str, Any]:
        booking = self.bookings.pop(booking_id, None)
        if not booking:
            raise ValueError("booking_id is unknown")
        return {"booking_id": booking_id, "cancelled": True, "simulated": True}


def flight_dict(flight: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy without exposing mutable simulator state."""
    return dict(flight)

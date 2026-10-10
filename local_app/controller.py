"""Bounded conversation controller with optional model-backed responses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from .guardrails import safe_fallback, validate_grounding, validate_input
from .knowledge import FlightKnowledgeBase
from .tools import FlightSearchResult, FlightSimulator, SearchCriteria, flight_dict

if TYPE_CHECKING:
    from .runtime import ModelAdapter

ADAPTERS = ("sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50")


@dataclass
class SessionState:
    adapter: str = "sft"
    turns: list[dict[str, str]] = field(default_factory=list)
    last_search: FlightSearchResult | None = None
    selected_flight_id: int | None = None
    passenger: str | None = None
    hold_id: str | None = None
    booking: dict[str, Any] | None = None
    criteria: SearchCriteria = field(default_factory=SearchCriteria)
    turn_count: int = 0


class ConversationController:
    def __init__(
        self,
        kb: FlightKnowledgeBase | None = None,
        *,
        max_turns: int = 12,
        audit=None,
        model_adapters: Mapping[str, "ModelAdapter"] | None = None,
        model_loader: Callable[[str], "ModelAdapter"] | None = None,
    ):
        self.kb = kb or FlightKnowledgeBase()
        self.simulator = FlightSimulator(self.kb)
        self.max_turns = max_turns
        self.audit = audit
        self.model_adapters = dict(model_adapters or {})
        self.model_loader = model_loader
        unknown = set(self.model_adapters) - set(ADAPTERS)
        if unknown:
            raise ValueError(f"unknown model adapter configuration: {', '.join(sorted(unknown))}")

    def new_session(self, adapter: str = "sft") -> SessionState:
        if adapter not in ADAPTERS:
            raise ValueError(f"unknown adapter: {adapter}; choose from {', '.join(ADAPTERS)}")
        return SessionState(adapter=adapter)

    def handle(self, state: SessionState, message: str) -> dict[str, Any]:
        message = validate_input(message)
        if self.audit:
            self.audit.write("user_message", {"adapter": state.adapter, "message": message})
        if state.turn_count >= self.max_turns:
            return {"text": safe_fallback("this conversation because the turn limit was reached"),
                    "simulated": True}
        state.turn_count += 1
        state.turns.append({"role": "user", "content": message})
        self._extract_state(state, message)
        deterministic_response, results = self._respond(state, message)
        try:
            validate_grounding(deterministic_response, results)
        except ValueError:
            deterministic_response = safe_fallback("that response")
            results = ()
        response, model_used, usage = self._model_response(
            state, deterministic_response, results
        )
        state.turns.append({"role": "assistant", "content": response})
        result = {"text": response, "adapter": state.adapter, "turn": state.turn_count,
                "results": [flight_dict(f) for f in results], "simulated": True,
                "model_used": model_used, "usage": usage}
        if self.audit:
            self.audit.write("assistant_response", result)
        return result

    def _model_response(
        self,
        state: SessionState,
        deterministic_response: str,
        results: tuple[dict[str, Any], ...],
    ) -> tuple[str, bool, dict[str, int] | None]:
        """Generate with the selected model and fail back to grounded local text.

        Tools and state transitions have already been handled by ``_respond``.
        The model can phrase the answer, but it cannot directly mutate state or
        expand the set of flights that may be mentioned.
        """
        retrieved = [flight_dict(flight) for flight in results]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a local flight-simulator assistant. The simulator is not a real "
                    "airline or payment system. Use only the supplied retrieved flights, never "
                    "invent a flight number or price, and keep the answer concise. The controller's "
                    f"safe draft is: {deterministic_response}"
                ),
            },
            *state.turns,
        ]
        try:
            adapter = self.model_adapters.get(state.adapter)
            if adapter is None and self.model_loader is not None:
                adapter = self.model_loader(state.adapter)
            if adapter is None:
                return deterministic_response, False, None
            response = adapter.generate(messages, retrieved)
            self._validate_model_output(response, results, transactional=state.booking is not None)
        except Exception as exc:
            if self.audit:
                self.audit.write(
                    "model_fallback",
                    {"adapter": state.adapter, "error_type": type(exc).__name__},
                )
            return deterministic_response, False, None
        return response.strip(), True, getattr(adapter, "last_usage", None)

    @staticmethod
    def _validate_model_output(
        response: str, results: tuple[dict[str, Any], ...], *, transactional: bool = False
    ) -> None:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("model returned an empty or non-text response")
        if len(response) > 4000:
            raise ValueError("model response is too long")
        validate_grounding(response, results)

        # Prices are grounded separately because the shared flight-ID validator
        # intentionally ignores ordinary numbers such as stop counts.
        mentioned_prices = {int(value) for value in re.findall(r"\$\s*(\d+)", response)}
        known_prices = {int(flight["price"]) for flight in results}
        if mentioned_prices - known_prices:
            raise ValueError("response mentions prices outside the retrieved context")
        lowered = response.lower()
        if results and re.search(r"\b(?:no|not any)\s+(?:matching\s+|available\s+)?flights?\b", lowered):
            raise ValueError("response contradicts the non-empty search result")
        if any(claim in lowered for claim in (
            "payment was charged", "real reservation", "airline confirmation number"
        )):
            raise ValueError("response makes an unsupported real-transaction claim")
        if transactional and "simulat" not in lowered:
            raise ValueError("booking response must state that the transaction is simulated")

    def _extract_state(self, state: SessionState, message: str) -> None:
        ids = [int(value) for value in re.findall(r"\b\d{3,5}\b", message)]
        if ids and state.last_search and any(int(f["flight_number"]) == ids[-1] for f in state.last_search.flights):
            state.selected_flight_id = ids[-1]
        if re.search(r"\b(my name is|name is)\b", message, re.I):
            state.passenger = re.split(r"\b(?:my name is|name is)\b", message, flags=re.I)[-1].strip(" .")

    def _respond(self, state: SessionState, message: str) -> tuple[str, tuple[dict[str, Any], ...]]:
        if state.selected_flight_id and state.passenger and re.search(r"\b(book|confirm|proceed|reserve)\b", message, re.I):
            hold = self.simulator.hold_flight(state.selected_flight_id, state.passenger)
            state.hold_id = hold["hold_id"]
            state.booking = self.simulator.book_flight(state.hold_id)
            return f"Your simulated booking is confirmed for flight {state.selected_flight_id}.", (hold["flight"],)
        criteria = self._criteria_from_text(message)
        updates = {name: value for name, value in asdict(criteria).items() if value is not None}
        if criteria.origin and not criteria.destination and state.criteria.origin:
            updates["destination"] = criteria.origin
            updates.pop("origin", None)
        if updates:
            state.criteria = replace(state.criteria, **updates)
        if state.criteria.origin and state.criteria.destination and updates:
            state.last_search = self.simulator.search_flights(state.criteria)
            if not state.last_search.flights:
                return "I couldn't find a matching flight in the local simulator.", ()
            shown = state.last_search.flights[:5]
            summary = "; ".join(f"{f['flight_number']} ({f['airline']}, ${f['price']})" for f in shown)
            return f"I found these simulated flights: {summary}. Which flight would you like?", shown
        if not state.criteria.origin or not state.criteria.destination:
            return "Which departure and destination airport codes should I search?", ()
        if not state.passenger:
            return "What name should I use for the passenger?", ()
        if state.last_search and state.last_search.flights:
            return "Please select one of the flights from the latest search.", ()
        return "Please provide departure and destination airports, travel dates, or a flight number.", ()

    def _criteria_from_text(self, text: str) -> SearchCriteria:
        # Airport codes in the corpus are uppercase; do not uppercase the whole
        # message first, or a name such as "Amy" becomes a false airport code.
        airports = re.findall(r"\b[A-Z]{3}\b", text)
        origin, destination = (airports + [None, None])[:2]
        max_price = None
        price = re.search(r"(?:under|below|max(?:imum)?|less than)\s*\$?\s*(\d+)", text, re.I)
        if price:
            max_price = int(price.group(1))
        nonstop = bool(re.search(r"\b(nonstop|direct)\b", text, re.I))
        connection_limit = re.search(r"at most\s+(\d+)\s+connections?", text, re.I)
        max_connections = 0 if nonstop else (
            int(connection_limit.group(1)) if connection_limit else None
        )
        dates = re.findall(r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])\b", text)
        months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
        departure_month = departure_day = return_month = return_day = None
        if dates:
            departure_month, departure_day = months[int(dates[0][0]) - 1], str(int(dates[0][1]))
        if len(dates) > 1:
            return_month, return_day = months[int(dates[1][0]) - 1], str(int(dates[1][1]))
        departure_time = return_time = None
        for bucket in ("morning", "afternoon", "evening", "night"):
            if re.search(rf"\b(?:return|arriv\w*)\b[^.]*\b{bucket}\b", text, re.I):
                return_time = bucket
            elif re.search(rf"\b(?:depart\w*|leave|outbound)\b[^.]*\b{bucket}\b", text, re.I):
                departure_time = bucket
        return SearchCriteria(origin=origin, destination=destination,
                              departure_month=departure_month, departure_day=departure_day,
                              return_month=return_month, return_day=return_day,
                              departure_time=departure_time, return_time=return_time,
                              max_price=max_price,
                              max_connections=max_connections)

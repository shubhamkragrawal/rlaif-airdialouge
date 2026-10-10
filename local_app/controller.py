"""Bounded conversation controller with optional model-backed responses."""

from __future__ import annotations

import re
import json
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from .guardrails import (TransactionClaimError, detect_transaction_claim, safe_fallback,
                         validate_grounding, validate_input, validate_transaction_claim)
from .knowledge import FlightKnowledgeBase
from .outcomes import (Intent, TurnOutcome, make_turn_outcome,
                       validate_booking_evidence, validate_hold_result)
from .tools import FlightSearchResult, FlightSimulator, SearchCriteria, flight_dict

if TYPE_CHECKING:
    from .runtime import ModelAdapter

ADAPTERS = ("sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50")
BOOKING_COMMAND_PATTERN = r"(?:book|reserve|confirm|proceed\s+with|continue\s+with|go\s+ahead\s+with)"


def _normalize_intent_text(text: str) -> str:
    return text.translate(str.maketrans({
        "’": "'", "‘": "'", "ʼ": "'", "–": "-", "—": "-", "−": "-",
    }))


def _is_negated(text: str, match_start: int) -> bool:
    text = _normalize_intent_text(text)
    clause_start = max(text.rfind(mark, 0, match_start) for mark in ".!?;") + 1
    prefix = text[clause_start:match_start]
    return bool(re.search(r"\b(?:do\s+not|don't|dont|never|not|cannot|can't|shouldn't|should\s+not)\b[^,]*$", prefix, re.I))


def _is_booking_intent(text: str) -> bool:
    """Match affirmative requests/commands, not negation, questions, or availability."""
    text = _normalize_intent_text(text)
    # An explicit later withdrawal cancels an earlier affirmative command.
    modifiers = r"(?:(?:actually|immediately|ever|really|just|please|possibly|at\s+all|yet|then|right\s+now)\s+){0,8}"
    withdrawal = rf"\b(?:do\s+not|don't|dont|never|should\s+not|shouldn't)\s+{modifiers}{BOOKING_COMMAND_PATTERN}\b"
    if re.search(withdrawal, text, re.I):
        return False
    affirmative = re.search(rf"\b{BOOKING_COMMAND_PATTERN}\b", text, re.I)
    if affirmative and re.search(r"(?:^|[.!?;,—-])\s*(?:no|nope|cancel|never\s+mind|nevermind)\s*[.!?;,—-]*$", text[affirmative.end():], re.I):
        return False
    command = re.search(
        rf"\b{BOOKING_COMMAND_PATTERN}\b",
        text,
        re.I,
    )
    if not command or _is_negated(text, command.start()):
        return False
    prefix = text[:command.start()].strip()
    direct = not prefix or bool(re.fullmatch(r"(?:please|yes|okay|ok|go ahead|please\s+go ahead|and|then|now|i'll|i will|i want to|i'd like to|i would like to|can you|could you|would you|please\s+can you|please\s+could you)(?:\s+(?:please|now|go ahead|and|then))*", prefix, re.I))
    if not direct:
        return False
    # Availability and capability questions are not authorization to transact.
    if re.match(r"\s*(?:is|are|was|were)\b", text, re.I) and re.search(r"\bavailable\b|\bavailability\b", text, re.I):
        return False
    if re.match(r"\s*(?:can|could|would)\s+i\b", text, re.I):
        return False
    return True


def _explicit_selections(text: str) -> list[tuple[str, int | None]]:
    """Lex all flight references before interpreting pronouns/articles."""
    text = _normalize_intent_text(text)
    if re.fullmatch(r"\s*[+-]?\d+\s*", text):
        token = text.strip()
        return [(token, int(token) if token.isdigit() else None)]
    if re.match(r"\s*(?:is|are|was|were)\b", text, re.I) and re.search(r"\bavailable\b", text, re.I):
        return []
    # Parse the entire token next to the flight/selection phrase. Capturing a
    # numeric prefix alone would accidentally accept 9999X or 1027.5.
    tokens = re.findall(r"\bflight\s+(?:number\s+|#\s*)?([^\s,;!?]+)", text, re.I)
    if not tokens:
        m = re.search(r"\b(?:book|reserve|confirm|select|choose|proceed\s+with|continue\s+with|go\s+ahead\s+with)\s+(?:the\s+|a\s+|this\s+|that\s+)?([^\s,;!?]+)", text, re.I)
        tokens = [m.group(1)] if m else []
    selections = []
    for token in tokens:
        if token.lower() in {"it", "this", "that", "the", "a", "my", "our", "booking", "reservation", "flight"}:
            continue
        if re.fullmatch(r"\d+\.", token):
            token = token[:-1]
        selections.append((token, int(token) if re.fullmatch(r"\d{3,5}", token) else None))
    return selections


def _explicit_selection(text: str) -> tuple[str, int | None] | None:
    selections = _explicit_selections(text)
    return selections[0] if selections else None


def _is_selection_intent(text: str) -> bool:
    text = _normalize_intent_text(text)
    if re.match(r"\s*(?:is|are|was|were)\b", text, re.I) and re.search(r"\bavailable\b", text, re.I):
        return False
    if re.fullmatch(r"\s*[+-]?\d+\s*", text):
        return True
    if re.match(r"\s*flight\b", text, re.I):
        return True
    selection = re.search(r"\b(?:book|reserve|confirm|select|choose|proceed\s+with|continue\s+with|go\s+ahead\s+with)\b", text, re.I)
    return bool(selection and not _is_negated(text, selection.start()))


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
        try:
            outcome = self._respond(state, message)
        except Exception as exc:
            outcome = make_turn_outcome(
                intent=Intent.REFUSE,
                deterministic_response="I couldn't safely complete that simulator request. Please try again.",
                fallback_reason="tool_error:" + type(exc).__name__,
            )
        results = outcome.search_results.flights[:5] if outcome.search_results else ()
        booking_intent = _is_booking_intent(message)
        authoritative_text = (booking_intent or outcome.booking_result is not None or outcome.fallback_reason is not None
                              or (outcome.search_results is not None and not outcome.search_results.flights))
        if outcome.booking_result is not None:
            results = (outcome.booking_result["flight"],)
        try:
            if outcome.fallback_reason is None:
                self._validate_model_output(outcome.deterministic_response, outcome)
        except ValueError:
            outcome = make_turn_outcome(intent=Intent.REFUSE, deterministic_response=safe_fallback("that response"), fallback_reason="grounding_validation_failed")
            authoritative_text = True
        if authoritative_text:
            response, model_used, usage = outcome.deterministic_response, False, None
        else:
            response, model_used, usage, model_fallback_reason = self._model_response(state, outcome)
            if not model_used and (self.model_adapters.get(state.adapter) is not None or self.model_loader is not None):
                outcome = replace(outcome, fallback_reason=model_fallback_reason or "model_validation_failed")
        state.turns.append({"role": "assistant", "content": response})
        result = {"text": response, "adapter": state.adapter, "turn": state.turn_count,
                "results": [flight_dict(f) for f in results], "simulated": True,
                "model_used": model_used, "usage": usage,
                "fallback_reason": outcome.fallback_reason,
                "validation_reason": outcome.fallback_reason}
        if self.audit:
            self.audit.write("assistant_response", result)
        return result

    def _model_response(
        self,
        state: SessionState,
        outcome: TurnOutcome,
    ) -> tuple[str, bool, dict[str, int] | None, str | None]:
        """Generate with the selected model and fail back to grounded local text.

        Tools and state transitions have already been handled by ``_respond``.
        The model can phrase the answer, but it cannot directly mutate state or
        expand the set of flights that may be mentioned.
        """
        deterministic_response = outcome.deterministic_response
        results = self._outcome_flights(outcome)
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
                return deterministic_response, False, None, None
            response = adapter.generate(messages, retrieved)
            self._validate_model_output(response, outcome)
        except Exception as exc:
            if self.audit:
                self.audit.write(
                    "model_fallback",
                    {"adapter": state.adapter, "error_type": type(exc).__name__},
                )
            reason = ("unsupported_transaction_claim" if "unsupported_transaction_claim" in str(exc)
                      else "model_validation_failed")
            return deterministic_response, False, None, reason
        return response.strip(), True, getattr(adapter, "last_usage", None), None

    @staticmethod
    def _outcome_flights(outcome: TurnOutcome) -> tuple[Mapping[str, Any], ...]:
        if outcome.booking_result is not None:
            return (outcome.booking_result["flight"],)
        if outcome.search_results is not None:
            return outcome.search_results.flights[:5]
        return ()

    def _validated_search(self, value: Any, criteria: SearchCriteria) -> FlightSearchResult:
        if not isinstance(value, FlightSearchResult) or value.criteria != criteria:
            raise ValueError("malformed simulator search result")
        if not isinstance(value.flights, tuple):
            raise ValueError("malformed simulator flight collection")
        snapshot = []
        seen = set()
        for flight in value.flights:
            if not isinstance(flight, Mapping):
                raise ValueError("malformed simulator flight")
            flight_id = flight.get("flight_number")
            if isinstance(flight_id, bool) or not isinstance(flight_id, int) or flight_id in seen:
                raise ValueError("invalid simulator flight number")
            inventory_flight = self.kb.get(flight_id)
            if inventory_flight is None or dict(flight) != inventory_flight:
                raise ValueError("simulator search returned a flight outside local inventory")
            seen.add(flight_id)
            snapshot.append(dict(flight))
        return FlightSearchResult(tuple(snapshot), value.criteria)

    @classmethod
    def _validate_model_output(cls, response: str, outcome: TurnOutcome) -> None:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("model returned an empty or non-text response")
        if len(response) > 4000:
            raise ValueError("model response is too long")
        stripped = response.strip()
        try:
            raw_value = json.loads(stripped)
        except (ValueError, TypeError):
            raw_value = None
        if stripped in {"[]", "{}"} or isinstance(raw_value, (dict, list)):
            raise ValueError("model returned raw structured data")
        if re.search(r"(?i)(?:sessionstate\s*\(|\b(?:state|selected_flight_id|last_search|hold_id|booking)\s*[:=])", stripped):
            raise ValueError("model returned internal state")
        results = cls._outcome_flights(outcome)
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
        if detect_transaction_claim(response):
            booking = outcome.booking_result
            hold = outcome.hold_result
            evidence = None if booking is None else {
                "booking_id": booking.get("booking_id"), "hold_id": booking.get("hold_id"),
                "flight": booking.get("flight"),
                "flight_id": booking.get("flight", {}).get("flight_number"),
                "passenger": booking.get("passenger"), "status": booking.get("status"),
                "simulated": booking.get("simulated"), "hold": hold,
            }
            try:
                validate_transaction_claim(response, evidence)
            except TransactionClaimError as exc:
                raise ValueError(exc.reason) from exc
        if outcome.simulator_disclosure_required and "simulat" not in lowered:
            raise ValueError("transaction response must state that the transaction is simulated")

    def _extract_state(self, state: SessionState, message: str) -> None:
        ids = [int(value) for value in re.findall(r"\b\d{3,5}\b", message)]
        if ids and state.last_search and any(int(f["flight_number"]) == ids[-1] for f in state.last_search.flights):
            state.selected_flight_id = ids[-1]
        if re.search(r"\b(my name is|name is)\b", message, re.I):
            state.passenger = re.split(r"\b(?:my name is|name is)\b", message, flags=re.I)[-1].strip(" .")

    def _respond(self, state: SessionState, message: str) -> TurnOutcome:
        booking_requested = _is_booking_intent(message)
        selection_request = _is_selection_intent(message)
        explicit_selections = _explicit_selections(message) if selection_request else []
        explicit_selection = explicit_selections[0] if explicit_selections else None
        criteria = self._criteria_from_text(message)
        updates = {name: value for name, value in asdict(criteria).items() if value is not None}
        if criteria.origin and not criteria.destination and state.criteria.origin:
            updates["destination"] = criteria.origin
            updates.pop("origin", None)
        material_change = bool(updates and replace(state.criteria, **updates) != state.criteria)
        if material_change:
            next_criteria = replace(state.criteria, **updates)
            try:
                raw_search = (self.simulator.search_flights(next_criteria)
                              if next_criteria.origin and next_criteria.destination else None)
                next_search = (self._validated_search(raw_search, next_criteria)
                               if next_criteria.origin and next_criteria.destination else None)
            except Exception as exc:
                return make_turn_outcome(intent=Intent.REFUSE,
                    deterministic_response="I couldn't safely complete that simulator search. Please try again.",
                    fallback_reason="tool_error:" + type(exc).__name__)
            # Invalidate stale transaction state before evaluating any booking intent.
            state.criteria = next_criteria
            state.selected_flight_id = None
            state.hold_id = None
            state.last_search = next_search
            if next_search is not None:
                search = next_search
                if explicit_selections:
                    invalid = next(((token, flight_id) for token, flight_id in explicit_selections
                                    if flight_id is None or not any(int(f["flight_number"]) == flight_id
                                                                    for f in search.flights)), None)
                    if invalid is not None:
                        token, _flight_id = invalid
                        state.selected_flight_id = None
                        return make_turn_outcome(intent=Intent.REFUSE,
                            deterministic_response=f"Flight {token} is not among the current simulator options. Please choose a flight from the latest search.",
                            search_results=search, fallback_reason="invalid_flight_selection")
                if not search.flights:
                    return make_turn_outcome(intent=Intent.SEARCH, deterministic_response="I couldn't find a matching flight in the local simulator.", search_results=search)
                if not booking_requested:
                    shown = search.flights[:5]
                    summary = "; ".join(f"{f['flight_number']} ({f['airline']}, ${f['price']})" for f in shown)
                    return make_turn_outcome(intent=Intent.SEARCH, deterministic_response=f"I found these simulated flights: {summary}. Which flight would you like?", search_results=search)
        explicit_id = explicit_selection[1] if explicit_selection else None
        invalid = next(((token, flight_id) for token, flight_id in explicit_selections
                        if flight_id is None or not state.last_search
                        or not any(int(f["flight_number"]) == flight_id for f in state.last_search.flights)), None)
        if invalid is not None:
            state.selected_flight_id = None
            token, _flight_id = invalid
            return make_turn_outcome(intent=Intent.REFUSE, deterministic_response=f"Flight {token} is not among the current simulator options. Please choose a flight from the latest search.", search_results=state.last_search, fallback_reason="invalid_flight_selection")
        if explicit_selection and explicit_id is not None and state.last_search:
            state.selected_flight_id = explicit_id
        passenger_match = re.search(r"\b(?:my name is|name is|book for|passenger)\s+([A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){1,3})|\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})\s+is traveling", message)
        if passenger_match:
            state.passenger = (passenger_match.group(1) or passenger_match.group(2)).strip()[:100]
        if booking_requested and state.booking is not None:
            cached = state.booking
            prior_hold = self.simulator.holds.get(state.hold_id)
            latest_flight = (next((f for f in state.last_search.flights
                                  if int(f["flight_number"]) == state.selected_flight_id), None)
                             if state.last_search else None)
            existing_bookings = deepcopy(self.simulator.bookings)
            existing_holds = deepcopy(self.simulator.holds)
            try:
                if (not isinstance(cached, dict) or state.hold_id != cached.get("hold_id")
                        or not isinstance(prior_hold, dict) or latest_flight is None
                        or cached.get("flight", {}).get("flight_number") != state.selected_flight_id
                        or cached.get("passenger") != state.passenger):
                    raise ValueError("cached booking no longer matches the current selection and hold")
                if self.simulator.bookings.get(cached.get("booking_id")) != cached:
                    raise ValueError("cached booking is absent or differs from simulator storage")
                validate_hold_result(prior_hold)
                if (prior_hold["hold_id"] != state.hold_id
                        or prior_hold["flight"] != latest_flight
                        or prior_hold["passenger"] != state.passenger):
                    raise ValueError("persisted hold does not match current search and passenger")
                fresh_booking = self.simulator.book_flight(state.hold_id)
                validate_booking_evidence(fresh_booking, prior_hold)
                # Re-read both stores after the simulator call: a hook or
                # concurrent mutation must not leave confirmation based on a
                # returned object whose persisted hold/booking has changed.
                if (self.simulator.holds.get(state.hold_id) != prior_hold
                        or not isinstance(fresh_booking, Mapping)
                        or self.simulator.bookings.get(fresh_booking.get("booking_id")) != fresh_booking
                        or fresh_booking != cached
                        or fresh_booking["flight"] != latest_flight
                        or fresh_booking["hold_id"] != state.hold_id
                        or fresh_booking["passenger"] != state.passenger
                        or fresh_booking["flight"]["flight_number"] != state.selected_flight_id):
                    raise ValueError("fresh booking evidence differs from cached booking")
                return self._booking_outcome(fresh_booking, prior_hold)
            except Exception:
                self.simulator.bookings.clear()
                self.simulator.bookings.update(existing_bookings)
                self.simulator.holds.clear()
                self.simulator.holds.update(existing_holds)
                state.booking = None
                state.hold_id = None
                return make_turn_outcome(intent=Intent.REFUSE, deterministic_response="I couldn't verify the earlier simulated booking. Please reset the session before trying again.", fallback_reason="stale_booking_evidence")
        if booking_requested:
            selected = state.selected_flight_id
            latest = state.last_search
            if selected is None or latest is None or not any(int(f["flight_number"]) == selected for f in latest.flights):
                return make_turn_outcome(intent=Intent.REFUSE, deterministic_response="Please select a flight from the latest simulator search before booking.", search_results=latest, fallback_reason="selection_not_in_latest_search")
            if not state.passenger:
                return make_turn_outcome(intent=Intent.CLARIFY, deterministic_response="What name should I use for the passenger?", search_results=latest)
            # Verify membership immediately before simulator side effects; only commit state after both succeed.
            if not any(int(f["flight_number"]) == selected for f in state.last_search.flights):
                return make_turn_outcome(intent=Intent.REFUSE, deterministic_response="That flight is not among the current simulator options.", search_results=latest, fallback_reason="invalid_flight_selection")
            original_session = (state.selected_flight_id, state.passenger, state.hold_id, state.booking)
            simulator_holds = deepcopy(self.simulator.holds)
            simulator_bookings = deepcopy(self.simulator.bookings)
            try:
                hold = self.simulator.hold_flight(selected, state.passenger)
                validate_hold_result(hold)
                requested_flight = next(f for f in latest.flights
                                        if int(f["flight_number"]) == selected)
                if (hold["flight"]["flight_number"] != selected
                        or dict(hold["flight"]) != dict(requested_flight)
                        or hold["passenger"] != state.passenger.strip()):
                    raise ValueError("hold does not match the selected current-search flight and passenger")
                if self.simulator.holds.get(hold["hold_id"]) != hold:
                    raise ValueError("returned hold does not exactly match simulator storage")
                booking = self.simulator.book_flight(hold["hold_id"])
                validate_booking_evidence(booking, hold)
                if (booking["flight"]["flight_number"] != selected
                        or dict(booking["flight"]) != dict(requested_flight)
                        or booking["passenger"] != state.passenger.strip()):
                    raise ValueError("booking does not match the selected current-search flight and passenger")
                if self.simulator.bookings.get(booking["booking_id"]) != booking:
                    raise ValueError("returned booking does not exactly match simulator storage")
                if self.simulator.holds.get(hold["hold_id"]) != hold:
                    raise ValueError("persisted hold changed during booking")
                booking_outcome = self._booking_outcome(booking, hold)
            except Exception as exc:
                self.simulator.holds.clear()
                self.simulator.holds.update(simulator_holds)
                self.simulator.bookings.clear()
                self.simulator.bookings.update(simulator_bookings)
                (state.selected_flight_id, state.passenger, state.hold_id, state.booking) = original_session
                return make_turn_outcome(intent=Intent.REFUSE, deterministic_response="I couldn't safely complete that simulated booking. Please try again.", search_results=latest, fallback_reason="tool_error:" + type(exc).__name__)
            state.hold_id, state.booking = hold["hold_id"], booking
            return booking_outcome
        if state.criteria.origin and state.criteria.destination and not state.last_search:
            try:
                search = self._validated_search(self.simulator.search_flights(state.criteria), state.criteria)
            except Exception as exc:
                return make_turn_outcome(intent=Intent.REFUSE,
                    deterministic_response="I couldn't safely complete that simulator search. Please try again.",
                    fallback_reason="tool_error:" + type(exc).__name__)
            state.last_search = search
            if not search.flights:
                return make_turn_outcome(intent=Intent.SEARCH, deterministic_response="I couldn't find a matching flight in the local simulator.", search_results=search)
            shown = search.flights[:5]
            summary = "; ".join(f"{f['flight_number']} ({f['airline']}, ${f['price']})" for f in shown)
            return make_turn_outcome(intent=Intent.SEARCH, deterministic_response=f"I found these simulated flights: {summary}. Which flight would you like?", search_results=state.last_search)
        if not state.criteria.origin or not state.criteria.destination:
            return make_turn_outcome(intent=Intent.CLARIFY, deterministic_response="Which departure and destination airport codes should I search?")
        if not state.passenger:
            return make_turn_outcome(intent=Intent.CLARIFY, deterministic_response="What name should I use for the passenger?", search_results=state.last_search)
        if state.last_search and state.last_search.flights:
            return make_turn_outcome(intent=Intent.CLARIFY, deterministic_response="Please select one of the flights from the latest search.", search_results=state.last_search)
        return make_turn_outcome(intent=Intent.OTHER, deterministic_response="Please provide departure and destination airports, travel dates, or a flight number.")

    @staticmethod
    def _booking_outcome(booking, hold):
        flight_id = booking["flight"]["flight_number"]
        return make_turn_outcome(intent=Intent.BOOK, deterministic_response=f"Your simulated booking is confirmed for flight {flight_id}.", hold_result=hold, booking_result=booking)

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

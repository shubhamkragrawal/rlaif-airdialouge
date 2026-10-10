"""Bounded conversation controller with optional model-backed responses."""

from __future__ import annotations

import re
import inspect
import math
import threading
from hashlib import sha256
from copy import copy, deepcopy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from .guardrails import safe_fallback, validate_input
from .knowledge import FlightKnowledgeBase
from .outcomes import (Intent, TurnOutcome, make_turn_outcome,
                       validate_booking_evidence, validate_hold_result)
from .tools import FlightSearchResult, FlightSimulator, SearchCriteria, flight_dict
from .parsing import parse_passenger_name
from .output_validation import validate_output
from .limits import ExecutionBudget, ExecutionLimits, LimitReason

if TYPE_CHECKING:
    from .runtime import ModelAdapter

ADAPTERS = ("sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50")
BOOKING_COMMAND_PATTERN = r"(?:book|reserve|confirm|proceed\s+with|continue\s+with|go\s+ahead\s+with)"


def _normalize_passenger_introducers(text: str) -> str:
    """Canonicalize only explicit passenger introducers before parsing."""
    return re.sub(
        r"\bmy\s+name\s+is\b|\bbook\s+for\b|\bpassenger\b",
        lambda match: re.sub(r"\s+", " ", match.group(0)).casefold(),
        text, flags=re.I,
    )


def _parse_controller_passenger(text: str):
    parsed = parse_passenger_name(_normalize_passenger_introducers(text))
    phrase = (parsed.matched_phrase or "").casefold()
    if (parsed.status == "rejected"
            and parsed.reason == "airport_code_or_acronym"
            and phrase in {"my name is", "book for", "passenger", "for"}
            and parsed.candidate
            and (phrase != "for" or len(parsed.candidate.split()) >= 2)
            and parsed.candidate == parsed.candidate.upper()):
        parsed = parse_passenger_name("passenger " + parsed.candidate.title())
    return parsed


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
        if token.lower() in {"it", "this", "that", "the", "a", "my", "our", "for", "booking", "reservation", "flight"}:
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
    budget: ClassVar[ExecutionBudget | None] = None


class _BudgetExceeded(RuntimeError):
    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__(reasons[0].value)


class _CapacityExceeded(RuntimeError):
    def __init__(self, capacity):
        self.reason = f"{capacity}_capacity_exceeded"
        super().__init__(self.reason)


class ConversationController:
    def __init__(
        self,
        kb: FlightKnowledgeBase | None = None,
        *,
        max_turns: int = 12,
        audit=None,
        model_adapters: Mapping[str, "ModelAdapter"] | None = None,
        model_loader: Callable[[str], "ModelAdapter"] | None = None,
        limits: ExecutionLimits | None = None,
    ):
        self.kb = kb or FlightKnowledgeBase()
        self.simulator = FlightSimulator(self.kb)
        self.max_turns = max_turns
        self.limits = limits or ExecutionLimits(max_turns=max_turns)
        self.audit = audit
        self.model_adapters = dict(model_adapters or {})
        self.model_loader = model_loader
        self._adapter_call_lock = threading.Lock()
        self._call_workers = {"model": 0, "tool": 0}
        self._call_limits = {"model": 2, "tool": 2}
        self._store_revision_lock = threading.Lock()
        self._store_revisions: dict[tuple[str, str], int] = {}
        unknown = set(self.model_adapters) - set(ADAPTERS)
        if unknown:
            raise ValueError(f"unknown model adapter configuration: {', '.join(sorted(unknown))}")

    def new_session(self, adapter: str = "sft") -> SessionState:
        if adapter not in ADAPTERS:
            raise ValueError(f"unknown adapter: {adapter}; choose from {', '.join(ADAPTERS)}")
        state = SessionState(adapter=adapter)
        state.budget = ExecutionBudget(self.limits)
        return state

    def handle(self, state: SessionState, message: str) -> dict[str, Any]:
        message = validate_input(message)
        parsed_passenger = _parse_controller_passenger(message)
        if self.audit:
            if parsed_passenger.found:
                self.audit.register_passenger(parsed_passenger.name)
            self.audit.write("user_message", {"adapter": state.adapter, "message": message,
                                               "passenger_name": parsed_passenger.name,
                                               "passenger_candidate": parsed_passenger.candidate
                                               if parsed_passenger.status == "rejected" else None})
        budget = state.budget or ExecutionBudget(self.limits)
        state.budget = budget
        reasons = budget.consume_turn()
        if reasons:
            return self._limit_response(reasons, state)
        state.turn_count += 1
        state.turns.append({"role": "user", "content": message})
        if parsed_passenger.found:
            state.passenger = parsed_passenger.name
        elif (parsed_passenger.status == "rejected" and
              ((parsed_passenger.matched_phrase or "").casefold() in {"passenger", "book for", "my name is"}
               or parsed_passenger.reason not in {"command_or_airport_code", "airport_code_or_acronym"}
               or (parsed_passenger.matched_phrase == "for"
                   and len((parsed_passenger.candidate or "").split()) >= 2
                   and not re.search(r"\b(?:flight|reservation|booking|please|today|tomorrow)\b",
                                     parsed_passenger.candidate or "", re.I)))):
            # An explicit but rejected passenger must never inherit a previous
            # passenger from session state.
            state.passenger = None
            reason = "passenger_" + (parsed_passenger.reason or "invalid")
            response = "I couldn't use that passenger name. Please provide a valid name up to 100 characters."
            state.turns.append({"role": "assistant", "content": response})
            result = {"text": response, "adapter": state.adapter, "turn": state.turn_count,
                      "results": [], "simulated": True, "model_used": False,
                      "usage": None, "fallback_reason": reason, "validation_reason": reason}
            if self.audit: self.audit.write("assistant_response", result)
            return result
        try:
            outcome = self._respond(state, message)
        except _BudgetExceeded as exc:
            reason = exc.reasons[0].value
            outcome = make_turn_outcome(intent=Intent.REFUSE,
                deterministic_response=self._limit_response(exc.reasons, state)["text"],
                fallback_reason=reason)
        except _CapacityExceeded as exc:
            outcome = make_turn_outcome(intent=Intent.REFUSE,
                deterministic_response="The simulator is temporarily at capacity. Please try again.",
                fallback_reason=exc.reason)
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
                errors = validate_output(outcome.deterministic_response, outcome)
                if errors: raise ValueError(errors[0].code)
        except ValueError as exc:
            code = str(exc) or "malformed_output"
            outcome = make_turn_outcome(intent=Intent.REFUSE, deterministic_response=safe_fallback("that response"), fallback_reason=code)
            authoritative_text = True
        if authoritative_text:
            response, model_used, usage = outcome.deterministic_response, False, None
        else:
            response, model_used, usage, model_fallback_reason = self._model_response(state, outcome)
            deadline_reasons = state.budget.exceeded_reasons() if state.budget else ()
            if deadline_reasons:
                model_used, usage = False, None
                response = outcome.deterministic_response
                model_fallback_reason = deadline_reasons[0].value
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
            if adapter is None and self.model_loader is None:
                return deterministic_response, False, None, None
            if self.limits.max_generation_tokens == 0:
                return deterministic_response, False, None, LimitReason.GENERATION_LENGTH.value
            reasons = state.budget.check_generation_length(self.limits.max_generation_tokens) if state.budget else ()
            if reasons:
                return deterministic_response, False, None, reasons[0].value
            remaining = state.budget.remaining_seconds() if state.budget else self.limits.max_session_seconds
            def load_and_generate(cancel_event):
                if adapter is not None:
                    loaded = adapter
                else:
                    loader_params = inspect.signature(self.model_loader).parameters
                    loader_kwargs = {}
                    if "cancel_event" in loader_params: loader_kwargs["cancel_event"] = cancel_event
                    if "deadline_seconds" in loader_params: loader_kwargs["deadline_seconds"] = remaining
                    loaded = self.model_loader(state.adapter, **loader_kwargs)
                if cancel_event.is_set():
                    raise TimeoutError("session_time_exceeded")
                generate = loaded.generate
                try:
                    parameters = inspect.signature(generate).parameters
                except (TypeError, ValueError):
                    parameters = {}
                kwargs = {}
                if "max_new_tokens" in parameters:
                    kwargs["max_new_tokens"] = min(self.limits.max_generation_tokens,
                        max(0, math.ceil(remaining * max(1, self.limits.max_generation_tokens) /
                                   max(self.limits.max_session_seconds, 1e-9))))
                if "deadline_seconds" in parameters:
                    kwargs["deadline_seconds"] = remaining
                if "cancel_event" in parameters:
                    kwargs["cancel_event"] = cancel_event
                return generate(messages, retrieved, **kwargs), loaded

            response, adapter = self._bounded_adapter_call(load_and_generate, remaining)
            usage = getattr(adapter, "last_usage", None)
            if isinstance(usage, Mapping):
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if isinstance(output_tokens, int) and not isinstance(output_tokens, bool) and output_tokens > self.limits.max_generation_tokens:
                    return deterministic_response, False, None, LimitReason.GENERATION_LENGTH.value
                if (isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens <= 0
                        or isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens <= 0):
                    return deterministic_response, False, None, "malformed_usage"
            elif usage is None and isinstance(response, str):
                # UTF-8 bytes provide a deterministic conservative upper bound
                # when the adapter does not report counters.
                input_tokens = len(repr((messages, retrieved)).encode("utf-8"))
                output_tokens = len(response.encode("utf-8"))
                if output_tokens > self.limits.max_generation_tokens:
                    return deterministic_response, False, None, LimitReason.GENERATION_LENGTH.value
                usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                         "estimated": True}
            else:
                return deterministic_response, False, None, "malformed_usage"
            if state.budget and state.budget.exceeded_reasons():
                reason = state.budget.exceeded_reasons()[0].value
                return deterministic_response, False, None, reason
            errors = validate_output(response, outcome)
            if errors:
                raise ValueError(errors[0].code)
        except Exception as exc:
            if self.audit:
                self.audit.write(
                    "model_fallback",
                    {"adapter": state.adapter, "error_type": type(exc).__name__,
                     "worker_active": self.adapter_worker_active,
                     "limitation": ("non_cooperative_call_remains_in_a_bounded_isolated_worker"
                                    if self.adapter_worker_active else None)},
                )
            reason = (state.budget.exceeded_reasons()[0].value if isinstance(exc, TimeoutError) and state.budget and state.budget.exceeded_reasons()
                      else str(exc) if str(exc) in {"model_capacity_exceeded", "tool_capacity_exceeded"}
                      else "model_timeout") if isinstance(exc, TimeoutError) else str(exc) if str(exc) in {"unsupported_transaction_claim", "simulator_disclosure_missing",
                                               "turn_limit_exceeded", "retry_limit_exceeded",
                                               "tool_call_limit_exceeded", "session_time_exceeded",
                                               "generation_length_exceeded"} else (
                next((e.code for e in validate_output(response, outcome)), "model_runtime_error")
                if 'response' in locals() else "model_runtime_error")
            return deterministic_response, False, None, reason
        return response.strip(), True, usage, None

    @property
    def adapter_worker_active(self) -> bool:
        with self._adapter_call_lock:
            return self._call_workers["model"] > 0

    def _bounded_adapter_call(self, invoke, timeout, *, capacity="model"):
        """Bound model and tool calls in independent capacity pools."""
        with self._adapter_call_lock:
            if self._call_workers[capacity] >= self._call_limits[capacity]:
                raise TimeoutError(f"{capacity}_capacity_exceeded")
            self._call_workers[capacity] += 1
        done = threading.Event()
        cancel_event = threading.Event()
        outcome: dict[str, Any] = {}

        def run():
            try:
                outcome["value"] = invoke(cancel_event)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                with self._adapter_call_lock:
                    self._call_workers[capacity] -= 1
                done.set()

        threading.Thread(target=run, name=f"rlaif-{capacity}-call", daemon=True).start()
        if not done.wait(max(0.0, timeout)):
            cancel_event.set()
            raise TimeoutError("model_timeout")
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    @staticmethod
    def _outcome_flights(outcome: TurnOutcome) -> tuple[Mapping[str, Any], ...]:
        from .output_validation import _flights
        return _flights(outcome)

    @staticmethod
    def _limit_response(reasons, state):
        reason = reasons[0].value if hasattr(reasons[0], "value") else str(reasons[0])
        label = {"turn_limit_exceeded":"turn limit", "session_time_exceeded":"session time limit",
                 "tool_call_limit_exceeded":"tool call limit", "retry_limit_exceeded":"retry limit",
                 "generation_length_exceeded":"generation length limit"}.get(reason, "execution limit")
        text = f"I can't continue this simulator request because the {label} was reached."
        return {"text":text,"adapter":state.adapter,"turn":state.turn_count,"results":[],"simulated":True,
                "model_used":False,"usage":None,"fallback_reason":reason,"validation_reason":reason}

    @staticmethod
    def _consume_tool(state):
        reasons = state.budget.consume_tool_call() if state.budget else ()
        if reasons:
            raise _BudgetExceeded(reasons)

    @staticmethod
    def _check_deadline(state):
        reasons = state.budget.exceeded_reasons() if state.budget else ()
        if reasons:
            raise _BudgetExceeded(reasons)

    def _bounded_tool_call(self, state, invoke, *, restore_shared=False):
        """Bound read-only simulator calls by the same session deadline."""
        self._check_deadline(state)
        remaining = state.budget.remaining_seconds() if state.budget else self.limits.max_session_seconds
        try:
            def isolated(_cancel):
                # Wrappers may close over the live simulator despite receiving
                # a staged copy. Capture only this transaction's owned keys;
                # whole-store restoration can erase another session's writes.
                owned = restore_shared if isinstance(restore_shared, Mapping) else None
                revision_keys = [(store_name, key)
                                 for store_name in ("holds", "bookings")
                                 for key in (owned or {}).get(store_name, ())]
                with self._store_revision_lock:
                    revisions = {key: self._store_revisions.get(key, 0)
                                 for key in revision_keys}
                before_holds = {key: deepcopy(self.simulator.holds.get(key))
                                for key in (owned or {}).get("holds", ())}
                before_bookings = {key: deepcopy(self.simulator.bookings.get(key))
                                   for key in (owned or {}).get("bookings", ())}
                try:
                    value = invoke()
                    return value
                finally:
                    if _cancel.is_set():
                        hold_keys = set(before_holds)
                        booking_keys = set(before_bookings)
                        if isinstance(locals().get("value"), Mapping):
                            if isinstance(value.get("hold_id"), str): hold_keys.add(value["hold_id"])
                            if isinstance(value.get("booking_id"), str): booking_keys.add(value["booking_id"])
                        for store, before, keys in (
                                (self.simulator.holds, before_holds, hold_keys),
                                (self.simulator.bookings, before_bookings, booking_keys)):
                            for key in keys:
                                store_name = "holds" if store is self.simulator.holds else "bookings"
                                revision_key = (store_name, key)
                                with self._store_revision_lock:
                                    unchanged = (self._store_revisions.get(revision_key, 0)
                                                 == revisions.get(revision_key, 0))
                                    if not unchanged:
                                        continue
                                    if key in before and before[key] is not None:
                                        store[key] = before[key]
                                    else:
                                        store.pop(key, None)
            value = self._bounded_adapter_call(isolated, remaining, capacity="tool")
        except TimeoutError as exc:
            if str(exc) == "tool_capacity_exceeded":
                raise _CapacityExceeded("tool") from exc
            raise _BudgetExceeded((LimitReason.SESSION_TIME,)) from exc
        self._check_deadline(state)
        return value

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
        errors = validate_output(response, outcome)
        if errors:
            raise ValueError(errors[0].code)

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
                if next_criteria.origin and next_criteria.destination: self._consume_tool(state)
                raw_search = (self._bounded_tool_call(state, lambda: self.simulator.search_flights(next_criteria))
                              if next_criteria.origin and next_criteria.destination else None)
                if next_criteria.origin and next_criteria.destination: self._check_deadline(state)
                next_search = (self._validated_search(raw_search, next_criteria)
                               if next_criteria.origin and next_criteria.destination else None)
            except Exception as exc:
                if isinstance(exc, (_BudgetExceeded, _CapacityExceeded)): raise
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
        if booking_requested and state.booking is not None:
            cached = state.booking
            prior_hold = self.simulator.holds.get(state.hold_id)
            latest_flight = (next((f for f in state.last_search.flights
                                  if int(f["flight_number"]) == state.selected_flight_id), None)
                             if state.last_search else None)
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
                self._consume_tool(state)
                fresh_booking = self._bounded_tool_call(
                    state, lambda: self.simulator.book_flight(state.hold_id),
                    restore_shared={"holds": (state.hold_id,),
                                    "bookings": (cached.get("booking_id"),)})
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
                outcome = self._booking_outcome(fresh_booking, prior_hold)
                self._check_deadline(state)
                return outcome
            except Exception as exc:
                if isinstance(exc, (_BudgetExceeded, _CapacityExceeded)): raise
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
            hold_id = "hold_" + sha256(f"{int(selected)}\0{state.passenger.strip()}".encode("utf-8")).hexdigest()[:24]
            booking_id = "booking_" + sha256(hold_id.encode("utf-8")).hexdigest()[:24]
            transaction_keys = (("holds", hold_id), ("bookings", booking_id))
            with self._store_revision_lock:
                transaction_revisions = {
                    key: self._store_revisions.get(key, 0) for key in transaction_keys
                }
            owned_revisions = None
            staged_simulator = copy(self.simulator)
            staged_simulator.holds = deepcopy(self.simulator.holds)
            staged_simulator.bookings = deepcopy(self.simulator.bookings)
            try:
                self._consume_tool(state)
                hold = self._bounded_tool_call(
                    state, lambda: staged_simulator.hold_flight(selected, state.passenger),
                    restore_shared={"holds": (hold_id,)})
                self._check_deadline(state)
                validate_hold_result(hold)
                requested_flight = next(f for f in latest.flights
                                        if int(f["flight_number"]) == selected)
                if (hold["flight"]["flight_number"] != selected
                        or dict(hold["flight"]) != dict(requested_flight)
                        or hold["passenger"] != state.passenger.strip()):
                    raise ValueError("hold does not match the selected current-search flight and passenger")
                if staged_simulator.holds.get(hold["hold_id"]) != hold:
                    raise ValueError("returned hold does not exactly match simulator storage")
                self._consume_tool(state)
                booking = self._bounded_tool_call(
                    state, lambda: staged_simulator.book_flight(hold["hold_id"]),
                    restore_shared={"holds": (hold_id,), "bookings": (booking_id,)})
                self._check_deadline(state)
                validate_booking_evidence(booking, hold)
                if (booking["flight"]["flight_number"] != selected
                        or dict(booking["flight"]) != dict(requested_flight)
                        or booking["passenger"] != state.passenger.strip()):
                    raise ValueError("booking does not match the selected current-search flight and passenger")
                if staged_simulator.bookings.get(booking["booking_id"]) != booking:
                    raise ValueError("returned booking does not exactly match simulator storage")
                if staged_simulator.holds.get(hold["hold_id"]) != hold:
                    raise ValueError("persisted hold changed during booking")
                booking_outcome = self._booking_outcome(booking, hold)
                # The outcome constructor may run injected callbacks in tests or
                # application hooks. Recheck before either persistence boundary.
                self._check_deadline(state)
                with self._store_revision_lock:
                    self.simulator.holds[hold_id] = deepcopy(hold)
                    self.simulator.bookings[booking_id] = deepcopy(booking)
                    for key in transaction_keys:
                        self._store_revisions[key] = self._store_revisions.get(key, 0) + 1
                    # These revision values identify the writes made by this
                    # transaction. Rollback may undo them only while nobody
                    # has subsequently replaced the same store entries.
                    owned_revisions = {
                        key: self._store_revisions[key] for key in transaction_keys
                    }
                self._check_deadline(state)
                state.hold_id, state.booking = hold["hold_id"], booking
                self._check_deadline(state)
            except Exception as exc:
                hold_ids = {hold_id}
                booking_ids = {booking_id}
                if isinstance(locals().get("hold"), Mapping) and isinstance(hold.get("hold_id"), str):
                    hold_ids.add(hold["hold_id"])
                if isinstance(locals().get("booking"), Mapping) and isinstance(booking.get("booking_id"), str):
                    booking_ids.add(booking["booking_id"])
                for key in hold_ids:
                    revision_key = ("holds", key)
                    with self._store_revision_lock:
                        expected_revision = (owned_revisions.get(
                            revision_key, transaction_revisions.get(revision_key, 0))
                            if owned_revisions is not None
                            else transaction_revisions.get(revision_key, 0))
                        unchanged = (self._store_revisions.get(revision_key, 0)
                                     == expected_revision)
                        if unchanged:
                            original = simulator_holds.get(key)
                            if self.simulator.holds.get(key) != original:
                                if key in simulator_holds:
                                    self.simulator.holds[key] = deepcopy(original)
                                else:
                                    self.simulator.holds.pop(key, None)
                                self._store_revisions[revision_key] = self._store_revisions.get(revision_key, 0) + 1
                for key in booking_ids:
                    revision_key = ("bookings", key)
                    with self._store_revision_lock:
                        expected_revision = (owned_revisions.get(
                            revision_key, transaction_revisions.get(revision_key, 0))
                            if owned_revisions is not None
                            else transaction_revisions.get(revision_key, 0))
                        unchanged = (self._store_revisions.get(revision_key, 0)
                                     == expected_revision)
                        if unchanged:
                            original = simulator_bookings.get(key)
                            if self.simulator.bookings.get(key) != original:
                                if key in simulator_bookings:
                                    self.simulator.bookings[key] = deepcopy(original)
                                else:
                                    self.simulator.bookings.pop(key, None)
                                self._store_revisions[revision_key] = self._store_revisions.get(revision_key, 0) + 1
                (state.selected_flight_id, state.passenger, state.hold_id, state.booking) = original_session
                if isinstance(exc, (_BudgetExceeded, _CapacityExceeded)): raise
                return make_turn_outcome(intent=Intent.REFUSE, deterministic_response="I couldn't safely complete that simulated booking. Please try again.", search_results=latest, fallback_reason="tool_error:" + type(exc).__name__)
            return booking_outcome
        if state.criteria.origin and state.criteria.destination and not state.last_search:
            try:
                self._consume_tool(state)
                search = self._validated_search(self._bounded_tool_call(
                    state, lambda: self.simulator.search_flights(state.criteria)), state.criteria)
                self._check_deadline(state)
            except Exception as exc:
                if isinstance(exc, (_BudgetExceeded, _CapacityExceeded)): raise
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
        passenger = _parse_controller_passenger(text)
        if passenger.found and passenger.name:
            text = re.sub(re.escape(passenger.name), " ", text, flags=re.I)
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

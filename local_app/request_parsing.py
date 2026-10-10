"""Immutable, read-only parsing of one flight request turn.

The result does not merge session state or perform an action. Consumers should
reject requests with conflicts before any search or booking mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .parsing import PassengerNameParse, parse_passenger_name, passenger_name_candidates


@dataclass(frozen=True)
class FlightSelection:
    """One explicit flight reference, retaining both source token and ID."""

    raw: str
    flight_id: int | None


@dataclass(frozen=True)
class ParsedRequest:
    origin: str | None = None
    destination: str | None = None
    departure_month: str | None = None
    departure_day: str | None = None
    return_month: str | None = None
    return_day: str | None = None
    departure_time: str | None = None
    return_time: str | None = None
    max_connections: int | None = None
    max_price: int | None = None
    passenger: PassengerNameParse | None = None
    flight_selections: tuple[FlightSelection, ...] = ()
    search_intent: bool = False
    selection_intent: bool = False
    booking_intent: bool = False
    provided: frozenset[str] = frozenset()
    ambiguities: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @property
    def complete_route(self) -> bool:
        return self.origin is not None and self.destination is not None

    @property
    def incomplete(self) -> bool:
        return self.search_intent and not self.complete_route

    @property
    def can_execute(self) -> bool:
        """Whether this parse has no detected contradiction or ambiguity."""
        return not self.conflicts and not self.ambiguities


_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
_DATE = re.compile(r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])\b")
_CODE = re.compile(r"\b[A-Z]{3}\b")
_BOOK_PATTERN = r"(?:book|reserve|confirm|proceed\s+with|continue\s+with|go\s+ahead\s+with)"
_SEARCH = re.compile(r"\b(?:search|find|look\s+for|show)\b", re.I)
_TIME_BUCKETS = ("morning", "afternoon", "evening", "night")


def _normalized_intent_text(text: str) -> str:
    return text.translate(str.maketrans({
        "’": "'", "‘": "'", "ʼ": "'", "–": "-", "—": "-", "−": "-",
    }))


def booking_withdrawn(text: str) -> bool:
    """Recognize explicit negation or a terminal cancellation of a booking command."""
    text = _normalized_intent_text(text)
    affirmative = re.search(rf"\b{_BOOK_PATTERN}\b", text, re.I)
    if not affirmative:
        return False
    tail = text[affirmative.end():]
    if re.search(r"(?:^|[.!?;,—-])\s*(?:no|nope|cancel|never\s+mind|nevermind)\s*[.!?;,—-]*$", tail, re.I):
        return True
    if re.search(r"\b(?:but\s+)?(?:do\s+not|don't|dont|never|should\s+not|shouldn't)\s+(?:proceed|go\s+ahead|continue|book|reserve|confirm)\b", tail, re.I):
        return True
    # Only a terminal, standalone cancellation clause withdraws authority;
    # ordinary travel references such as cancellation coverage do not.
    return bool(re.search(
        r"[,;.!?]\s*(?:(?:actually|please|just)\s+)?"
        r"cancel(?:\s+(?:(?:that|this)\s+)?(?:request|booking|reservation))?"
        r"\s*[.!?]*$", tail, re.I,
    ))


def _booking_intent(text: str) -> bool:
    """Mirror controller semantics, including explicit later cancellation."""
    text = _normalized_intent_text(text)
    modifiers = r"(?:(?:actually|immediately|ever|really|just|please|possibly|at\s+all|yet|then|right\s+now)\s+){0,8}"
    if re.search(rf"\b(?:do\s+not|don't|dont|never|should\s+not|shouldn't)\s+{modifiers}{_BOOK_PATTERN}\b", text, re.I):
        return False
    affirmative = re.search(rf"\b{_BOOK_PATTERN}\b", text, re.I)
    if affirmative and booking_withdrawn(text):
        return False
    if not affirmative:
        return False
    prefix = text[:affirmative.start()].strip()
    direct = not prefix or bool(re.fullmatch(
        r"(?:please|yes|okay|ok|go ahead|please\s+go ahead|and|then|now|i'll|i will|i want to|i'd like to|i would like to|can you|could you|would you|please\s+can you|please\s+could you)(?:\s+(?:please|now|go ahead|and|then))*",
        prefix, re.I,
    )) or bool(re.search(r"\b(?:from\s+[A-Z]{3}\s+(?:to|for)\s+[A-Z]{3}|[A-Z]{3}\s+to\s+[A-Z]{3}|under\s+\$?\d+|below\s+\$?\d+|at\s+most\s+\d+\s+connections?)\b", prefix, re.I))
    if not direct:
        return False
    if re.match(r"\s*(?:is|are|was|were)\b", text, re.I) and re.search(r"\bavailable\b|\bavailability\b", text, re.I):
        return False
    return not re.match(r"\s*(?:can|could|would)\s+i\b", text, re.I)


def _passenger(message: str) -> PassengerNameParse:
    """Apply the controller's explicit-introducer and uppercase-name rule."""
    normalized = re.sub(
        r"\bmy\s+name\s+is\b|\bbook\s+for\b|\bpassenger\b",
        lambda match: re.sub(r"\s+", " ", match.group(0)).casefold(),
        message, flags=re.I,
    )
    parsed = parse_passenger_name(normalized)
    phrase = (parsed.matched_phrase or "").casefold()
    if (parsed.status == "rejected" and parsed.reason == "airport_code_or_acronym"
            and phrase in {"my name is", "book for", "passenger", "for"}
            and parsed.candidate
            and (phrase != "for" or len(parsed.candidate.split()) >= 2)
            and parsed.candidate == parsed.candidate.upper()):
        parsed = parse_passenger_name("passenger " + parsed.candidate.title())
    return parsed


def _selections(text: str) -> tuple[FlightSelection, ...]:
    text = _normalized_intent_text(text)
    if re.fullmatch(r"\s*[+-]?\d+\s*", text):
        raw = text.strip()
        return (FlightSelection(raw, int(raw) if re.fullmatch(r"\d{3,5}", raw) else None),)
    if re.match(r"\s*(?:is|are|was|were)\b", text, re.I) and re.search(r"\bavailable\b", text, re.I):
        return ()
    refs = re.findall(r"\bflight\s+(?:number\s+|#\s*)([^\s,;!?]+)|\bflight\s+([^\s,;!?]+)", text, re.I)
    raws = [next(part for part in pair if part) for pair in refs]
    raws = [raw for raw in raws if raw.casefold() not in {"from", "to"}]
    if not raws:
        command = re.search(r"\b(?:book|reserve|confirm|select|choose|proceed\s+with|continue\s+with|go\s+ahead\s+with)\s+(?:the\s+|a\s+|this\s+|that\s+)?([^\s,;!?]+)", text, re.I)
        if command:
            token = command.group(1)
            if token.casefold() not in {"it", "this", "that", "the", "a", "my", "our", "for", "from", "to", "booking", "reservation", "flight"}:
                raws.append(token)
    entries = []
    for token in raws:
        raw = token[:-1] if re.fullmatch(r"\d+\.", token) else token
        flight_id = int(raw) if re.fullmatch(r"\d{3,5}", raw) else None
        entries.append(FlightSelection(raw, flight_id))
    return tuple(entries)


def parse_request(message: str) -> ParsedRequest:
    """Extract search, selection, and booking facts from a single user turn."""
    if not isinstance(message, str) or not message.strip():
        return ParsedRequest(ambiguities=("empty_request",))

    passenger = _passenger(message)
    introduced_passengers = passenger_name_candidates(message)
    scan_text = message
    if passenger.found and passenger.name:
        scan_text = re.sub(re.escape(passenger.name), " ", scan_text, flags=re.I)

    route_patterns = (
        re.compile(r"\b(?i:from)\s+([A-Z]{3})\s+(?i:to|for)\s+([A-Z]{3})\b"),
        re.compile(r"\b([A-Z]{3})\s+(?i:to)\s+([A-Z]{3})\b"),
    )
    route_matches = [match for pattern in route_patterns for match in pattern.finditer(scan_text)]
    route = min(route_matches, key=lambda match: match.start()) if route_matches else None
    codes = _CODE.findall(scan_text)
    if route:
        origin, destination = route.group(1), route.group(2)
    else:
        origin, destination = (codes + [None, None])[:2]
        from_code = re.search(r"\b(?i:from)\s+([A-Z]{3})\b", scan_text)
        to_code = re.search(r"\b(?i:to)\s+([A-Z]{3})\b", scan_text)
        if from_code:
            origin = from_code.group(1)
        if to_code:
            destination = to_code.group(1)

    ambiguities: list[str] = []
    conflicts: list[str] = []
    route_pairs = {(match.group(1), match.group(2)) for match in route_matches}
    if len(route_pairs) > 1:
        conflicts.append("conflicting_routes")
    named = {candidate.name for candidate in introduced_passengers if candidate.found}
    if len(named) > 1:
        conflicts.append("conflicting_passenger_names")
    if len(set(codes)) > 2:
        conflicts.append("multiple_airport_codes")
    dates = list(_DATE.finditer(scan_text))
    normalized_dates = [(_MONTHS[int(match.group(1)) - 1], str(int(match.group(2)))) for match in dates]
    departure = returning = (None, None)
    unlabelled = []
    for match, date in zip(dates, normalized_dates):
        context = scan_text[max(scan_text.rfind(mark, 0, match.start()) for mark in ".!?;") + 1:match.start()]
        labels = list(re.finditer(r"\b(?:return|returning|back|depart\w*|outbound|leave)\b", context, re.I))
        label = labels[-1].group(0).casefold() if labels else ""
        if label in {"return", "returning", "back"}:
            if returning != (None, None) and returning != date:
                conflicts.append("multiple_return_dates")
            returning = date
        elif label.startswith("depart") or label in {"outbound", "leave"}:
            if departure != (None, None) and departure != date:
                conflicts.append("multiple_departure_dates")
            departure = date
        else:
            unlabelled.append(date)
    if departure == (None, None) and unlabelled:
        departure = unlabelled.pop(0)
    if returning == (None, None) and unlabelled:
        returning = unlabelled.pop(0)
    if len(set(normalized_dates)) > 2:
        conflicts.append("multiple_date_pairs")

    prices = [int(match.group(1)) for match in re.finditer(
        r"(?:under|below|max(?:imum)?|less\s+than|up\s+to)\s*\$?\s*(\d+)", message, re.I)]
    limits = [int(match.group(1)) for match in re.finditer(r"at\s+most\s+(\d+)\s+connections?", message, re.I)]
    if re.search(r"\b(?:nonstop|direct)\b", message, re.I):
        limits.append(0)
    if len(set(prices)) > 1:
        conflicts.append("conflicting_max_price")
    if len(set(limits)) > 1:
        conflicts.append("conflicting_max_connections")

    departure_time = return_time = None
    for bucket in _TIME_BUCKETS:
        for match in re.finditer(rf"\b{bucket}\b", message, re.I):
            prefix = message[max(message.rfind(mark, 0, match.start()) for mark in ".!?;") + 1:match.start()]
            if re.search(r"\b(?:return|arriv\w*)\b", prefix, re.I):
                if return_time and return_time != bucket:
                    conflicts.append("conflicting_return_time")
                return_time = bucket
            elif re.search(r"\b(?:depart\w*|leave|outbound)\b", prefix, re.I):
                if departure_time and departure_time != bucket:
                    conflicts.append("conflicting_departure_time")
                departure_time = bucket

    selections = _selections(message)
    selection_intent = bool(selections) or bool(re.search(
        rf"\b(?:select|choose|{_BOOK_PATTERN})\b", message, re.I,
    ))
    if len({(item.raw, item.flight_id) for item in selections}) > 1:
        conflicts.append("conflicting_flight_selections")
    booking = _booking_intent(message)
    explicit_search = bool(_SEARCH.search(message))
    has_search_criteria = bool(origin or destination or dates or prices or limits or departure_time or return_time)
    search = explicit_search or (has_search_criteria and not booking and not selection_intent)

    provided = set()
    fields = {
        "origin": origin, "destination": destination,
        "departure_date": departure[0] if departure[0] is not None else None,
        "return_date": returning[0] if returning[0] is not None else None,
        "departure_time": departure_time, "return_time": return_time,
        "max_price": prices[0] if prices else None,
        "max_connections": limits[0] if limits else None,
        "passenger": passenger.name if passenger.found else None,
        "flight_selections": selections or None,
    }
    provided.update(name for name, value in fields.items() if value is not None)
    if search:
        provided.add("search_intent")
    if selection_intent:
        provided.add("selection_intent")
    if booking:
        provided.add("booking_intent")
    if search and (origin is None or destination is None):
        ambiguities.append("incomplete_route")
    if passenger.status == "rejected":
        ambiguities.append("passenger_unusable")
    elif passenger.status == "not_found" and booking:
        ambiguities.append("missing_passenger")

    return ParsedRequest(
        origin=origin, destination=destination,
        departure_month=departure[0], departure_day=departure[1],
        return_month=returning[0], return_day=returning[1],
        departure_time=departure_time, return_time=return_time,
        max_connections=limits[0] if limits else None,
        max_price=prices[0] if prices else None,
        passenger=passenger, flight_selections=selections,
        search_intent=search, selection_intent=selection_intent,
        booking_intent=booking, provided=frozenset(provided),
        ambiguities=tuple(dict.fromkeys(ambiguities)),
        conflicts=tuple(dict.fromkeys(conflicts)),
    )


__all__ = ["FlightSelection", "ParsedRequest", "parse_request"]

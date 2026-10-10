"""Deterministic validation of generated text against a :class:`TurnOutcome`.

The module is intentionally independent of the controller so callers can use
the same machine-readable checks in generation, audits, and offline evaluation.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .guardrails import detect_transaction_claim
from .outcomes import TurnOutcome

MAX_OUTPUT_LENGTH = 4000
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>|</?tool_call\b[^>]*>|\[/?(?:INST|SYS|ASSISTANT|USER|TOOL)[^\]]*\]", re.I)
_STATE_RE = re.compile(
    r"(?i)(?:sessionstate\s*\(|\b(?:selected_flight_id|last_search|hold_id|booking_id|"
    r"booking|state)\s*[:=])"
)
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_PRICE_RE = re.compile(r"(?:\$\s*|\bUSD\s*)(\d+(?:,\d{3})*(?:\.\d{1,2})?)|\b(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:dollars?|USD)\b", re.I)
_PRICE_WORDS_RE = re.compile(r"\b((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and|[- ]){2,})\s+dollars?\b", re.I)
_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_AIRLINE_FOR_FLIGHT_RE = re.compile(
    r"\b([A-Z][A-Za-z&'-]*(?:\s+(?!for\b)[A-Z][A-Za-z&'-]*){0,2})\s+(?:for\s+)?flight\s+(\d{4,5})\b", re.I
)
_FLIGHT_ON_AIRLINE_RE = re.compile(
    r"\bflight\s+(\d{4,5})\s+(?:(?:is|was)\s+)?(?:on|with|via|operated\s+by)\s+"
    r"([A-Z][A-Za-z&'-]*(?:\s+[A-Z][A-Za-z&'-]*){0,2})\b", re.I
)
_NON_AIRLINE_LEADS = {"please", "choose", "select", "book", "reserve", "take", "catch", "find", "search", "available", "scheduled"}
_AIRLINE_BRANDS = {
    "delta", "frontier", "united", "american", "southwest", "alaska", "jetblue", "spirit",
    "hawaiian", "allegiant", "sun country", "qatar", "emirates", "lufthansa", "air france",
    "british airways", "air canada", "aer lingus", "ryanair", "easyjet", "turkish airlines",
}


@dataclass(frozen=True)
class ValidationError:
    """Stable issue record suitable for JSON serialization."""

    code: str
    message: str
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            result["details"] = dict(self.details)
        return result


class OutputValidationException(ValueError):
    def __init__(self, errors: tuple[ValidationError, ...]):
        self.errors = errors
        super().__init__(errors[0].code if errors else "output_invalid")


def _flights(outcome: TurnOutcome) -> tuple[dict[str, Any], ...]:
    if outcome.booking_result is not None:
        return (outcome.booking_result["flight"],)
    if outcome.hold_result is not None:
        return (outcome.hold_result["flight"],)
    if outcome.search_results is not None:
        return tuple(outcome.search_results.flights[:5])
    return ()


def _issue(code: str, message: str, **details: Any) -> ValidationError:
    return ValidationError(code, message, details or None)


def _clauses(text: str) -> list[str]:
    """Split coordinated prose so a negation in one claim cannot mask another."""
    return [part.strip() for part in re.split(r"[.!?;\n,]+|\b(?:and|but|however|yet|despite|although|though)\b", text, flags=re.I) if part.strip()]


def _affirmative_clause(clause: str, pattern: str) -> bool:
    """Check a claim and only its own governing negation."""
    for match in re.finditer(pattern, clause, re.I):
        prefix = clause[max(0, match.start() - 48):match.start()]
        claim = match.group(0)
        # A negation after the matched status belongs to that status ("ticket
        # not allocated"); a negation elsewhere in the clause does not.
        negated = bool(re.search(
            r"\b(?:not|never|isn't|isnt|wasn't|wasnt|hasn't|hasnt|haven't|havent|didn't|didnt)\b"
            r"(?:\s+\w+){0,4}\s*$", prefix, re.I
        ) or re.search(
            r"^\s*\b(?:not|never|isn't|isnt|wasn't|wasnt|hasn't|hasnt|haven't|havent|didn't|didnt)\b"
            r"(?:\s+\w+){0,4}\b", clause[match.end():], re.I
        ) or re.search(
            r"\b(?:not|never|isn't|isnt|wasn't|wasnt|hasn't|hasnt|haven't|havent|didn't|didnt)\b",
            claim, re.I
        ))
        if not negated and not re.search(r"\b(?:not|never|isn't|isnt|wasn't|wasnt|hasn't|hasnt|haven't|havent|didn't|didnt)\b(?:\s+\w+){0,3}\s*$", prefix, re.I):
            return True
    return False


def validate_output(text: Any, outcome: TurnOutcome) -> tuple[ValidationError, ...]:
    """Return all deterministic validation errors for text grounded in outcome."""
    if not isinstance(text, str):
        return (_issue("malformed_output", "Output must be text", actual_type=type(text).__name__),)
    if not text.strip():
        return (_issue("empty_output", "Output is empty"),)
    errors: list[ValidationError] = []
    if not "".join(ch for ch in text if not ch.isspace() and __import__("unicodedata").category(ch) not in {"Cf", "Cc", "Cs"}):
        errors.append(_issue("empty_output", "Output has no visible text"))
    if len(text) > MAX_OUTPUT_LENGTH:
        errors.append(_issue("output_too_long", "Output exceeds the character limit",
                             length=len(text), limit=MAX_OUTPUT_LENGTH))

    stripped = text.strip()
    if "```" in stripped:
        if re.match(r"(?s)^```(?:json)?\s*[\[{]", stripped, re.I):
            errors.append(_issue("code_fenced_json", "Output contains code-fenced JSON"))
        else:
            errors.append(_issue("code_fence", "Output contains a code fence"))
    else:
        try:
            import json
            raw = json.loads(stripped)
        except (ValueError, TypeError):
            raw = None
            if stripped.startswith(("{", "[")):
                errors.append(_issue("malformed_json", "Output looks like malformed JSON"))
        if isinstance(raw, (dict, list)):
            errors.append(_issue("raw_json", "Output is raw JSON rather than user-facing text"))
    if _STATE_RE.search(stripped):
        errors.append(_issue("internal_state", "Output exposes internal state"))
    tokens = sorted(set(_SPECIAL_TOKEN_RE.findall(text)))
    if tokens:
        errors.append(_issue("control_token", "Output contains internal control tokens", tokens=tokens))

    _check_repetition(text, errors)
    _check_grounding(text, outcome, errors)
    return tuple(errors)


def validate_or_raise(text: Any, outcome: TurnOutcome) -> None:
    """Raise with structured errors when ``text`` fails validation."""
    errors = validate_output(text, outcome)
    if errors:
        raise OutputValidationException(errors)


def _check_repetition(text: str, errors: list[ValidationError]) -> None:
    sentences = [re.sub(r"\s+", " ", s).strip(" .!?\t").casefold()
                 for s in _SENTENCE_RE.split(text) if s.strip()]
    flat = re.sub(r"[.!?\s]+", " ", text).strip().casefold()
    flat_words = _WORD_RE.findall(flat)
    # Detect loops inside one prose segment. Keeping punctuation boundaries
    # avoids treating repeated fields in structured multi-flight summaries as loops.
    segments = [re.sub(r"\s+", " ", part).strip().casefold()
                for part in re.split(r"[.!?;\n]+", text) if part.strip()]
    short_loop = False
    for segment in segments:
        words = [word for word in _WORD_RE.findall(segment) if not word.isdigit()]
        for size in (1, 2, 3):
            if any(words[i:i + size] == words[i + size:i + size * 2] == words[i + size * 2:i + size * 3]
                   for i in range(max(0, len(words) - size * 3 + 1))):
                short_loop = True
                break
        if short_loop:
            break
    if short_loop:
        errors.append(_issue("repeated_sentence", "Output repeats a short phrase loop"))
    duplicates = sorted(sentence for sentence, count in Counter(sentences).items()
                        if count > 1 and len(_WORD_RE.findall(sentence)) >= 1)
    if duplicates:
        errors.append(_issue("repeated_sentence", "Output repeats a complete sentence", sentences=duplicates))

    lines = [re.sub(r"\s+", " ", line).strip().casefold()
             for line in text.splitlines() if line.strip()]
    duplicate_lines = sorted(line for line, count in Counter(lines).items() if count > 1 and len(line) >= 1)
    if duplicate_lines:
        errors.append(_issue("duplicate_lines", "Output contains duplicate lines", lines=duplicate_lines))

    # Mask flight identifiers, and count repeated 5-grams within a prose
    # segment. Repeated itinerary fields across separate flight summaries are
    # expected and must not be classified as a generation loop.
    repeated = []
    for segment in segments:
        masked = re.sub(r"\b\d{4,5}\b", "flightid", segment.casefold())
        words = [word.casefold() for word in _WORD_RE.findall(masked)]
        ngrams = [tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
        repeated.extend(gram for gram, count in Counter(ngrams).items() if count >= 3)
    if repeated:
        errors.append(_issue("repeated_ngram", "Output repeats a five-word phrase", ngrams=[" ".join(g) for g in repeated[:5]]))

    # Repeating the opening at the end is a common generation loop. Compare
    # sentence content after punctuation/whitespace normalization.
    if len(sentences) >= 3 and sentences[0] == sentences[-1] and len(_WORD_RE.findall(sentences[0])) >= 4:
        errors.append(_issue("closing_loop", "Output closes by repeating its opening sentence"))


def _check_grounding(text: str, outcome: TurnOutcome, errors: list[ValidationError]) -> None:
    lowered = text.casefold()
    flights = _flights(outcome)
    known_ids = {str(f.get("flight_number")) for f in flights}
    mentioned_ids = set(re.findall(r"\b\d{4,5}\b", text))
    unknown_ids = sorted(mentioned_ids - known_ids)
    if unknown_ids:
        errors.append(_issue("unsupported_flight_id", "Output mentions flight IDs outside the outcome", flight_ids=unknown_ids))

    known_prices = set()
    for flight in flights:
        if flight.get("price") is not None:
            try: known_prices.add(Decimal(str(flight["price"])))
            except InvalidOperation: pass
    mentioned_prices = set()
    for match in _PRICE_RE.findall(text):
        try: mentioned_prices.add(Decimal(next(value for value in match if value).replace(",", "")))
        except InvalidOperation: pass
    for match in _PRICE_WORDS_RE.finditer(text):
        total = current = 0
        for word in re.findall(r"[a-z]+", match.group(1).casefold()):
            if word == "and":
                continue
            if word == "hundred":
                current = max(current, 1) * 100
            elif word == "thousand":
                total += max(current, 1) * 1000
                current = 0
            else:
                current += _NUMBER_WORDS[word]
        mentioned_prices.add(Decimal(total + current))
    wrong_prices = sorted((str(value) for value in mentioned_prices - known_prices),
                          key=lambda value: (len(value), value))
    if wrong_prices:
        errors.append(_issue("unsupported_price", "Output mentions prices outside the outcome", prices=wrong_prices))

    _check_flight_associations(text, flights, errors)

    empty = outcome.search_results is not None and not outcome.search_results.flights
    no_results = bool(re.search(r"\b(?:no (?:available )?flights?|zero flights?|not any flights?|nothing matches|nothing is available|all flights? (?:are )?unavailable|none of the flights? (?:are )?(?:available|unavailable))\b|couldn['’]?t find (?:any )?(?:matching )?flights?", lowered))
    clauses = _clauses(text)
    unavailable_claim = any(
        bool(re.search(r"\b(?:not\s+available|isn't\s+available|isnt\s+available)\b", clause, re.I))
        or _affirmative_clause(clause, r"\b(?:unavailable|sold\s+out|fully\s+booked)\b")
        for clause in clauses
    )
    positive = bool(re.search(r"\b(?:found|have|are|showing|available|options include)\s+(?:\w+\s+){0,2}flights?\b|\b\d+\s+flights?\b|\bflight\s+\d{3,5}\b", lowered))
    if flights and (no_results or unavailable_claim):
        errors.append(_issue("availability_contradiction", "Output denies flights present in the outcome"))
    elif empty and positive:
        errors.append(_issue("availability_contradiction", "Output claims flights when the outcome is empty"))

    payment_pattern = r"\b(?:payment|card)\b[^.!?]*\b(?:charged|processed|paid)\b|\b(?:charged|processed)\b[^.!?]*\b(?:payment|card)\b"
    payment_claim = any(_affirmative_clause(clause, payment_pattern) for clause in clauses)
    real_claim = any(_affirmative_clause(clause, r"\b(?:real reservation|airline confirmation number)\b") for clause in clauses)
    if payment_claim or real_claim:
        errors.append(_issue("unsupported_real_transaction", "Output asserts an unsupported real transaction"))
    says_simulated = any(_affirmative_clause(clause, r"\bsimulat\w*\b") for clause in clauses)
    transaction_status = r"(?:in place|held|hold|reserved|secured|confirmed|booked|active|guaranteed)"
    reservation_pattern = rf"\b(?:reservation|booking|seat|ticket)\b[^.!?]*\b(?:{transaction_status}|allocated|valid|remains\s+valid)\b|\b(?:ticket|seat)\b[^.!?]*\bready\b[^.!?]*\b(?:travel|flight|trip|fly)\b"
    reservation_claim = any(_affirmative_clause(clause, reservation_pattern) for clause in clauses)
    if (detect_transaction_claim(text) or reservation_claim or payment_claim
            or re.search(r"\b(?:made|completed|booked)\s+(?:your\s+)?(?:reservation|booking)\b", lowered)):
        booking = outcome.booking_result
        if booking is None:
            errors.append(_issue("unsupported_transaction_claim", "Output claims a completed transaction without booking evidence"))
        elif not says_simulated:
            errors.append(_issue("simulator_disclosure_missing", "Booking claim must disclose that it is simulated"))
        else:
            claims = outcome.authoritative_claims
            if (claims.booking_id != booking.get("booking_id") or claims.hold_id != booking.get("hold_id")
                    or claims.selected_flight_id != booking.get("flight", {}).get("flight_number")
                    or claims.booking_status != booking.get("status") or claims.passenger != booking.get("passenger")
                    or claims.simulated is not True or booking.get("simulated") is not True):
                errors.append(_issue("transaction_evidence_mismatch", "Booking fields do not match authoritative simulator evidence"))
            for field in ("booking_id", "hold_id"):
                value = booking.get(field)
                if value:
                    candidates = re.findall(rf"\b{field}\s*[:#]?\s*([\w-]+)", text, re.I)
                    if candidates and any(candidate.casefold() != str(value).casefold() for candidate in candidates):
                        errors.append(_issue("transaction_field_contradiction", "Output states a conflicting transaction identifier", field=field))
    elif outcome.simulator_disclosure_required and not says_simulated:
        errors.append(_issue("simulator_disclosure_missing", "Transaction outcome must disclose the simulator"))


def _check_flight_associations(text: str, flights: tuple[dict[str, Any], ...],
                               errors: list[ValidationError]) -> None:
    """Bind itinerary facts to the nearest explicitly named flight."""
    if not flights:
        return
    by_id = {str(flight.get("flight_number")): flight for flight in flights}
    brand_catalog = set(_AIRLINE_BRANDS)
    for item in flights:
        airline = re.sub(r"\s+(?:airlines?|airways?)$", "", str(item.get("airline", "")), flags=re.I).casefold()
        if airline:
            brand_catalog.add(airline)
    ids = list(re.finditer(r"\b\d{4,5}\b", text))
    issues: dict[str, set[str]] = {"unsupported_airline": set(), "unsupported_price": set(),
                                   "unsupported_route": set(), "unsupported_stops": set()}

    def nearest(position: int):
        left = max((text.rfind(mark, 0, position) for mark in ";\n"), default=-1) + 1
        right_candidates = [index for mark in ";\n" if (index := text.find(mark, position)) >= 0]
        right = min(right_candidates, default=len(text))
        local_ids = [item for item in ids if left <= item.start() < right]
        pool = local_ids or ids
        match = min(pool, key=lambda item: abs(item.start() - position)) if pool else None
        return by_id.get(match.group()) if match else None

    for match in re.finditer(r"\b([A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*){0,3}\s+(?:Airlines?|Airways?))\b", text, re.I):
        if match.group(1).casefold().startswith(("is on ", "was on ", "runs on ", "operates on ")):
            continue
        flight = nearest(match.start())
        known = {str(item.get("airline", "")).casefold() for item in flights}
        if match.group(1).casefold() not in known:
            issues["unsupported_airline"].add(match.group(1))
        if flight and match.group(1).casefold() != str(flight.get("airline", "")).casefold():
            issues["unsupported_airline"].add(match.group(1))
    for match in _AIRLINE_FOR_FLIGHT_RE.finditer(text):
        airline_phrase, flight_id = match.groups()
        if airline_phrase.casefold().split()[0] in _NON_AIRLINE_LEADS:
            continue
        flight = by_id.get(flight_id)
        if not flight:
            continue
        claimed = re.sub(r"\s+(?:airlines?|airways?)$", "", airline_phrase, flags=re.I).casefold()
        if claimed not in brand_catalog:
            continue
        actual = re.sub(r"\s+(?:airlines?|airways?)$", "", str(flight.get("airline", "")), flags=re.I).casefold()
        if claimed != actual:
            issues["unsupported_airline"].add(airline_phrase)
    for match in re.finditer(r"\bflight\s+(\d{4,5})\b[^.!?\n]{0,24}?\b(?:on|with|via)\s+([A-Z][A-Za-z&'-]*)\b", text, re.I):
        flight_id, airline_phrase = match.groups()
        flight = by_id.get(flight_id)
        if not flight:
            continue
        claimed = re.sub(r"\s+(?:airlines?|airways?)$", "", airline_phrase, flags=re.I).casefold()
        if claimed not in brand_catalog:
            continue
        actual = re.sub(r"\s+(?:airlines?|airways?)$", "", str(flight.get("airline", "")), flags=re.I).casefold()
        if claimed != actual:
            issues["unsupported_airline"].add(airline_phrase)
    for match in re.finditer(r"\b([A-Z][A-Za-z&'-]*(?:\s+[A-Z][A-Za-z&'-]*){0,2})\s+(?:operates|runs)\s+(?:flight\s+)?(\d{4,5})\b", text, re.I):
        airline_phrase, flight_id = match.groups()
        flight = by_id.get(flight_id)
        if flight and airline_phrase.casefold() in brand_catalog:
            actual = re.sub(r"\s+(?:airlines?|airways?)$", "", str(flight.get("airline", "")), flags=re.I).casefold()
            if airline_phrase.casefold() != actual:
                issues["unsupported_airline"].add(airline_phrase)
    price_claims: list[tuple[int, Decimal, str]] = []
    for match in _PRICE_RE.finditer(text):
        amount = next(value for value in match.groups() if value)
        try:
            price_claims.append((match.start(), Decimal(amount.replace(",", "")), amount))
        except InvalidOperation:
            pass
    for match in _PRICE_WORDS_RE.finditer(text):
        total = current = 0
        for word in re.findall(r"[a-z]+", match.group(1).casefold()):
            if word == "and": continue
            if word == "hundred": current = max(current, 1) * 100
            elif word == "thousand": total += max(current, 1) * 1000; current = 0
            else: current += _NUMBER_WORDS[word]
        price_claims.append((match.start(), Decimal(total + current), match.group(0)))
    for position, amount_value, amount in price_claims:
        flight = nearest(position)
        if flight:
            try:
                if amount_value != Decimal(str(flight.get("price"))):
                    issues["unsupported_price"].add(amount)
            except InvalidOperation:
                issues["unsupported_price"].add(amount)
    route_patterns = (
        re.compile(r"\bfrom\s+([A-Z]{3})\s+to\s+([A-Z]{3})\b"),
        re.compile(r"\b([A-Z]{3})\s*(?:-|/|→)\s*([A-Z]{3})\b"),
        re.compile(r"\b([A-Z]{3})\s+to\s+([A-Z]{3})\b"),
    )
    for pattern in route_patterns:
        for match in re.compile(pattern.pattern, pattern.flags | re.I).finditer(text):
            flight = nearest(match.start())
            if flight and (match.group(1).upper() != str(flight.get("departure_airport", "")).upper()
                           or match.group(2).upper() != str(flight.get("return_airport", "")).upper()):
                issues["unsupported_route"].add(f"{match.group(1)}-{match.group(2)}")
    number_words = {"zero": 0, "one": 1, "a": 1, "an": 1, "pair": 2, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    for match in re.finditer(r"\b(nonstop|non-stop|direct)\b|\b(\d+|zero|one|a|an|pair|two|three|four|five|six|seven|eight|nine|ten)\s+(?:of\s+)?(stops?|connections?)\b", text, re.I):
        flight = nearest(match.start())
        if not flight:
            continue
        claim = (match.group(1) or match.group(2)).casefold()
        expected = int(flight.get("num_connections", -1))
        if claim in {"nonstop", "non-stop", "direct"} and expected != 0:
            issues["unsupported_stops"].add(match.group(1))
        elif match.group(3) and (int(claim) if claim.isdigit() else number_words.get(claim)) != expected:
            issues["unsupported_stops"].add(match.group(1))
    # Dates and clock times are meaningful only when the corresponding field
    # exists in the simulator record. Bind claims to nearby flight IDs, as with
    # route, price, and airline facts above.
    months = {name.casefold(): index for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1)}
    months.update({name[:3].casefold(): index for index, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1)})
    month_pattern = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    date_re = re.compile(rf"\b({month_pattern})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)

    def fact_role(position: int) -> str:
        boundaries = []
        for boundary in re.finditer(r"[.!?;\n]", text):
            index = boundary.start()
            if text[index] == "." and re.search(
                r"\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)$",
                text[max(0, index - 5):index], re.I
            ):
                continue
            boundaries.append(index)
        start = max((index + 1 for index in boundaries if index < position), default=0)
        end = min((index for index in boundaries if index >= position), default=len(text))
        sentence = text[start:end]
        choices = [(match.start(),
                    "return" if match.group().casefold().startswith("return") else "departure")
                   for match in re.finditer(r"\b(?:return\w*|depart\w*|leave\w*|outbound)\b", sentence[:position - start], re.I)]
        if choices:
            return choices[-1][1]
        return "departure"

    for match in date_re.finditer(text):
        flight = nearest(match.start())
        returning = fact_role(match.start()) == "return"
        month_field, day_field = (("return_month", "return_day") if returning else ("departure_month", "departure_day"))
        if flight and flight.get(month_field) and flight.get(day_field):
            if (months.get(match.group(1).casefold()[:3], -1) != months.get(str(flight[month_field]).casefold()[:3], -1)
                    or int(match.group(2)) != int(flight[day_field])):
                issues["unsupported_date"] = issues.get("unsupported_date", set()) | {match.group(0)}
    time_re = re.compile(r"\bnoon\b|\bmidnight\b|\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b|\b([01]?\d|2[0-3]):([0-5]\d)\b", re.I)
    for match in time_re.finditer(text):
        flight = nearest(match.start())
        returning = fact_role(match.start()) == "return"
        time_field = "return_time_num" if returning and flight and flight.get("return_time_num") is not None else "departure_time_num"
        if flight and flight.get(time_field) is not None:
            if match.group(0).casefold() == "noon":
                hour, minute = 12, 0
            elif match.group(0).casefold() == "midnight":
                hour, minute = 0, 0
            elif match.group(4):
                hour, minute = int(match.group(4)), int(match.group(5))
            else:
                hour = int(match.group(1)) % 12 + (12 if match.group(3).casefold().startswith("p") else 0)
                minute = int(match.group(2) or 0)
            expected = flight[time_field]
            try:
                expected_hour, expected_minute = (int(part) for part in str(expected).split(":")) if ":" in str(expected) else (int(expected), 0)
            except (TypeError, ValueError):
                expected_hour, expected_minute = int(expected), 0
            code = "unsupported_return_time" if time_field == "return_time_num" else "unsupported_departure_time"
            if (hour, minute) != (expected_hour, expected_minute):
                issues[code] = issues.get(code, set()) | {match.group(0)}
    messages = {
        "unsupported_airline": "Output associates a flight with the wrong airline",
        "unsupported_price": "Output associates a flight with the wrong price",
        "unsupported_route": "Output associates a flight with the wrong route",
        "unsupported_stops": "Output associates a flight with the wrong stop count",
        "unsupported_date": "Output associates a flight with the wrong departure date",
        "unsupported_departure_time": "Output associates a flight with the wrong departure time",
        "unsupported_return_time": "Output associates a flight with the wrong return time",
    }
    for code, values in issues.items():
        if values:
            errors.append(_issue(code, messages[code], claims=sorted(values)))

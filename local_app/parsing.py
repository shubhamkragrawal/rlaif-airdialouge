"""Conservative passenger-name extraction for flight requests.

This module is intentionally independent of the conversation controller so it
can be integrated without coupling parsing policy to booking state.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


MAX_NAME_LENGTH = 100

# These commonly occur as airport identifiers or instructions, and should not
# become passenger names just because they appear after a name introducer.
_NON_NAMES = {
    "book", "booking", "cancel", "change", "confirm", "flight", "flights",
    "help", "hold", "lookup", "reserve", "reservation", "search", "ticket",
    "travel", "traveling", "please", "now", "next", "week", "month", "today",
    "tomorrow", "me", "myself", "passenger",
    "jfk", "lax", "iad", "las", "sfo", "ord", "lhr", "cdg", "dfw",
    "atl", "den", "sea", "mia", "bos", "ewr", "lga", "dca", "phx",
}
_TAIL = re.compile(
    r"\s+(?:for|to|from|on|with|and|but|please|search|find|book|booking|"
    r"flight|flights|ticket|tickets|tomorrow|today|next|week|month|"
    r"depart\w*|return\w*|outbound|under|below|maximum|max|less|"
    r"nonstop|direct|connections?|at\s+most|up\s+to)\b.*$",
    re.IGNORECASE,
)
_INTRODUCER = re.compile(r"\b(?:my\s+name\s+is|book\s+for|passenger)\b\s*", re.I)
_CONTEXT_FOR = re.compile(r"\bfor\b\s*", re.I)
_TRAVELING = re.compile(
    r"\b(?:passengers?\s+)?"
    r"(?P<names>[\w\u0300-\u036f\u0900-\u097fÀ-ÖØ-öø-ÿĀ-ž'’ʼ-]+(?:\s+[\w\u0300-\u036f\u0900-\u097fÀ-ÖØ-öø-ÿĀ-ž'’ʼ-]+){0,10})"
    r"\s+(?:is|are|will\s+be)\s+travel(?:ing|ling)\b", re.I,
)
_NAME_CONJUNCTION = re.compile(r"\s+and\s+", re.I)
_NAME_TOKEN = re.compile(r"[^'’-]+(?:['’-][^'’-]+)*")


def _valid_name_token(token: str) -> bool:
    if not _NAME_TOKEN.fullmatch(token):
        return False
    parts = re.split(r"['’-]", token)
    return all(part and unicodedata.category(part[0]).startswith("L")
               and all(unicodedata.category(char)[0] == "L"
                       or (unicodedata.category(char)[0] == "M" and ord(char) < 0x0900)
                       for char in part)
               for part in parts)


def _normalized_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Build a normalized matching view whose characters point to source spans."""
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    punctuation = str.maketrans({
        "’": "'", "‘": "'", "ʼ": "'", "‐": "-", "‑": "-",
        "‒": "-", "–": "-", "—": "-", "−": "-",
    })
    while index < len(text):
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        cluster = unicodedata.normalize("NFKC", text[index:end]).translate(punctuation)
        for char in cluster:
            chars.append(char)
            spans.append((index, end))
        index = end
    return "".join(chars), spans


def _source_slice(text: str, spans: list[tuple[int, int]], start: int, end: int) -> str:
    if start >= end or not spans:
        return ""
    return text[spans[start][0]:spans[end - 1][1]]


@dataclass(frozen=True)
class PassengerNameParse:
    """Structured outcome of parsing a passenger name from one message."""

    status: str  # "found", "not_found", or "rejected"
    name: str | None = None
    candidate: str | None = None
    matched_phrase: str | None = None
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.status == "found"


def normalize_name_text(value: str) -> str:
    """Normalize compatibility forms and typographic name punctuation."""
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(str.maketrans({
        "’": "'", "‘": "'", "ʼ": "'", "‐": "-", "‑": "-",
        "‒": "-", "–": "-", "—": "-", "−": "-",
    }))
    return " ".join(value.strip().strip(" \t\r\n.,;:!?\"“”()[]{}<>_").split())


def _validate(candidate: str) -> tuple[str | None, str | None]:
    candidate = re.split(r"[,;.!?]", candidate, maxsplit=1)[0]
    candidate = normalize_name_text(candidate)
    candidate = _TAIL.sub("", candidate).strip(" \t.,;:!?\"“”()[]{}<>")
    candidate = " ".join(candidate.split())
    if not candidate:
        return None, "empty_candidate"
    if len(candidate) > MAX_NAME_LENGTH:
        return None, "name_too_long"
    if len(candidate) < 2:
        return None, "too_short"
    tokens = candidate.split()
    if len(tokens) > 6:
        return None, "too_many_name_parts"
    if any(token.casefold() in _NON_NAMES for token in tokens):
        return None, "command_or_airport_code"
    if any(len(token) > 5 and len(set(token.casefold())) == 1 for token in tokens):
        return None, "invalid_repeated_character"
    if not all(_valid_name_token(token) for token in tokens):
        return None, "unsafe_characters"
    # Reject all-uppercase short tokens such as JFK or SFO, including codes
    # absent from the common-code list above.
    if any(len(token) <= 4 and token.isupper() for token in tokens):
        return None, "airport_code_or_acronym"
    return candidate, None


def _candidate_text(raw: str) -> str:
    """Return only the candidate portion, without leaking unrelated tail text."""
    candidate = re.split(r"[,;.!?]", raw, maxsplit=1)[0]
    candidate = normalize_name_text(candidate)
    candidate = _TAIL.sub("", candidate).strip(" \t.,;:!?\"“”()[]{}<>")
    return " ".join(candidate.split())


def parse_passenger_name(message: str) -> PassengerNameParse:
    """Extract a plausible passenger name from explicit or contextual phrasing.

    Supports ``my name is``, ``book for``, ``passenger NAME``, ``NAME is
    traveling``, and name phrases embedded in search or booking requests.
    No candidate is inferred from an unanchored arbitrary sentence.
    """
    if not isinstance(message, str) or not message.strip():
        return PassengerNameParse("not_found", reason="empty_message")

    normalized, spans = _normalized_with_spans(message)
    candidates: list[tuple[str, str]] = []
    for match in _INTRODUCER.finditer(normalized):
        candidates.append((_source_slice(message, spans, match.end(), len(normalized)), match.group(0).strip()))
    if re.search(r"\b(?:search|find|book|booking|reserve|reservation)\b", normalized, re.I):
        for match in _CONTEXT_FOR.finditer(normalized):
            candidates.append((_source_slice(message, spans, match.end(), len(normalized)), "for"))
    traveling = _TRAVELING.search(normalized)
    if traveling:
        names_start, names_end = traveling.span("names")
        pieces = _NAME_CONJUNCTION.split(traveling.group("names"))
        cursor = names_start
        for piece in pieces:
            piece_start = normalized.find(piece, cursor, names_end)
            piece_end = piece_start + len(piece)
            candidates.append((_source_slice(message, spans, piece_start, piece_end), traveling.group(0)))
            cursor = piece_end

    if not candidates:
        return PassengerNameParse("not_found", reason="no_name_introducer")

    # Prefer the first explicit occurrence; do not silently skip an unsafe
    # explicitly supplied value and accidentally bind a later unrelated name.
    raw, phrase = candidates[0]
    name, reason = _validate(raw)
    if name is None:
        return PassengerNameParse("rejected", candidate=_candidate_text(raw),
                                  matched_phrase=phrase, reason=reason)
    return PassengerNameParse("found", name=name, candidate=name,
                              matched_phrase=phrase)


def passenger_name_candidates(message: str) -> tuple[PassengerNameParse, ...]:
    """Return parse outcomes for every independently introduced passenger phrase.

    The conversational parser intentionally returns the first candidate; audit
    redaction uses this companion to cover every passenger mention in a string.
    """
    if not isinstance(message, str) or not message.strip():
        return ()
    normalized, spans = _normalized_with_spans(message)
    anchors: list[tuple[int, str, str | None]] = [
        (match.start(), _source_slice(message, spans, match.end(), len(normalized)), match.group(0).strip())
        for match in _INTRODUCER.finditer(normalized)
    ]
    if re.search(r"\b(?:search|find|book|booking|reserve|reservation)\b", normalized, re.I):
        anchors.extend((match.start(), _source_slice(message, spans, match.end(), len(normalized)), "for")
                       for match in _CONTEXT_FOR.finditer(normalized))
    for match in _TRAVELING.finditer(normalized):
        names_start, names_end = match.span("names")
        cursor = names_start
        for piece in _NAME_CONJUNCTION.split(match.group("names")):
            piece_start = normalized.find(piece, cursor, names_end)
            piece_end = piece_start + len(piece)
            anchors.append((piece_start, _source_slice(message, spans, piece_start, piece_end), match.group(0)))
            cursor = piece_end

    outcomes = []
    for _, raw, phrase in sorted(anchors, key=lambda anchor: anchor[0]):
        name, reason = _validate(raw)
        if name is None:
            outcomes.append(PassengerNameParse(
                "rejected", candidate=_candidate_text(raw),
                matched_phrase=phrase, reason=reason,
            ))
        else:
            outcomes.append(PassengerNameParse(
                "found", name=name, candidate=name, matched_phrase=phrase,
            ))
    return tuple(outcomes)


__all__ = ["MAX_NAME_LENGTH", "PassengerNameParse", "normalize_name_text", "parse_passenger_name", "passenger_name_candidates"]

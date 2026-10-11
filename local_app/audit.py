"""Redacted JSONL audit log for local sessions."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parsing import parse_passenger_name, passenger_name_candidates

CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)")
_WORD = r"[^\W\d_](?:[^\W\d_]|\u0300-\u036f)*(?:['’ʼ\-‐‑‒–—−][^\W\d_](?:[^\W\d_]|\u0300-\u036f)*)*"
_NON_NAME_WORDS = {
    "book", "booking", "cancel", "change", "confirm", "flight", "flights",
    "help", "hold", "lookup", "reserve", "reservation", "search", "ticket",
    "travel", "traveling", "please", "now", "next", "week", "month", "today",
    "tomorrow", "passenger", "passengers", "email", "phone", "card", "from", "to", "and",
    "is", "are", "will", "be", "after", "before", "through", "from", "with",
    "the", "my", "your", "our", "jfk", "lax", "iad", "las", "sfo", "ord",
    "lhr", "cdg", "dfw", "atl", "den", "sea", "mia", "bos", "ewr", "lga",
    "dca", "phx",
}


def passenger_name_phrases(text: str) -> tuple[str, ...]:
    """Find names in passenger/travel contexts without treating prose as names."""
    normalized, spans = _canonical_with_spans(text)
    contexts = list(re.finditer(
        r"\b(?:passengers?|travell?ing|travelled|traveller|passenger\s+name|"
        r"name\s+of\s+passenger|email|phone|card|pii)\b", normalized, re.I))
    phrases = []
    # Preserve the original conservative title-case detector for names embedded
    # beside PII labels (for example, "Amy Gonzalez, email ...").
    run = []
    for match in re.finditer(_WORD, text, re.UNICODE):
        word = match.group()
        components = re.split(r"['’ʼ\-‐‑‒–—−]", word)
        titlecase = (all(part and part[0].isupper() and
                         unicodedata.normalize("NFKC", part[1:]).lower() == part[1:]
                         for part in components) and word.casefold() not in _NON_NAME_WORDS)
        if titlecase:
            run.append(match)
        else:
            if len(run) >= 2:
                phrases.append(text[run[0].start():run[-1].end()])
            run = []
    if len(run) >= 2:
        phrases.append(text[run[0].start():run[-1].end()])
    for context in contexts:
        # Names follow a passenger label, or immediately precede a travel verb.
        context_word = context.group().casefold()
        if context_word in {"email", "phone", "card", "pii"}:
            region_start, region_end = max(0, context.start() - 120), min(len(normalized), context.end() + 120)
        elif context_word.startswith(("travel", "travell")):
            region_start, region_end = max(0, context.start() - 120), context.start()
        else:
            region_start, region_end = context.end(), min(len(normalized), context.end() + 120)
        region = normalized[region_start:region_end]
        words = list(re.finditer(_WORD, region, re.UNICODE))
        run = []
        for word_match in words:
            word = word_match.group()
            source_start = spans[region_start + word_match.start()][0]
            source_end = spans[region_start + word_match.end() - 1][1]
            source_word = text[source_start:source_end]
            folded = word.casefold()
            looks_name = (folded not in _NON_NAME_WORDS and
                          (source_word.isupper() or
                           all(ch.lower() == ch.upper() for ch in source_word) or
                           (source_word[0].isupper() and source_word[1:] == source_word[1:].lower()) or
                           any(unicodedata.category(ch).startswith("M") for ch in source_word)))
            if looks_name:
                run.append(word_match)
            else:
                if len(run) >= 2:
                    a, b = region_start + run[0].start(), region_start + run[-1].end()
                    phrases.append(text[spans[a][0]:spans[b - 1][1]])
                run = []
        if len(run) >= 2:
            a, b = region_start + run[0].start(), region_start + run[-1].end()
            phrases.append(text[spans[a][0]:spans[b - 1][1]])
    return tuple(dict.fromkeys(phrases))


class AuditLog:
    def __init__(self, path: str | Path | None = None):
        default = os.environ.get("RLAIF_AUDIT_LOG", ".local/audit.jsonl")
        self.path = Path(path or default)
        self._passengers: set[str] = set()

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._learn_structured(payload)
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
                  "payload": self._redact(payload)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def register_passenger(self, name: str | None) -> None:
        if isinstance(name, str) and name.strip():
            self._passengers.add(name.strip())

    def _learn_structured(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold() in {"passenger", "passenger_name", "passenger_candidate", "name"} and isinstance(item, str):
                    self.register_passenger(item)
                else:
                    self._learn_structured(item)
        elif isinstance(value, list):
            for item in value: self._learn_structured(item)

    def _redact(self, value):
        if isinstance(value, str):
            parsed = parse_passenger_name(value)
            self.register_passenger(parsed.candidate)
            for candidate in passenger_name_candidates(value):
                self.register_passenger(candidate.candidate)
            for candidate in passenger_name_phrases(value):
                self.register_passenger(candidate)
            for name in sorted(self._passengers, key=len, reverse=True):
                value = _redact_equivalent(value, name)
            value = CARD.sub("[REDACTED]", value)
            value = EMAIL.sub("[REDACTED]", value)
            value = PHONE.sub("[REDACTED]", value)
            return value
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in {"password", "payment"}:
                    continue
                if str(key).casefold() in {"passenger", "passenger_name", "passenger_candidate", "name"} and item:
                    if isinstance(item, str): self._passengers.add(item.strip())
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self._redact(item)
            return result
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        return value


def _canonical_with_spans(text: str):
    """NFKC/casefold text while retaining offsets into the original string."""
    chars, spans = [], []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        cluster = unicodedata.normalize("NFKC", text[index:end]).casefold()
        cluster = cluster.translate(str.maketrans({
            "’": "'", "‘": "'", "ʼ": "'", "‐": "-", "‑": "-",
            "‒": "-", "–": "-", "—": "-", "−": "-",
        }))
        for char in cluster:
            if char.isspace():
                if chars and chars[-1] == " ":
                    spans[-1] = (spans[-1][0], end)
                    continue
                char = " "
            chars.append(char)
            spans.append((index, end))
        index = end
    return "".join(chars), spans


def _redact_equivalent(text: str, name: str) -> str:
    needle, _ = _canonical_with_spans(name)
    haystack, spans = _canonical_with_spans(text)
    if not needle:
        return text
    matches = []
    cursor = 0
    while (found := haystack.find(needle, cursor)) >= 0:
        start, _ = spans[found]
        _, end = spans[found + len(needle) - 1]
        matches.append((start, end))
        cursor = found + len(needle)
    for start, end in reversed(matches):
        text = text[:start] + "[REDACTED]" + text[end:]
    return text

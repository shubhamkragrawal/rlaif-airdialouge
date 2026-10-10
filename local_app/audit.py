"""Redacted JSONL audit log for local sessions."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)")
NAME = re.compile(r"\b((?:my\s+)?name\s+is)\s+[^.,;!?]+", re.I)


class AuditLog:
    def __init__(self, path: str | Path | None = None):
        default = os.environ.get("RLAIF_AUDIT_LOG", ".local/audit.jsonl")
        self.path = Path(path or default)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
                  "payload": self._redact(payload)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _redact(self, value):
        if isinstance(value, str):
            value = CARD.sub("[REDACTED]", value)
            value = EMAIL.sub("[REDACTED]", value)
            value = PHONE.sub("[REDACTED]", value)
            return NAME.sub(r"\1 [REDACTED]", value)
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items() if k not in {"password", "payment"}}
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        return value

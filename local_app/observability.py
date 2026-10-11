"""Immutable, privacy-aware records for interactive runs and frozen evaluations.

These records are data foundations only: callers decide where and when to persist
them. Interactive telemetry and evaluation artifacts use distinct record types.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any

from .audit import AuditLog, passenger_name_phrases

RUN_SCHEMA = "rlaif.observability.run"
TURN_SCHEMA = "rlaif.observability.turn"
EVALUATION_SCHEMA = "rlaif.observability.evaluation_artifact"
SCHEMA_VERSION = 1
_SENSITIVE_KEYS = {"passenger", "passenger_name", "passenger_candidate", "name"}
_STANDALONE_NAME = re.compile(
    r"^\s*[\w]+(?:[’'\-‐‑‒–—−][\w]+)*(?:\s+[\w]+(?:[’'\-‐‑‒–—−][\w]+)*){1,3}\s*$",
    re.UNICODE,
)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("records cannot contain non-finite numbers")
        return value
    raise TypeError(f"unsupported record value: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


def _redact(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Apply the interactive audit redactor to a detached record snapshot."""
    logger = AuditLog(path="")
    payload = _plain(value)
    logger._learn_structured(payload)
    return _freeze(logger._redact(payload))


def canonical_json(value: Any) -> str:
    """Stable UTF-8-friendly JSON encoding used for manifests and hashes."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def record_json(record: Any) -> str:
    if not isinstance(record, (RunRecord, TurnRecord, EvaluationArtifact)):
        raise TypeError("record must be a supported observability record")
    payload = _plain(record)
    logger = AuditLog(path="")
    logger._learn_structured(payload)
    redacted = logger._redact(payload)
    validate_redacted(redacted)
    return canonical_json(redacted)


def record_sha256(record: Any) -> str:
    return hashlib.sha256(record_json(record).encode("utf-8")).hexdigest()


def validate_redacted(record: Any) -> None:
    """Raise ValueError if any key or free-text value may expose a passenger name."""
    payload = _plain(record)
    from .parsing import passenger_name_candidates

    def contains_name(text: str) -> bool:
        if "[REDACTED]" in text:
            # Keep a boundary so contextual parsing cannot join words on
            # either side of a legitimately redacted name.
            text = text.replace("[REDACTED]", ". ")
        if not text.strip():
            return False
        # The contextual extractor catches names in ordinary prose (including
        # rejected Unicode candidates); the anchored form covers bare names.
        candidates = passenger_name_candidates(text)
        if any(candidate.candidate for candidate in candidates):
            return True
        if passenger_name_phrases(text):
            return True
        normalized = unicodedata.normalize("NFKC", text).strip()
        if not _STANDALONE_NAME.fullmatch(normalized):
            return False
        words = re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)
        return len(words) >= 2 and all(word[0].isupper() for word in words)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold() in _SENSITIVE_KEYS and item != "[REDACTED]":
                    raise ValueError("passenger identity fields must be redacted")
                if contains_name(str(key)):
                    raise ValueError("record key appears to contain an unredacted passenger name")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and contains_name(value):
            raise ValueError("record value appears to contain an unredacted passenger name")

    visit(payload)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    controller_version: str
    schema_version: int = SCHEMA_VERSION
    adapter: Mapping[str, Any] = field(default_factory=dict)
    hardware: Mapping[str, Any] = field(default_factory=dict)
    generation_settings: Mapping[str, Any] = field(default_factory=dict)
    record_type: str = field(default=RUN_SCHEMA, init=False)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.controller_version, "controller_version")
        _require_version(self.schema_version)
        safe = _redact({name: getattr(self, name) for name in
                        ("adapter", "hardware", "generation_settings")})
        for name, value in safe.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TurnRecord:
    run_id: str
    scenario_id: str
    controller_version: str
    adapter: Mapping[str, Any] = field(default_factory=dict)
    tool_inputs: tuple[Mapping[str, Any], ...] = ()
    tool_outputs: tuple[Mapping[str, Any], ...] = ()
    transaction_evidence_ids: tuple[str, ...] = ()
    validation_result: str | None = None
    fallback_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    peak_memory_bytes: int | None = None
    final_outcome: str = "unknown"
    hardware: Mapping[str, Any] = field(default_factory=dict)
    generation_settings: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    record_type: str = field(default=TURN_SCHEMA, init=False)

    def __post_init__(self) -> None:
        for name in ("run_id", "scenario_id", "controller_version", "final_outcome"):
            _require_text(getattr(self, name), name)
        _require_version(self.schema_version)
        for name in ("input_tokens", "output_tokens", "peak_memory_bytes"):
            number = getattr(self, name)
            if number is not None and (isinstance(number, bool) or not isinstance(number, int) or number < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.latency_ms is not None and (not isinstance(self.latency_ms, (int, float))
                or isinstance(self.latency_ms, bool) or not math.isfinite(self.latency_ms)
                or self.latency_ms < 0):
            raise ValueError("latency_ms must be a finite non-negative number or None")
        if any(not isinstance(item, str) or not item for item in self.transaction_evidence_ids):
            raise ValueError("transaction_evidence_ids must contain non-empty strings")
        object.__setattr__(self, "transaction_evidence_ids", tuple(self.transaction_evidence_ids))
        protected = _redact({name: getattr(self, name) for name in
                             ("adapter", "hardware", "generation_settings",
                              "tool_inputs", "tool_outputs")})
        for name, value in protected.items():
            if name in ("tool_inputs", "tool_outputs"):
                value = tuple(value)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EvaluationArtifact:
    """Frozen evaluation metadata, kept distinct from interactive run telemetry."""
    run_id: str
    scenario_id: str
    controller_version: str
    adapter: Mapping[str, Any] = field(default_factory=dict)
    generation_settings: Mapping[str, Any] = field(default_factory=dict)
    final_outcome: str = "unknown"
    schema_version: int = SCHEMA_VERSION
    record_type: str = field(default=EVALUATION_SCHEMA, init=False)

    def __post_init__(self) -> None:
        for name in ("run_id", "scenario_id", "controller_version", "final_outcome"):
            _require_text(getattr(self, name), name)
        _require_version(self.schema_version)
        protected = _redact({"adapter": self.adapter,
                             "generation_settings": self.generation_settings})
        object.__setattr__(self, "adapter", protected["adapter"])
        object.__setattr__(self, "generation_settings", protected["generation_settings"])


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_version(value: int) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported observability schema version: {value}")


__all__ = ["EvaluationArtifact", "RunRecord", "TurnRecord", "canonical_json",
           "record_json", "record_sha256", "validate_redacted"]

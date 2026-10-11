"""Optional model runtime boundary.

The app remains usable without ML dependencies. On a workstation with the bundled
Qwen environment, ``TransformersAdapter`` can be connected without changing the
controller, tools, or guardrails.
"""

from __future__ import annotations

import hashlib
import gc
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .controller import ADAPTERS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_ROOT = PROJECT_ROOT / "local_models"
DEFAULT_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
os.environ.setdefault("HF_HOME", str(DEFAULT_ADAPTER_ROOT / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class RuntimeConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_paths: dict[str, str] | None = None
    revision: str | None = DEFAULT_MODEL_REVISION
    max_new_tokens: int = 160
    temperature: float = 0.2

    @classmethod
    def local_defaults(cls) -> "RuntimeConfig":
        paths = {
            name: str(DEFAULT_ADAPTER_ROOT / name)
            for name in ADAPTERS
            if (DEFAULT_ADAPTER_ROOT / name / "adapter_config.json").is_file()
        }
        return cls(adapter_paths=paths)


class RuntimeUnavailable(RuntimeError):
    pass


class ModelAdapter:
    name = "base"

    def generate(self, messages: list[dict[str, str]], retrieved_flights: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class TransformersAdapter(ModelAdapter):
    """Lazy PEFT/Transformers adapter; imports are delayed until selected."""

    def __init__(self, name: str, config: RuntimeConfig):
        if name not in ADAPTERS:
            raise ValueError(f"unknown adapter: {name}")
        paths = config.adapter_paths or {}
        adapter_path = paths.get(name)
        if not adapter_path:
            raise RuntimeUnavailable(f"no local path configured for adapter '{name}'")
        if not Path(adapter_path).exists():
            raise RuntimeUnavailable(f"adapter path does not exist: {adapter_path}")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeUnavailable(
                "model runtime needs torch, transformers, and peft; "
                "the default 16 GB Mac setup intentionally runs the deterministic baseline"
            ) from exc
        self.name = name
        self.config = config
        dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
        kwargs = {"revision": config.revision} if config.revision else {}
        # Starting model mode must never trigger an implicit model download.
        kwargs["local_files_only"] = True
        tokenizer_source = (
            adapter_path
            if (Path(adapter_path) / "tokenizer_config.json").is_file()
            else config.base_model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)
        base = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=dtype, **kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = str(dtype).replace("torch.", "")
        self.model.to(self.device).eval()
        self._peak_memory_bytes = 0

    def generate(self, messages, retrieved_flights, *, max_new_tokens: int | None = None,
                 deadline_seconds: float | None = None, cancel_event=None):
        import torch
        import time
        from transformers import StoppingCriteria, StoppingCriteriaList
        started = time.monotonic()
        if not isinstance(messages, list) or not all(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError("messages must be chat messages with role and text content")
        if not isinstance(retrieved_flights, list):
            raise ValueError("retrieved_flights must be a list")

        grounded_messages = [
            {
                "role": "system",
                "content": (
                    "Grounding context for this turn follows as JSON. Mention only flight numbers "
                    "and prices present in it. An empty list means no flight may be claimed: "
                    + json.dumps(retrieved_flights, sort_keys=True, separators=(",", ":"))
                ),
            },
            *messages,
        ]
        prompt = self.tokenizer.apply_chat_template(
            grounded_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_limit = self.config.max_new_tokens if max_new_tokens is None else min(
            self.config.max_new_tokens, max(0, int(max_new_tokens)))
        if token_limit <= 0 or deadline_seconds is not None and deadline_seconds <= 0:
            raise RuntimeUnavailable("generation budget exhausted")
        generation_kwargs = {
            "max_new_tokens": token_limit,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
        else:
            generation_kwargs.update({"temperature": None, "top_p": None, "top_k": None})
        if deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
            raise RuntimeUnavailable("session_time_exceeded")
        stop_at = None if deadline_seconds is None else started + deadline_seconds

        class _StopOnDeadlineOrCancellation(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                cancelled = cancel_event is not None and cancel_event.is_set()
                expired = stop_at is not None and time.monotonic() >= stop_at
                return torch.full((input_ids.shape[0],), cancelled or expired,
                                  dtype=torch.bool, device=input_ids.device)

        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnDeadlineOrCancellation()])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        self._record_peak_memory(torch)
        # MPS generation is synchronous and cannot be safely interrupted midway.
        # A late result is discarded by raising before it can reach the controller.
        if ((cancel_event is not None and cancel_event.is_set())
                or deadline_seconds is not None and time.monotonic() - started >= deadline_seconds):
            raise RuntimeUnavailable("session_time_exceeded")
        generated = output[0][inputs["input_ids"].shape[1]:]
        self.last_usage = {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(generated.shape[0]),
        }
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not response:
            raise RuntimeUnavailable("model generated an empty response")
        return response

    def metadata(self) -> dict[str, Any]:
        adapter_path = Path((self.config.adapter_paths or {})[self.name])
        raw = json.loads((adapter_path / "adapter_config.json").read_text())
        return {
            "adapter": self.name,
            "base_model": self.config.base_model,
            "base_model_revision": self.config.revision,
            "adapter_config": raw,
            "adapter_sha256": _artifact_hash(adapter_path),
            "device": self.device,
            "precision": self.dtype,
            "generation_settings": _generation_settings(self.config),
            "peak_memory_bytes": self._peak_memory_bytes,
            "peft_type": raw.get("peft_type"),
        }

    def _record_peak_memory(self, torch) -> None:
        try:
            if self.device == "mps":
                self._peak_memory_bytes = max(self._peak_memory_bytes, int(torch.mps.current_allocated_memory()))
            elif str(self.device).startswith("cuda"):
                self._peak_memory_bytes = max(self._peak_memory_bytes, int(torch.cuda.max_memory_allocated(self.device)))
        except (AttributeError, RuntimeError):
            pass


class RuntimeManager:
    """Lazily load one policy at a time, which is essential on a 16 GB Mac."""

    def __init__(self, config: RuntimeConfig | None = None, factory=TransformersAdapter):
        self.config = config or RuntimeConfig.local_defaults()
        self.factory = factory
        self._name: str | None = None
        self._runtime: ModelAdapter | None = None
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._request_id = 0
        self._loading_name: str | None = None
        self._errors: dict[str, str] = {}

    def configured(self, name: str) -> bool:
        path = (self.config.adapter_paths or {}).get(name)
        return bool(path and (Path(path) / "adapter_config.json").is_file())

    def adapter_status(self, name: str) -> dict[str, Any]:
        """Return a stable, secret-free snapshot of one adapter's lifecycle."""
        with self._lock:
            if self._runtime is not None and self._name == name:
                state = "ready"
            elif self._loading_name == name:
                state = "loading"
            else:
                path = (self.config.adapter_paths or {}).get(name)
                if not path:
                    state = "unavailable"
                elif not Path(path).exists():
                    state = "configured"
                else:
                    config_file = Path(path) / "adapter_config.json"
                    try:
                        parsed = json.loads(config_file.read_text())
                        if not isinstance(parsed, dict):
                            raise ValueError
                    except (OSError, ValueError, json.JSONDecodeError):
                        state = "incompatible"
                    else:
                        state = "available"
            error = self._errors.get(name)
        result = _status_envelope(name, state)
        if error:
            result["provenance_status"] = "error"
        return result

    def status(self, name: str) -> dict[str, Any]:
        """Alias for adapter_status, suitable for UI integrations."""
        return self.adapter_status(name)

    def provenance(self, name: str | None = None) -> dict[str, Any]:
        """Return runtime provenance without local paths or credential material."""
        with self._lock:
            selected = name if name is not None else self._name
            runtime = self._runtime if selected is not None and selected == self._name else None
        if selected is None:
            return _status_envelope(None, "unavailable")
        envelope = self.adapter_status(selected)
        result: dict[str, Any] = dict(envelope)
        with self._lock:
            last_error = self._errors.get(selected)
        if last_error:
            result["error"] = last_error
        if runtime is not None:
            metadata = getattr(runtime, "metadata", None)
            if callable(metadata):
                try:
                    metadata_result = metadata()
                    metadata_result = dict(metadata_result) if isinstance(metadata_result, dict) else {}
                    result.update(metadata_result)
                    result["adapter"] = selected
                    result.setdefault("base_model", self.config.base_model)
                    result.setdefault("base_model_revision", self.config.revision)
                    result.setdefault("generation_settings", _generation_settings(self.config))
                    path = (self.config.adapter_paths or {}).get(selected)
                    if path:
                        result.setdefault("adapter_sha256", _artifact_hash(Path(path)))
                        try:
                            result.setdefault("adapter_config", json.loads(
                                (Path(path) / "adapter_config.json").read_text()))
                        except (OSError, ValueError, json.JSONDecodeError):
                            pass
                    result.update(envelope)
                    result["adapter"] = selected
                    return _safe_provenance(result)
                except Exception as exc:
                    result["provenance_status"] = "error"
                    result["error"] = _safe_error(exc)
                    return result
        result.update({"base_model": self.config.base_model,
                       "base_model_revision": self.config.revision,
                       "generation_settings": _generation_settings(self.config)})
        path = (self.config.adapter_paths or {}).get(selected)
        if path and self.configured(selected):
            try:
                config = json.loads((Path(path) / "adapter_config.json").read_text())
                result["adapter_config"] = config
                result["adapter_sha256"] = _artifact_hash(Path(path))
                result["provenance_status"] = "partial"
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                result["provenance_status"] = "error"
                result["error"] = _safe_error(exc)
        result["adapter"] = selected
        return _safe_provenance(result)

    def get(self, name: str, *, cancel_event=None, deadline_seconds: float | None = None) -> ModelAdapter:
        if name not in ADAPTERS:
            raise RuntimeUnavailable("unknown adapter")
        path = (self.config.adapter_paths or {}).get(name)
        if not path:
            raise RuntimeUnavailable(f"adapter '{name}' is not configured")
        if not self.configured(name):
            raise RuntimeUnavailable(f"adapter '{name}' is not available locally")
        started = time.monotonic()
        with self._lock:
            if ((cancel_event is not None and cancel_event.is_set()) or
                    (deadline_seconds is not None and time.monotonic() - started >= deadline_seconds)):
                raise RuntimeUnavailable("session_time_exceeded")
            if self._runtime is not None and self._name == name:
                return self._runtime
            self._request_id += 1
            ticket = self._request_id
            old, self._runtime, self._name = self._runtime, None, None
            self._loading_name = name
            self._errors.pop(name, None)
        self._dispose(old)
        candidate = None
        try:
            with self._load_lock:
                with self._lock:
                    if ticket != self._request_id:
                        raise RuntimeUnavailable("runtime_load_superseded")
                self._check_budget(started, cancel_event, deadline_seconds)
                candidate = self.factory(name, self.config)
        except Exception as exc:
            with self._lock:
                if ticket == self._request_id:
                    self._loading_name = None
                    self._errors[name] = _safe_error(exc)
            if isinstance(exc, RuntimeUnavailable):
                raise
            raise RuntimeUnavailable(_safe_error(exc)) from exc
        rejected_reason = None
        with self._lock:
            expired = ((cancel_event is not None and cancel_event.is_set()) or
                       (deadline_seconds is not None and time.monotonic() - started >= deadline_seconds))
            if expired or ticket != self._request_id:
                rejected_reason = "session_time_exceeded" if expired else "runtime_load_superseded"
            else:
                self._runtime, self._name = candidate, name
                self._loading_name = None
        if rejected_reason is not None:
            # Cleanup can import torch / release MPS cache. Keep that work out
            # of the manager lock so other sessions can still inspect/load state.
            self._dispose(candidate)
            with self._lock:
                if ticket == self._request_id:
                    self._loading_name = None
            raise RuntimeUnavailable(rejected_reason)
        return candidate

    @staticmethod
    def _check_budget(started, cancel_event, deadline_seconds):
        if (cancel_event is not None and cancel_event.is_set()) or (
                deadline_seconds is not None and time.monotonic() - started >= deadline_seconds):
            raise RuntimeUnavailable("session_time_exceeded")

    def release(self) -> None:
        with self._lock:
            self._request_id += 1
            old, self._runtime = self._runtime, None
            self._name = None
            self._loading_name = None
        self._dispose(old)

    @staticmethod
    def _dispose(runtime) -> None:
        # Keep cleanup independent of the manager lock and leave custom adapter
        # objects alone; dropping the manager reference releases them normally.
        if runtime is None:
            return
        try:
            if isinstance(runtime, TransformersAdapter):
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()
        except (ImportError, RuntimeError):
            gc.collect()


def _generation_settings(config: RuntimeConfig) -> dict[str, Any]:
    return {"max_new_tokens": config.max_new_tokens, "temperature": config.temperature}


def _status_envelope(adapter: str | None, lifecycle_status: str) -> dict[str, Any]:
    available = lifecycle_status in {"available", "ready"}
    if lifecycle_status in {"available", "ready", "loading"}:
        mode = "local_model"
    else:
        mode = "deterministic_fallback"
    provenance_status = {
        "ready": "complete",
        "available": "partial",
        "loading": "pending",
        "incompatible": "error",
        "configured": "unavailable",
        "unavailable": "unavailable",
    }[lifecycle_status]
    return {
        "adapter": adapter,
        "lifecycle_status": lifecycle_status,
        "mode": mode,
        "available": available,
        "provenance_status": provenance_status,
    }


def _artifact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode() if path.is_dir() else item.name.encode())
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_error(exc: Exception) -> str:
    # Arbitrary backend exceptions can embed credentials or request payloads.
    # Preserve known runtime diagnostics; expose only the exception class otherwise.
    message = "adapter operation failed"
    if isinstance(exc, RuntimeUnavailable):
        message = " ".join(str(exc).split())
        message = re.sub(r"/(?:[^\s:'\"]+/)+[^\s:'\"]*", "[local path]", message)
    return f"{type(exc).__name__}: {message[:240]}"


def _safe_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    blocked = {"adapter_path", "path", "token", "secret", "password", "api_key", "apikey"}
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()
                    if key.casefold() not in blocked
                    and not any(part in key.casefold() for part in ("secret", "password", "api_key"))
                    and not key.casefold().endswith("_token")}
        if isinstance(item, list):
            return [clean(value) for value in item]
        if isinstance(item, str):
            item = re.sub(r"/(?:[^\s:'\"]+/)+[^\s:'\"]*", "[local path]", item)
            return re.sub(r"(?i)(?:token|secret|password|api[_-]?key)\s*[:=]\s*\S+",
                          "[redacted]", item)
        return item
    return clean(value)

"""Optional model runtime boundary.

The app remains usable without ML dependencies. On a workstation with the bundled
Qwen environment, ``TransformersAdapter`` can be connected without changing the
controller, tools, or guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
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
        tokenizer_source = (
            adapter_path
            if (Path(adapter_path) / "tokenizer_config.json").is_file()
            else config.base_model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)
        base = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=dtype, **kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device).eval()

    def generate(self, messages, retrieved_flights):
        import torch
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
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
        else:
            generation_kwargs.update({"temperature": None, "top_p": None, "top_k": None})
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
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
            "revision": self.config.revision,
            "adapter_path": str(adapter_path),
            "device": self.device,
            "peft_type": raw.get("peft_type"),
        }


class RuntimeManager:
    """Lazily load one policy at a time, which is essential on a 16 GB Mac."""

    def __init__(self, config: RuntimeConfig | None = None, factory=TransformersAdapter):
        self.config = config or RuntimeConfig.local_defaults()
        self.factory = factory
        self._name: str | None = None
        self._runtime: ModelAdapter | None = None

    def configured(self, name: str) -> bool:
        path = (self.config.adapter_paths or {}).get(name)
        return bool(path and (Path(path) / "adapter_config.json").is_file())

    def get(self, name: str) -> ModelAdapter:
        if not self.configured(name):
            raise RuntimeUnavailable(f"adapter '{name}' is not available locally")
        if self._runtime is None or self._name != name:
            self.release()
            self._runtime = self.factory(name, self.config)
            self._name = name
        return self._runtime

    def release(self) -> None:
        if self._runtime is None:
            return
        self._runtime = None
        self._name = None
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

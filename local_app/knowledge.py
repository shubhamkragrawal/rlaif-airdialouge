"""Structured local flight knowledge base.

The corpus is used as a deterministic scenario source; no network or model call is
made here. A scenario is selected by stable dialogue id so smoke tests are repeatable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "airdialogue_sampled_5k.jsonl"


class FlightKnowledgeBase:
    def __init__(self, corpus_path: str | Path = DEFAULT_CORPUS, scenario_id: int = 0):
        self.corpus_path = Path(corpus_path)
        self.scenario_id = scenario_id
        self.scenario = self._load_scenario(scenario_id)
        self.flights = tuple(self.scenario.get("kb", ()))
        self._by_id = {int(f["flight_number"]): f for f in self.flights}

    def _load_scenario(self, scenario_id: int) -> dict[str, Any]:
        if scenario_id < 0:
            raise ValueError("scenario_id must be non-negative")
        with self.corpus_path.open(encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                if index == scenario_id:
                    return json.loads(line)
        raise ValueError(f"scenario_id {scenario_id} is not present in {self.corpus_path}")

    def get(self, flight_id: int) -> dict[str, Any] | None:
        return self._by_id.get(int(flight_id))

    def all(self) -> tuple[dict[str, Any], ...]:
        return self.flights

"""Distribution invariants for dependency-light clean installs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib
import unittest

from local_app.evaluation import load_scenarios
from local_app.knowledge import DEFAULT_CORPUS, FlightKnowledgeBase


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SHA256 = "dda7898144f9a326f558088353d3fcc16da6f4865daec7e8c4324f17496959e4"


class PackagingTests(unittest.TestCase):
    def test_core_metadata_is_dependency_free_and_model_extra_is_pinned(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        project = metadata["project"]
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(
            project["optional-dependencies"]["model"],
            [
                "torch==2.5.1", "transformers==4.46.3", "peft==0.13.2",
                "accelerate==1.1.1", "numpy==1.26.4", "safetensors==0.4.5",
                "sentencepiece==0.2.0",
            ],
        )
        self.assertIn("data/*.jsonl", metadata["tool"]["setuptools"]["package-data"]["local_app"])

    def test_packaged_fixture_is_hash_pinned_exact_cohort_and_usable(self):
        payload = DEFAULT_CORPUS.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), FIXTURE_SHA256)
        rows = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual(len(rows), 75)
        self.assertTrue(all(set(row) == {"intent", "expected_action", "dialogue", "kb"} for row in rows))
        scenarios = load_scenarios()
        self.assertEqual(len(scenarios), 50)
        self.assertEqual(scenarios[0].scenario_id, 0)
        self.assertEqual(scenarios[-1].scenario_id, 74)
        self.assertIn(53, {scenario.scenario_id for scenario in scenarios})
        for scenario in scenarios:
            kb = FlightKnowledgeBase(scenario_id=scenario.scenario_id)
            self.assertTrue(kb.all())
        self.assertTrue((ROOT / "local_app/data/PROVENANCE.md").is_file())


if __name__ == "__main__":
    unittest.main()

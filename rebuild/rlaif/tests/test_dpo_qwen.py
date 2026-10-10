"""CPU checks for DPO data isolation and reference provenance, no model download."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import rlaif.training.dpo_qwen as dpo
import rlaif.shared.data as D


class DataIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = D.load_splits()
        cls.records = {r["dialogue_id"]: r for r in D.load_dialogues()}
        did = cls.splits["pair_gen"][0]
        turns = D.parse_turns(cls.records[did]["dialogue"])
        turn = D.agent_turn_indices(turns)[1]
        cls.row = {"scenario_id": did, "turn_index": turn, "prompt": "rendered chat",
                   "messages": D.build_messages(cls.records[did], turns, turn),
                   "chosen": "A factual booking reply", "rejected": "An invented flight",
                   "scores": {"0": 8, "1": 7, "2": 6, "3": 4},
                   "chosen_index": 0, "rejected_index": 3,
                   "gap": 4, "source_sha256": cls.splits["source_sha256"],
                   "split_seed": cls.splits["seed"]}

    def load(self, row):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(json.dumps(row) + "\n")
            return dpo.load_pairs(path, self.splits, self.records)

    def test_valid_pair(self):
        self.assertEqual(len(self.load(self.row)), 1)

    def test_heldout_cannot_enter_training(self):
        row = dict(self.row, scenario_id=self.splits["held_out"][0])
        with self.assertRaisesRegex(ValueError, "pair_gen"):
            self.load(row)

    def test_modified_source_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.load(dict(self.row, source_sha256="different"))

    def test_modified_kb_or_system_prompt_rejected(self):
        row = copy.deepcopy(self.row)
        row["messages"][0]["content"] += " altered KB"
        with self.assertRaisesRegex(ValueError, "messages disagree"):
            self.load(row)

    def test_greeting_prefix_rejected(self):
        turns = D.parse_turns(self.records[self.row["scenario_id"]]["dialogue"])
        with self.assertRaisesRegex(ValueError, "non-greeting"):
            self.load(dict(self.row, turn_index=D.agent_turn_indices(turns)[0]))

    def test_normalized_duplicate_responses_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            self.load(dict(self.row, rejected=" A FACTUAL   BOOKING REPLY "))

    def test_near_tie_rejected(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            self.load(dict(self.row, gap=2))

    def test_fabricated_gap_rejected(self):
        with self.assertRaisesRegex(ValueError, "judge scores"):
            self.load(dict(self.row, gap=5))

    def test_sft_cannot_come_from_another_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps(
                {"base_model_name_or_path": dpo.MODEL_ID}))
            (Path(tmp) / "run_meta.json").write_text(json.dumps(
                {"source_sha256": "different", "split_seed": self.splits["seed"],
                 "model": dpo.MODEL_ID, "finished_utc": "2026-10-08T00:00:00Z"}))
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                dpo.validate_sft(adapter, self.splits, dpo.MODEL_ID)


if __name__ == "__main__":
    unittest.main()

"""CPU tests for rejecting fabricated preferences and crash-safe resumption."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types

from rlaif.preferences.gen_pairs import (VertexJudge, candidate_order, decision_turn_reason, distinct_count, judge_protocol_fingerprint,
                       judge_with_correction, semantic_ranking_violations,
                       load_journal, load_replay_evidence, main as pairs_main, parse_checked_ranking,
                       ranking_response_schema, ReplaySampler, select_turn, statistics_for)
from rlaif.preferences.rank_rubric import build_ranking_prompt
from rlaif.preferences.parse_ranking import parse_ranking


GOOD = {"reasoning": "B is best", "scores": {"A": 3, "B": 9, "C": 3, "D": 1},
        "ranking": ["B", "A", "C", "D"]}


class PairTests(unittest.TestCase):
    def test_parser_layers(self):
        self.assertEqual(parse_ranking("```json\n" + json.dumps(GOOD) + "\n```"), GOOD)
        dirty = json.dumps(GOOD).replace("B is best", "B\x01is best")
        self.assertEqual(parse_ranking(dirty)["ranking"], GOOD["ranking"])
        broken = 'reasoning example "scores": {"A": 1}, "ranking": ["A"] END ' + json.dumps(GOOD)[:-1]
        self.assertEqual(parse_ranking(broken)["scores"], GOOD["scores"])

    def test_invalid_judge_never_produces_preference(self):
        for changes in ({"scores": {"A": 9}}, {"ranking": ["A", "A", "C", "D"]},
                        {"ranking": ["D", "A", "C", "B"]},
                        {"scores": {"A": True, "B": 9, "C": 3, "D": 1}},
                        {"scores": {"A": 3, "B": 11, "C": 3, "D": 1}}):
            self.assertIsNone(parse_ranking(json.dumps(dict(GOOD, **changes))))
        self.assertIsNone(parse_ranking('{"scores":{"A":2},"ranking":['))
        self.assertIsNone(parse_ranking(None))

    def test_degenerate_and_greeting_selection(self):
        self.assertEqual(distinct_count([" YES\nplease ", "yes please", "No"]), 2)
        rec = {"dialogue_id": 12, "dialogue": ["customer: Hi", "agent: Hello",
                "customer: Boston please", "agent: Which dates?", "customer: Today",
                "agent: No flights"]}
        for seed in range(20):
            _, turn = select_turn(rec, seed)
            self.assertIn(turn, (3, 5))

    def test_journal_repairs_only_crash_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal"
            path.write_bytes(b'{"scenario_id":1}\n{"scenario_id":')
            self.assertEqual(load_journal(path), [{"scenario_id": 1}])
            self.assertEqual(path.read_bytes(), b'{"scenario_id":1}\n')
            path.write_bytes(b'{"scenario_id":1}')
            load_journal(path)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            path.write_bytes(b'bad\n{"scenario_id":1}\n')
            with self.assertRaises(ValueError):
                load_journal(path)

    def test_content_ties_are_invariant_to_positions(self):
        candidates = ["Which dates?", "Dates please", "Which dates?", "Bad flight"]
        first = {"scores": {"A": 9, "B": 9, "C": 9, "D": 1}}
        reverse = {"scores": {"A": 1, "B": 9, "C": 9, "D": 9}}
        self.assertEqual(candidate_order(first, [0, 1, 2, 3], candidates),
                         candidate_order(reverse, [3, 2, 1, 0], candidates))
        # A factual score reversal must still change the winner; ties cannot hide it.
        changed = {"scores": {"A": 10, "B": 1, "C": 1, "D": 1}}
        self.assertNotEqual(candidate_order(first, [0, 1, 2, 3], candidates)[0],
                            candidate_order(changed, [3, 2, 1, 0], candidates)[0])

    def test_grounding_evidence_is_required_and_consistent(self):
        self.assertIsNone(parse_checked_ranking(json.dumps(GOOD)))
        checks = {label: {"named_flights": [], "factual_errors": [],
                         "constraint_errors": [], "progress_errors": [], "next_step": "Ask dates"}
                  for label in GOOD["scores"]}
        checked = dict(GOOD, checks=checks)
        self.assertIsNotNone(parse_checked_ranking(json.dumps(checked)))
        checks["B"]["factual_errors"] = ["Named flight is not in KB"]
        self.assertIsNone(parse_checked_ranking(json.dumps(checked)))
        checks["B"]["factual_errors"] = []
        checks["B"]["constraint_errors"] = ["Exceeds maximum price"]
        self.assertIsNone(parse_checked_ranking(json.dumps(checked)))
        checks["B"]["constraint_errors"] = []
        checks["B"]["progress_errors"] = ["Asks for already known airport"]
        self.assertIsNone(parse_checked_ranking(json.dumps(checked)))
        # Progress failure5 is not a suitability error, so it does not trigger cap4.
        progress_ok = dict(checked, scores={"A": 3, "B": 5, "C": 3, "D": 1})
        self.assertIsNotNone(parse_checked_ranking(json.dumps(progress_ok)))

    def test_crosscheck_uses_exact_kb_and_reverse_order(self):
        kb = "AVAILABLE FLIGHTS:\n1001 | JetBlue | $100\n1002 | AA | $200"
        candidates = ["Book1001", "Flight 1001", "Flight 9999", "Which dates?"]
        prompt, order = build_ranking_prompt(kb, [], candidates, permutation=[3, 2, 1, 0])
        self.assertEqual(order, [3, 2, 1, 0])
        self.assertIn("9999 | NOT PRESENT IN KB", prompt)
        self.assertIn("1001 | JetBlue | $100", prompt)
        self.assertEqual(prompt.count("content_id="), 4)
        with self.assertRaises(ValueError):
            build_ranking_prompt(kb, [], candidates, permutation=[0, 0, 2, 3])

    def test_gate_fingerprint_binds_settings_and_code(self):
        original = judge_protocol_fingerprint(1024, 2048)
        self.assertEqual(original, judge_protocol_fingerprint(1024, 2048))
        self.assertNotEqual(original, judge_protocol_fingerprint(0, 2048))
        self.assertNotEqual(original, judge_protocol_fingerprint(1024, 3072))
        with patch("rlaif.preferences.gen_pairs.file_sha256", return_value="changed-code"):
            self.assertNotEqual(original, judge_protocol_fingerprint(1024, 2048))

    def test_decision_prefix_excludes_easy_clarifications_and_closings(self):
        rec = {"dialogue_id": 12, "dialogue": ["customer: Hi", "agent: Hello",
                "customer: Boston please", "agent: Which dates?", "customer: Today",
                "agent: Flight 1001 costs $200. Confirm?", "customer: Yes please",
                "agent: Booked flight 1001", "customer: Thank you", "agent: Welcome"]}
        for seed in range(20):
            _, index = select_turn(rec, seed, "decision")
            self.assertEqual(index, 5)
        self.assertEqual(decision_turn_reason("No flights are available"), "availability_decision")
        self.assertIsNone(decision_turn_reason("Which dates?"))
        self.assertIsNone(decision_turn_reason("Thank you for choosing us"))
        no_decision = {"dialogue_id": 5, "dialogue": ["customer: Hi", "agent: Hello",
                                                     "customer: Boston", "agent: Which dates?"]}
        self.assertIsNone(select_turn(no_decision, 1, "decision")[1])
        self.assertEqual(select_turn(no_decision, 1, "random")[1], 3)

    def test_schema_requires_progress_separate_from_suitability(self):
        schema = ranking_response_schema()
        self.assertEqual(schema["required"], ["reasoning", "scores", "ranking", "checks"])
        for label in ("A", "B", "C", "D"):
            check = schema["properties"]["checks"]["properties"][label]
            self.assertIn("progress_errors", check["required"])
            self.assertIn("constraint_errors", check["required"])

    def replay_fixture(self, directory, count=1):
        records = {}
        source = Path(directory) / "source"
        source.mkdir()
        metadata = {"model": "Qwen/Qwen2.5-3B-Instruct", "adapter_fingerprint": "base",
                    "source_sha256": "source-hash", "split_seed": 20260922, "seed": 20260922,
                    "prefix_policy": "decision", "temperature": .9, "top_p": .95,
                    "max_new_tokens": 128, "max_prompt_tokens": 3072,
                    "judge_model": "gemini-2.5-flash", "judge_protocol_fingerprint": "old-judge"}
        (source / "run_meta.json").write_text(json.dumps(metadata))
        rows = []
        for scenario_id in range(count):
            records[scenario_id] = {"dialogue_id": scenario_id, "kb": [], "dialogue": [
                "customer: Hi", "agent: Hello", "customer: Boston today", "agent: Flight 1001 costs $100"]}
            rows.append({"scenario_id": scenario_id, "turn_index": 3, "candidates": [
                "Good", "Neutral", "Bad", "Worst"], "distinct": 4, "prompt_tokens": 100})
        journal = source / "scenarios.jsonl"
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return metadata, records, journal

    def test_replay_validates_generation_and_source_without_touching_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, records, journal = self.replay_fixture(directory)
            original = journal.read_bytes()
            current = dict(metadata, judge_model="gemini-2.5-pro", judge_protocol_fingerprint="new-judge")
            rows, provenance = load_replay_evidence(journal, current, [0], records, str(journal))
            self.assertEqual(rows[0]["candidates"], ["Good", "Neutral", "Bad", "Worst"])
            self.assertEqual(provenance["source_judge_model"], "gemini-2.5-flash")
            self.assertEqual(journal.read_bytes(), original)
            for field, value in (("adapter_fingerprint", "other-adapter"), ("temperature", 1.0),
                                 ("source_sha256", "other-source"), ("prefix_policy", "random")):
                with self.assertRaises(ValueError):
                    load_replay_evidence(journal, dict(current, **{field: value}), [0], records, str(journal))
            with self.assertRaises(ValueError):
                load_replay_evidence(journal, current, [0, 1], records, str(journal))
            row = json.loads(journal.read_text())
            row["turn_index"] = 1
            journal.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_replay_evidence(journal, current, [0], records, str(journal))

    def test_replay_checks_exact_prompt_and_token_count(self):
        sampler = ReplaySampler.__new__(ReplaySampler)
        sampler.rows = {0: {"prompt_tokens": 100, "candidates": ["A", "B", "C", "D"],
                            "prompt": "rendered"}}
        sampler.tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: "rendered",
            encode=lambda *args, **kwargs: list(range(100)))
        self.assertEqual(sampler.sample(0, [])[1], ["A", "B", "C", "D"])
        sampler.rows[0]["prompt_tokens"] = 101
        with self.assertRaises(ValueError):
            sampler.sample(0, [])
        sampler.rows[0]["prompt_tokens"] = 100
        sampler.rows[0]["prompt"] = "tampered"
        with self.assertRaises(ValueError):
            sampler.sample(0, [])

    def test_cpu_replay_runs_all_gates_without_policy_loader(self):
        class FakeJudge:
            def __init__(self, *args):
                pass
            def rank(self, prompt):
                mapping = {"Good": 9, "Neutral": 7, "Bad": 5, "Worst": 1}
                scores = {}
                for line in prompt.splitlines():
                    if " [content_id=" in line:
                        label = line[0]
                        scores[label] = mapping[line.rsplit(": ", 1)[1]]
                checks = {label: {"named_flights": [], "factual_errors": [],
                                  "constraint_errors": [], "progress_errors": [],
                                  "next_step": "Measured candidate quality"} for label in scores}
                return json.dumps({"reasoning": "Grounded comparison", "scores": scores,
                                   "ranking": sorted(scores, key=lambda label: -scores[label]),
                                   "checks": checks})
        def fake_replay_init(sampler, model, rows):
            sampler.rows = rows
            sampler.tokenizer = types.SimpleNamespace(
                apply_chat_template=lambda *args, **kwargs: "rendered",
                encode=lambda *args, **kwargs: list(range(100)))
        with tempfile.TemporaryDirectory() as directory:
            metadata, records, journal = self.replay_fixture(directory, 20)
            output = Path(directory) / "output"
            split = {"seed": metadata["split_seed"], "source_sha256": metadata["source_sha256"],
                     "pair_gen": list(records)}
            argv = ["gen_pairs.py", "--dry-run", "--replay-candidates", str(journal),
                    "--judge-model", "gemini-2.5-pro", "--out-dir", str(output)]
            with patch("sys.argv", argv), patch("rlaif.preferences.gen_pairs.load_splits", return_value=split), \
                    patch("rlaif.preferences.gen_pairs.load_dialogues", return_value=list(records.values())), \
                    patch("rlaif.preferences.gen_pairs.Sampler", side_effect=AssertionError("GPU loader invoked")), \
                    patch.object(ReplaySampler, "__init__", fake_replay_init), \
                    patch("rlaif.preferences.gen_pairs.VertexJudge", FakeJudge), contextlib.redirect_stdout(io.StringIO()):
                pairs_main()
            stats = json.loads((output / "dryrun_stats.json").read_text())
            self.assertTrue(stats["passed"])
            self.assertEqual(stats["attempted"], 20)
            self.assertEqual(stats["stability_tested"], 5)
            self.assertEqual(stats["kept"], 20)
            self.assertIn("source_journal_sha256", stats["replay_provenance"])
            self.assertNotIn("replay_provenance", stats["metadata"])
            self.assertEqual(journal.read_text().count("\n"), 20)

    def checked_response(self):
        checks = {label: {"named_flights": [], "factual_errors": [], "constraint_errors": [],
                          "progress_errors": [], "next_step": "Grounded next action"}
                  for label in GOOD["scores"]}
        return dict(GOOD, checks=checks)

    def test_semantic_correction_is_one_call_and_preserves_identity_and_logs(self):
        valid = self.checked_response()
        invalid = json.loads(json.dumps(valid))
        invalid["checks"]["B"]["factual_errors"] = ["Wrong price compared with exact KB row"]
        first, corrected = json.dumps(invalid), json.dumps(valid)
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            prompts = []
            class FakeJudge:
                calls = 0
                def rank(self, prompt):
                    self.calls += 1
                    prompts.append(prompt)
                    if self.calls == 2:
                        # First evidence was made durable BEFORE attempting paid correction.
                        prior = json.loads(trace_path.read_text())
                        self.assert_role = prior["response_role"]
                        assert prior["raw"] == first
                        return corrected
                    return first
            judge = FakeJudge()
            prompt = "IMMUTABLE KB\nA [content_id=one]: First\nB [content_id=two]: Second"
            raw, parsed, trace = judge_with_correction(judge, prompt, trace_path, 12, "primary")
            self.assertEqual(judge.calls, 2)
            self.assertTrue(prompts[1].startswith(prompt))
            self.assertIn(json.dumps(first), prompts[1])
            self.assertIn("score=9 exceeds factual_errors cap=2", prompts[1])
            self.assertEqual(parsed["scores"], GOOD["scores"])
            self.assertEqual(raw, corrected)
            self.assertEqual(trace["correction_recovered"], 1)
            self.assertEqual(trace["api_attempts"], 2)
            self.assertEqual(trace["first_pass_raw"], first)
            self.assertEqual(len(trace_path.read_text().splitlines()), 2)

    def test_failed_correction_drops_without_clamping_or_third_call(self):
        invalid = self.checked_response()
        invalid["ranking"] = ["D", "C", "A", "B"]
        raw = json.dumps(invalid)
        judge = types.SimpleNamespace(calls=0)
        def reply(prompt):
            judge.calls += 1
            return raw
        judge.rank = reply
        effective, parsed, trace = judge_with_correction(judge, "Immutable candidates")
        self.assertIsNone(parsed)
        self.assertEqual(effective, raw)
        self.assertEqual(judge.calls, 2)
        self.assertEqual(trace["correction_recovered"], 0)
        self.assertEqual(trace["terminal_parse_failed"], 1)
        self.assertIn("scores must descend", " ".join(trace["first_pass_violations"]))

    def test_low_gap_and_incomplete_json_are_not_rejudged(self):
        valid = self.checked_response()
        valid["scores"] = {"A": 5, "B": 5, "C": 5, "D": 5}
        for raw in (json.dumps(valid), '{"scores":'):
            calls = []
            judge = types.SimpleNamespace(rank=lambda prompt: calls.append(prompt) or raw)
            _, _, trace = judge_with_correction(judge, "Immutable candidates")
            self.assertEqual(len(calls), 1)
            self.assertEqual(trace["correction_calls"], 0)
        self.assertIsNone(semantic_ranking_violations('{"scores":'))

    def test_corrections_do_not_dilute_failure_gate_denominator(self):
        rows = [{"scenario_id": i, "status": "kept", "distinct": 4, "gap": 4,
                 "judge_calls": 2, "logical_judgements": 1, "parse_failures": 0,
                 "correction_calls": 1, "correction_recovered": 1,
                 "first_pass_semantic_failures": 1} for i in range(20)]
        for row in rows[:5]:
            row["stability"] = {"agrees": True}
        for row in rows[:2]:
            row.update(parse_failures=1, status="parse_failed", correction_recovered=0)
        stats = statistics_for(rows, {}, True)
        self.assertEqual(stats["parse_failure_rate"], .1)
        self.assertFalse(stats["gates"]["parse_failures"])
        self.assertEqual(stats["first_pass_semantic_failure_rate"], 1)
        self.assertEqual(stats["correction_failed"], 2)

    def test_paid_api_cap_is_enforced_before_call(self):
        judge = VertexJudge.__new__(VertexJudge)
        judge.calls = judge.max_calls = 1
        capped = []
        judge.on_cap = lambda: capped.append(True)
        fake = types.ModuleType("google.genai")
        fake.types = types.SimpleNamespace()
        with patch.dict("sys.modules", {"google.genai": fake}):
            with self.assertRaises(SystemExit):
                judge.rank("never sent")
        self.assertEqual(capped, [True])

    def test_gates_count_drops_in_yield(self):
        rows = [{"scenario_id": i, "status": "kept", "distinct": 4, "gap": 4,
                 "judge_calls": 1, "parse_failures": 0} for i in range(20)]
        for row in rows[:5]:
            row["stability"] = {"agrees": True}
            row["judge_calls"] = 2
        self.assertTrue(statistics_for(rows, {}, True)["passed"])
        for row in rows[:11]:
            row.update(status="parse_failed", parse_failures=1)
            row.pop("gap")
        stats = statistics_for(rows, {}, True)
        self.assertFalse(stats["passed"])
        self.assertEqual(stats["yield_rate"], .45)
        self.assertEqual(stats["parse_failed"], 11)


if __name__ == "__main__":
    unittest.main()

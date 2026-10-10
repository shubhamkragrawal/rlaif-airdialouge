import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys

from rlaif.rewards.ppo_reward import GroundedRewardAdapter, RewardBudgetExceeded, PPOVertexJudge, TransientJudgeUnavailable

RECORD = {"kb": [{"flight_number": 1001}]}
PREFIX = [{"role": "user", "content": "Can you help me book?"}]


def answer(score=8, factual=None):
    return json.dumps({"score": score, "reasoning": "Asks for missing travel details.",
        "checks": {"named_flights": [], "factual_errors": factual or [],
                   "constraint_errors": [], "progress_errors": [], "next_step": "Ask route"}})


class Judge:
    model = "fake"
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0
    def rank(self, prompt):
        self.calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class APIError(Exception):
    def __init__(self, code):
        super().__init__("API failed")
        self.code = code


class Rewards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # render_kb is irrelevant to protocol validation in these tests.
        self.record = {"kb": []}

    def adapter(self, responses):
        judge = Judge(responses)
        return GroundedRewardAdapter(judge, self.tmp.name), judge

    def score(self, adapter, response="What route do you need?", **kwargs):
        return adapter.score(self.record, PREFIX, response, kwargs.get("version", "snapshot0"),
                             kwargs.get("rollout", "r1"))

    def test_durable_cache_and_policy_binding(self):
        adapter, judge = self.adapter([answer()])
        result = self.score(adapter)
        self.assertAlmostEqual(result["reward"], 2.5 / 4.5)
        resumed = GroundedRewardAdapter(judge, self.tmp.name)
        self.assertEqual(result, self.score(resumed))
        self.assertEqual(judge.calls, 1)
        with self.assertRaises(ValueError):
            self.score(resumed, version="snapshot1")
        with self.assertRaises(ValueError):
            self.score(resumed, response="Different reply")

    def test_bounded_consistency_correction(self):
        adapter, judge = self.adapter([answer(9, ["Invented price"]), answer(2, ["Invented price"])])
        result = self.score(adapter)
        self.assertEqual(result["score"], 2)
        self.assertEqual(judge.calls, 2)
        self.assertTrue(result["calls"][0]["violations"])
        self.assertIn("Invented price", result["calls"][0]["raw"])

    def test_no_fabricated_reward_or_third_call(self):
        adapter, judge = self.adapter([answer(9, ["Invented price"])] * 3)
        result = self.score(adapter)
        self.assertEqual(result["status"], "dropped")
        self.assertIsNone(result["reward"])
        self.assertEqual(judge.calls, 2)

    def test_malformed_json_dropped_without_paid_repair(self):
        adapter, judge = self.adapter(["invalid"])
        self.assertIsNone(self.score(adapter)["reward"])
        self.assertEqual(judge.calls, 1)

    def test_unknown_flight_penalized_without_api_call(self):
        adapter, judge = self.adapter([])
        self.assertEqual(self.score(adapter, "Flight 9999 is available.")["reward"], -1)
        self.assertEqual(judge.calls, 0)

    def test_cap_leaves_resumable_sample_and_first_response(self):
        adapter, judge = self.adapter([answer(9, ["Invented"]), RewardBudgetExceeded("cap")])
        with self.assertRaises(RewardBudgetExceeded):
            self.score(adapter)
        saved = json.loads(next(Path(self.tmp.name).glob("*.json")).read_text())
        self.assertEqual(saved["status"], "pending")
        self.assertEqual(len(saved["calls"]), 1)
        judge.responses = [answer(2, ["Invented"])]
        self.assertEqual(self.score(adapter)["score"], 2)
        self.assertEqual(judge.calls, 3)  # first pass not repeated; cap + correction

    def test_api_request_follows_durable_reservation_sync(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import sys
        judge = PPOVertexJudge.__new__(PPOVertexJudge)
        judge.path = Path(self.tmp.name) / "usage.json"
        judge.model = "fake"
        judge.config = {"max_calls": 2, "max_tokens": 100, "max_output_tokens": 10,
                        "thinking_budget": 0}
        judge.usage = {"calls": 0, "reported_total_tokens": 0,
                       "reserved_unreported_tokens": 0, "errors": 0}
        synced = []
        judge.sync = lambda: synced.append(json.loads(judge.path.read_text()))
        def request(**kwargs):
            self.assertEqual(len(synced), 1)
            self.assertEqual(synced[0]["calls"], 1)
            self.assertEqual(synced[0]["reserved_unreported_tokens"], 11)
            return SimpleNamespace(text="{}", usage_metadata=SimpleNamespace(total_token_count=4))
        judge.client = SimpleNamespace(models=SimpleNamespace(generate_content=request))
        types = SimpleNamespace(GenerateContentConfig=lambda **kw: kw,
                                ThinkingConfig=lambda **kw: kw)
        modules = {"google": SimpleNamespace(), "google.genai": SimpleNamespace(types=types)}
        with patch.dict(sys.modules, modules):
            self.assertEqual(judge.rank("x"), "{}")
        self.assertEqual(len(synced), 2)
        self.assertEqual(synced[-1]["reserved_unreported_tokens"], 0)
        self.assertEqual(synced[-1]["reported_total_tokens"], 4)

    def test_usage_cap_checked_before_network_and_retained_after_resume(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import sys
        judge = PPOVertexJudge.__new__(PPOVertexJudge)
        judge.path = Path(self.tmp.name) / "usage.json"
        synced = []
        judge.sync = lambda: synced.append(json.loads(judge.path.read_text()))
        judge.config = {"max_calls": 1, "max_tokens": 100, "max_output_tokens": 10,
                        "thinking_budget": 0}
        judge.usage = {"calls": 1, "reported_total_tokens": 4,
                       "reserved_unreported_tokens": 7, "errors": 0}
        modules = {"google": SimpleNamespace(), "google.genai": SimpleNamespace(types=object())}
        with patch.dict(sys.modules, modules):
            with self.assertRaises(RewardBudgetExceeded):
                judge.rank("x")
        saved = json.loads(judge.path.read_text())
        self.assertEqual(saved["calls"], 1)
        self.assertEqual(saved["reserved_unreported_tokens"], 7)
        self.assertEqual(synced, [saved])

    def test_evidence_cannot_omit_explicit_known_flight(self):
        adapter, judge = self.adapter([answer(), answer()])
        # KB row fields used by renderer, so get a real record locally.
        from rlaif.shared.data import load_dialogues
        record = next(r for r in load_dialogues() if r["kb"])
        number = record["kb"][0]["flight_number"]
        result = adapter.score(record, PREFIX, "Flight %s is available." % number, "s0", "r2")
        self.assertIsNone(result["reward"])
        self.assertEqual(judge.calls, 2)

    def retry_judge(self, results, **limits):
        def start_patch(context):
            value = context.start()
            self.addCleanup(context.stop)
            return value
        judge = PPOVertexJudge.__new__(PPOVertexJudge)
        judge.path = Path(self.tmp.name) / "retry_usage.json"
        judge.model = "fake"
        judge.config = dict(max_calls=10, max_tokens=1000, max_output_tokens=10,
                            thinking_budget=0, max_attempts=4,
                            retry_initial_delay=5., retry_max_delay=30.)
        judge.config.update(limits)
        judge.usage = dict(calls=0, reported_total_tokens=0,
                           reserved_unreported_tokens=0, errors=0)
        synced, requests = [], []
        judge.sync = lambda: synced.append(json.loads(judge.path.read_text()))
        results = iter(results)
        def request(**kwargs):
            # Every request has a distinct durable reservation before transmission.
            latest = synced[-1]
            self.assertEqual(latest["calls"], len(requests) + 1)
            self.assertEqual(latest["attempt_log"][-1]["status"], "reserved")
            self.assertEqual(latest["reserved_unreported_tokens"], 11 * latest["calls"])
            requests.append(kwargs)
            value = next(results)
            if isinstance(value, Exception):
                raise value
            return SimpleNamespace(text=value, usage_metadata=SimpleNamespace(total_token_count=4))
        judge.client = SimpleNamespace(models=SimpleNamespace(generate_content=request))
        types = SimpleNamespace(GenerateContentConfig=lambda **kw: kw, ThinkingConfig=lambda **kw: kw)
        start_patch(patch.dict(sys.modules, {"google": SimpleNamespace(),
            "google.genai": SimpleNamespace(types=types)}))
        self.sleep = start_patch(patch("rlaif.rewards.ppo_reward.time.sleep"))
        start_patch(patch("rlaif.rewards.ppo_reward.random.SystemRandom.uniform", return_value=0))
        return judge, synced, requests

    def test_transient_retries_have_durable_attempts_and_conservative_usage(self):
        judge, synced, requests = self.retry_judge([APIError(429), APIError(503), "{}"])
        self.assertEqual(judge.rank("x"), "{}")
        self.assertEqual(len(requests), 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [5., 10.])
        self.assertEqual(judge.usage["errors"], 2)
        self.assertEqual(judge.usage["reserved_unreported_tokens"], 22)
        self.assertEqual(judge.usage["reported_total_tokens"], 4)
        self.assertEqual([a["status"] for a in synced[-1]["attempt_log"]], ["error", "error", "complete"])

    def test_permanent_and_unstructured_errors_are_not_retried(self):
        for error in (APIError(403), RuntimeError("429 in prose")):
            with self.subTest(error=error):
                judge, _, requests = self.retry_judge([error])
                self.assertIsNone(judge.rank("x"))
                self.assertEqual(len(requests), 1)
                self.sleep.assert_not_called()

    def test_retry_limit_is_finite(self):
        judge, _, requests = self.retry_judge([APIError(500)] * 3, max_attempts=2)
        with self.assertRaises(TransientJudgeUnavailable):
            judge.rank("x")
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.sleep.call_count, 1)
        self.assertEqual(judge.usage["reserved_unreported_tokens"], 22)

    def test_transient_exhaustion_preserves_pending_journal_and_correction(self):
        adapter, judge = self.adapter([answer(9, ["Invented"]), TransientJudgeUnavailable("HTTP429")])
        with self.assertRaises(TransientJudgeUnavailable):
            self.score(adapter)
        saved = json.loads(next(Path(self.tmp.name).glob("*.json")).read_text())
        self.assertEqual(saved["status"], "pending")
        self.assertEqual(len(saved["calls"]), 1)
        self.assertNotIn("reward", saved)
        judge.responses = [answer(2, ["Invented"])]
        self.assertEqual(self.score(adapter)["score"], 2)
        self.assertEqual(judge.calls, 3)

    def test_retry_cumulative_caps_stop_before_sleep_or_request(self):
        for caps in ({"max_calls": 1}, {"max_tokens": 21}):
            with self.subTest(caps=caps):
                judge, _, requests = self.retry_judge([APIError(429), "{}"], **caps)
                with self.assertRaises(RewardBudgetExceeded):
                    judge.rank("x")
                self.assertEqual(len(requests), 1)
                self.sleep.assert_not_called()
                self.assertEqual(json.loads(judge.path.read_text())["reserved_unreported_tokens"], 11)

    def test_retry_sync_failure_prevents_second_request(self):
        judge, _, requests = self.retry_judge([APIError(429), "{}"])
        original_sync = judge.sync
        def sync():
            original_sync()
            if judge.calls == 2:
                raise OSError("GCS upload failed")
        judge.sync = sync
        with self.assertRaises(OSError):
            judge.rank("x")
        self.assertEqual(len(requests), 1)
        self.assertEqual(judge.usage["errors"], 1)

    def test_empty_response_is_not_retried(self):
        judge, _, requests = self.retry_judge([""])
        self.assertIsNone(judge.rank("x"))
        self.assertEqual(len(requests), 1)
        self.sleep.assert_not_called()

    def test_sdk_retries_disabled_and_configuration_bound_on_resume(self):
        captured = []
        types = SimpleNamespace(HttpOptions=lambda **kw: kw, HttpRetryOptions=lambda **kw: kw)
        genai = SimpleNamespace(types=types, Client=lambda **kw: captured.append(kw))
        with patch.dict(sys.modules, {"google": SimpleNamespace(genai=genai), "google.genai": genai}):
            kwargs = dict(project="p", location="global", model="m",
                          usage_path=Path(self.tmp.name) / "constructor.json", max_calls=10, max_tokens=10000)
            judge = PPOVertexJudge(**kwargs)
            self.assertEqual(captured[0]["http_options"]["retry_options"], {"attempts": 1})
            judge._persist_usage()
            with self.assertRaises(ValueError):
                PPOVertexJudge(**kwargs, max_attempts=2)


if __name__ == "__main__":
    unittest.main()

"""PPO evaluation inherits the completed v3 benchmark without changing its metrics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rlaif.evaluation.eval_qwen import SIMULATOR_VERSION

ROOT = Path(__file__).resolve().parents[2]


def configuration(arm):
    return json.loads((ROOT / "cloud" / ("eval_%s_full_a100.json" % arm)).read_text())


def arguments(config):
    args = config["workerPoolSpecs"][0]["containerSpec"]["args"]
    out, index = {}, 0
    while index < len(args):
        flag = args[index]
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            out[flag] = args[index + 1]
            index += 2
        else:
            out[flag] = True
            index += 1
    return out


class PPOEvaluationReadiness(unittest.TestCase):
    def test_same_v3_protocol_and_bounded_single_gpu(self):
        self.assertEqual(SIMULATOR_VERSION, "customer-v3-known-flight-offers")
        ppo = configuration("ppo")
        pool = ppo["workerPoolSpecs"][0]
        self.assertEqual(len(ppo["workerPoolSpecs"]), 1)
        self.assertEqual(pool["replicaCount"], "1")
        self.assertEqual(pool["machineSpec"]["acceleratorCount"], 1)
        self.assertEqual(ppo["scheduling"]["timeout"], "14400s")
        ppo_args = arguments(ppo)
        exceptions = {"--arm", "--adapter", "--out", "--gcs-output"}
        common = {k: v for k, v in ppo_args.items() if k not in exceptions}
        for arm in ("base", "sft", "dpo"):
            other = configuration(arm)
            self.assertEqual(common, {k: v for k, v in arguments(other).items() if k not in exceptions})
            self.assertEqual(pool["machineSpec"], other["workerPoolSpecs"][0]["machineSpec"])
        self.assertTrue(pool["containerSpec"]["imageUri"].endswith(":20261009-v12"))
        self.assertEqual(ppo_args["--adapter"], "gs://portfolioproject-505823-rlaif/runs/20261009-ppo-v3/checkpoint-0019/adapter")
        self.assertEqual(ppo_args["--gcs-output"], "gs://portfolioproject-505823-rlaif/runs/20261009-eval-ppo-v3-r3")

    def test_ppo_cli_manifest_matches_existing_50(self):
        args = configuration("ppo")["workerPoolSpecs"][0]["containerSpec"]["args"].copy()
        index = args.index("--gcs-output")
        del args[index:index + 2]  # The test cannot upload or invoke cloud services.
        with tempfile.TemporaryDirectory() as directory:
            args[args.index("--out") + 1] = directory
            result = subprocess.run([sys.executable, "-m", "rlaif.evaluation.eval_qwen", *args, "--dry-run"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("no model loaded or Gemini calls made", result.stdout)
            manifest = json.loads((Path(directory) / "eval_manifest.json").read_text())
            self.assertEqual(manifest["manifest_sha256"],
                             "eae0dda2106cab5b293cc9c2f5096bef796507052c763c1b80bf106e7883089a")
            self.assertEqual(len(manifest["scenarios"]), 50)
            self.assertEqual(sum(bool(s["matching_flights"]) for s in manifest["scenarios"]), 25)

    def test_adaptive_ppo_eval_keeps_protocol_with_fresh_paths(self):
        baseline = arguments(configuration("ppo"))
        adaptive_config = configuration("ppo_adaptive")
        adaptive = arguments(adaptive_config)
        adaptive_args = adaptive_config["workerPoolSpecs"][0]["containerSpec"]["args"]
        for flag in ("--adapter", "--out", "--gcs-output"):
            self.assertEqual(adaptive_args.count(flag), 1,
                             "adaptive config must pass %s exactly once" % flag)
        exceptions = {"--adapter", "--out", "--gcs-output"}
        self.assertEqual({k: v for k, v in baseline.items() if k not in exceptions},
                         {k: v for k, v in adaptive.items() if k not in exceptions})
        self.assertEqual(adaptive_config["scheduling"]["timeout"], "5400s")
        self.assertTrue(adaptive_config["workerPoolSpecs"][0]["containerSpec"]["imageUri"].endswith(":20261009-v13"))
        self.assertEqual(adaptive["--adapter"],
            "gs://portfolioproject-505823-rlaif/runs/20261009-ppo-adaptive-v4/adapter")
        self.assertEqual(adaptive["--gcs-output"],
            "gs://portfolioproject-505823-rlaif/runs/20261009-eval-ppo-adaptive-v3")

    def test_ppo_requires_adapter_before_model_loading(self):
        result = subprocess.run([sys.executable, "-m", "rlaif.evaluation.eval_qwen", "--arm", "ppo", "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("trained arms require --adapter", result.stderr)


if __name__ == "__main__":
    unittest.main()

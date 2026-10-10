"""Generate bounded PPO configs; this never builds an image or launches a job."""
import argparse
import json
from pathlib import Path

try:
    from .identity import require_fresh_output, validate_identity
except ImportError:  # Direct script invocation.
    from identity import require_fresh_output, validate_identity


def make_config(project, region, tag, date, pilot, full_run_suffix="", model_revision=None):
    name = "ppo-pilot" if pilot else "ppo"
    run_name = name + ("-" + full_run_suffix if full_run_suffix else "")
    args = ["--adapter", f"gs://{project}-rlaif/runs/20261008-sft/sft/adapter",
        "--out-dir", f"/tmp/rlaif-{run_name}", "--project", project, "--location", region,
        "--gcs-output", f"gs://{project}-rlaif/runs/{date}-{run_name}",
        "--judge-model", "gemini-2.5-pro", "--max-updates", "2" if pilot else "50",
        "--max-scenarios", "2" if pilot else "50", "--max-rollouts", "8" if pilot else "200",
        "--max-wall-seconds", "2700" if pilot else "9000", "--prefixes-per-update", "1",
        "--candidates-per-prefix", "4", "--ppo-epochs", "2", "--max-new-tokens", "128",
        "--max-prompt-tokens", "3072", "--learning-rate", "5e-6",
        "--value-learning-rate", "1e-4", "--kl-coef", "0.1",
        "--max-judge-calls", "24" if pilot else "400",
        "--max-judge-tokens", "300000" if pilot else "4000000"]
    if model_revision:
        args[2:2] = ["--revision", model_revision]
    return {"workerPoolSpecs": [{"machineSpec": {
        "machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100",
        "acceleratorCount": 1}, "replicaCount": "1",
        "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100},
        "containerSpec": {"imageUri": f"{region}-docker.pkg.dev/{project}/rlaif/qwen:{tag}",
            "command": ["python", "-m", "rlaif.training.ppo_qwen"], "args": args}}],
        "scheduling": {"timeout": "3600s" if pilot else "9600s"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--date", required=True,
                        help="Fresh run date matching the YYYYMMDD image tag")
    parser.add_argument("--project", default="portfolioproject-505823")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--full-run-suffix", required=True,
        help="Fresh full-run suffix, for example closeout-v1")
    parser.add_argument("--allow-historical", action="store_true",
                        help="Forensic regeneration only; never use for a new cloud run")
    parser.add_argument("--model-revision",
                        help="Optional pinned Hugging Face commit for future exact reproduction")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New empty directory for the generated configs")
    args = parser.parse_args()
    validate_identity(args.tag, args.date, args.full_run_suffix,
                      allow_historical=args.allow_historical, require_suffix=True,
                      minimum_version=14)
    require_fresh_output(args.output_dir)
    args.output_dir.mkdir(parents=True)
    for pilot in (True, False):
        path = args.output_dir / ("ppo_pilot.json" if pilot else "ppo_full.json")
        path.write_text(json.dumps(make_config(args.project, args.region, args.tag, args.date,
            pilot, args.full_run_suffix, args.model_revision), indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()

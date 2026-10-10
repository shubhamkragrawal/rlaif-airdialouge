"""Generate a separate bounded continuation config; does not launch a cloud job."""
import argparse
import json
from pathlib import Path
try:
    from .prepare_ppo import make_config
except ImportError:  # Direct script invocation.
    from prepare_ppo import make_config


def config(project, region, tag, date, suffix, model_revision=None):
    result = make_config(project, region, tag, date, False, suffix, model_revision)
    spec = result["workerPoolSpecs"][0]["containerSpec"]
    spec["command"] = ["python", "-m", "rlaif.training.ppo_qwen_adaptive"]
    args = spec["args"]
    replacements = {"--ppo-epochs": "1", "--learning-rate": "1.5e-6",
        "--value-learning-rate": "2e-5", "--max-rollouts": "124",
        "--max-wall-seconds": "6600", "--max-judge-calls": "300",
        "--max-judge-tokens": "800000"}
    for key, value in replacements.items():
        args[args.index(key) + 1] = value
    args += ["--initial-checkpoint", f"gs://{project}-rlaif/runs/20261009-ppo-v3/checkpoint-0019",
             "--starting-update", "19", "--max-update-attempts", "4",
             "--retry-lr-decay", "0.5", "--min-learning-rate", "1e-7",
             "--gate-fraction", "0.8", "--advantage-clip", "5"]
    result["scheduling"]["timeout"] = "7200s"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--date", required=True,
                        help="Fresh run date matching the YYYYMMDD image tag")
    parser.add_argument("--project", default="portfolioproject-505823")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--suffix", required=True,
                        help="Fresh continuation suffix, for example closeout-v1")
    parser.add_argument("--allow-historical", action="store_true",
                        help="Forensic regeneration only; never use for a new cloud run")
    parser.add_argument("--model-revision",
                        help="Optional pinned Hugging Face commit for future exact reproduction")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New empty directory for the generated config")
    args = parser.parse_args()
    try:
        from .identity import require_fresh_output, validate_identity
    except ImportError:  # Direct script invocation.
        from identity import require_fresh_output, validate_identity
    validate_identity(args.tag, args.date, args.suffix,
                      allow_historical=args.allow_historical, require_suffix=True,
                      minimum_version=14)
    require_fresh_output(args.output_dir)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "ppo_adaptive.json"
    output.write_text(json.dumps(config(args.project, args.region, args.tag, args.date,
                                        args.suffix, args.model_revision), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

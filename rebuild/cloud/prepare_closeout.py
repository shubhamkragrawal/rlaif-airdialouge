"""Prepare a fresh immutable post-restructure closeout build and job configs.

This is metadata/config generation only. It refuses historical identities, requires a
pinned model revision, and refuses to overwrite an existing output directory. A later
operator can submit the recorded build context after reviewing its source manifest.
"""
import argparse
import json
from pathlib import Path

try:
    from .identity import require_fresh_output, validate_identity
    from .prepare import prepare_context
    from .prepare_ppo_adaptive import config as adaptive_config
except ImportError:  # Direct script invocation.
    from identity import require_fresh_output, validate_identity
    from prepare import prepare_context
    from prepare_ppo_adaptive import config as adaptive_config


def evaluation_config(project, region, tag, date, suffix, model_revision):
    image = f"{region}-docker.pkg.dev/{project}/rlaif/qwen:{tag}"
    output = f"gs://{project}-rlaif/runs/{date}-eval-ppo-{suffix}"
    return {"workerPoolSpecs": [{
        "machineSpec": {"machineType": "a2-highgpu-1g",
                         "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1},
        "replicaCount": "1",
        "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100},
        "containerSpec": {"imageUri": image,
                           "command": ["python", "-m", "rlaif.evaluation.eval_qwen"],
                           "args": ["--arm", "ppo", "--model", "Qwen/Qwen2.5-3B-Instruct",
                                    "--revision", model_revision, "--scenario-count", "50",
                                    "--seed", "20260922", "--max-agent-turns", "16",
                                    "--max-new-tokens", "160", "--simulator-model",
                                    "gemini-2.5-flash", "--secondary-judge", "--out",
                                    f"/tmp/eval-ppo-{suffix}", "--gcs-output", output,
                                    "--adapter", f"gs://{project}-rlaif/runs/{date}-ppo-{suffix}/adapter"]}
    }], "scheduling": {"timeout": "5400s"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--suffix", required=True,
                        help="Fresh suffix such as closeout-v1; it becomes part of every URI")
    parser.add_argument("--model-revision", required=True,
                        help="Pinned Hugging Face commit hash or immutable revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", default="portfolioproject-505823")
    parser.add_argument("--region", default="us-central1")
    args = parser.parse_args()
    validate_identity(args.tag, args.date, args.suffix, require_suffix=True, minimum_version=14)
    if not args.model_revision.strip() or args.model_revision in {"main", "latest"}:
        parser.error("--model-revision must be a pinned immutable revision, not a moving tag")
    require_fresh_output(args.output_dir)
    args.output_dir.mkdir(parents=True)
    metadata = prepare_context(args.project, args.region, args.tag, args.date, args.output_dir)
    metadata["model_revision"] = args.model_revision
    metadata["adaptive_config"] = adaptive_config(args.project, args.region, args.tag,
                                                    args.date, args.suffix, args.model_revision)
    metadata["evaluation_config"] = evaluation_config(args.project, args.region, args.tag,
                                                       args.date, args.suffix,
                                                       args.model_revision)
    (args.output_dir / "ppo_adaptive_closeout.json").write_text(
        json.dumps(metadata["adaptive_config"], indent=2) + "\n")
    (args.output_dir / "eval_ppo_closeout.json").write_text(
        json.dumps(metadata["evaluation_config"], indent=2) + "\n")
    (args.output_dir / "closeout_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()

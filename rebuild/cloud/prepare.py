"""Prepare an isolated post-restructure build context and bounded SFT configs.

This command only copies source files and writes local metadata. It never builds an
image or launches a Vertex job. New preparations require an immutable date/version
identity; historical contexts can be regenerated only with ``--allow-historical``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

try:
    from .identity import require_fresh_output, validate_identity
except ImportError:  # Direct script invocation.
    from identity import require_fresh_output, validate_identity


PROJECT = "portfolioproject-505823"
REGION = "us-central1"


def prepare_context(project, region, tag, date, output_dir=None):
    """Copy only supported package/data/build files and return context metadata."""
    image = f"{region}-docker.pkg.dev/{project}/rlaif/qwen:{tag}"
    rebuild = Path(__file__).resolve().parents[1]
    context = Path(tempfile.mkdtemp(prefix="rlaif-build-"))
    (context / "rebuild" / "cloud").mkdir(parents=True)
    (context / "rebuild" / "artifacts").mkdir()
    files = [rebuild.parent / "airdialogue_sampled_5k.jsonl"]
    files += sorted((rebuild / "rlaif").rglob("*.py"))
    files += sorted((rebuild / "cloud").glob("*.py"))
    # Keep checked-in protocol/config fixtures available to the in-image tests.
    files += sorted((rebuild / "cloud").glob("*.json"))
    files += [rebuild / "cloud" / "Dockerfile", rebuild / "cloud" / "requirements.txt"]
    files += [rebuild / "artifacts" / name
              for name in ["splits.json", "sft_examples.jsonl", "dataset_manifest.json"]]
    manifest = []
    for source in files:
        if not source.is_file():
            raise FileNotFoundError("required build input is missing: %s" % source)
        relative = source.relative_to(rebuild.parent)
        destination = context / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append({"path": str(relative),
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    (context / "rebuild" / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    metadata = {"schema_version": 2, "path": str(context), "image": image,
                "date": date, "source_layout": "rebuild/rlaif package",
                "files": manifest}
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "build_context.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def make_configs(project, region, tag, date, output_dir, model_revision=None):
    image = f"{region}-docker.pkg.dev/{project}/rlaif/qwen:{tag}"
    bucket = f"gs://{project}-rlaif"
    output = Path(output_dir)
    configs = {}
    for mode, steps, timeout in [("pilot", 20, "3600s"), ("sft", -1, "14400s")]:
        run_args = ["--run-name", mode, "--max-steps", str(steps),
                    "--gcs-output", f"{bucket}/runs/{date}-{mode}"]
        if model_revision:
            run_args[0:0] = ["--revision", model_revision]
        config = {"workerPoolSpecs": [{
            "machineSpec": {"machineType": "a2-highgpu-1g",
                             "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1},
            "replicaCount": "1",
            "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100},
            "containerSpec": {"imageUri": image,
                               "command": ["python", "rebuild/cloud/run.py"],
                               "args": run_args}
        }], "scheduling": {"timeout": timeout}}
        path = output / f"{mode}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        configs[mode] = str(path)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--date", required=True,
                        help="Fresh run date matching the YYYYMMDD image tag")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--output-dir", type=Path,
                        help="Durable metadata/config directory; defaults to a fresh temp directory")
    parser.add_argument("--allow-historical", action="store_true",
                        help="Forensic regeneration only; never use for a new cloud run")
    parser.add_argument("--model-revision",
                        help="Optional pinned Hugging Face commit for future exact reproduction")
    args = parser.parse_args()
    validate_identity(args.tag, args.date, allow_historical=args.allow_historical,
                      minimum_version=14)
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="rlaif-closeout-"))
    if args.output_dir:
        require_fresh_output(output)
    metadata = prepare_context(args.project, args.region, args.tag, args.date, output)
    configs = make_configs(args.project, args.region, args.tag, args.date, output,
                           args.model_revision)
    metadata["configs"] = configs
    (Path(output) / "build_context.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)
    print(metadata["path"])


if __name__ == "__main__":
    main()

"""Download the four final PEFT adapters from the private project bucket."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .runtime import DEFAULT_ADAPTER_ROOT


ADAPTER_SOURCES = {
    "sft": "gs://portfolioproject-505823-rlaif/runs/20261008-sft/sft/adapter",
    "dpo": "gs://portfolioproject-505823-rlaif/runs/20261008-dpo/adapter",
    "ppo_checkpoint_19": "gs://portfolioproject-505823-rlaif/runs/20261009-ppo-v3/checkpoint-0019/adapter",
    "ppo_adaptive_50": "gs://portfolioproject-505823-rlaif/runs/20261009-ppo-adaptive-v4/adapter",
}


def download(root: Path = DEFAULT_ADAPTER_ROOT, *, dry_run: bool = False) -> None:
    if shutil.which("gsutil") is None:
        raise RuntimeError("gsutil is required; install and authenticate the Google Cloud CLI")
    root.mkdir(parents=True, exist_ok=True)
    for name, source in ADAPTER_SOURCES.items():
        destination = root / name
        if (destination / "adapter_config.json").is_file():
            print(f"{name}: already present")
            continue
        print(f"{name}: {source} -> {destination}")
        if dry_run:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{source}/*", str(destination)],
            check=True,
        )
        if not (destination / "adapter_config.json").is_file():
            raise RuntimeError(f"{name}: download completed without adapter_config.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    download(args.root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

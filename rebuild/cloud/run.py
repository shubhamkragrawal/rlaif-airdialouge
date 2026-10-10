"""Managed job entry point. No credentials are baked into the container."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud.artifacts import sync_tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-output", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--revision", default=None,
                        help="Pinned Hugging Face commit or tag for exact reproduction")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found")
    print("[gpu]", torch.cuda.get_device_name(0), flush=True)
    root = Path("/tmp/rlaif-outputs")
    root.mkdir(exist_ok=True)
    log = root / "console.log"
    command = [sys.executable, "-m", "rlaif.training.sft_qwen", "--out-root", str(root),
               "--run-name", args.run_name, "--max-steps", str(args.max_steps),
               "--gcs-output", args.gcs_output.rstrip("/") + "/" + args.run_name]
    if args.revision:
        command += ["--revision", args.revision]
    started = time.time()
    code = 1
    try:
        with log.open("w") as handle:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
    finally:
        (root / "job_status.json").write_text(json.dumps({
            "exit_code": code, "elapsed_seconds": time.time() - started,
            "run_name": args.run_name, "max_steps": args.max_steps,
            "gpu": torch.cuda.get_device_name(0)}, indent=2))
        sync_tree(root, args.gcs_output)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

"""TRL 0.12.2 DPO from an SFT adapter and validated, disjoint preference pairs.

The frozen reference is a second copy of the SAME SFT adapter, never the base
Qwen model. Only the trained adapter is exported for evaluation. --dry-run uses
only the standard library to validate provenance; the GPU run also checks exact
chat rendering and refuses any prompt/completion truncation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import rlaif.shared.data as D

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def adapter_fingerprint(path):
    """Hash adapter configuration and weights, independent of its local path."""
    path = Path(path)
    files = [path / "adapter_config.json"]
    weights = sorted(path.glob("adapter_model*.safetensors"))
    if not weights:
        weights = sorted(path.glob("adapter_model*.bin"))
    if not files[0].is_file() or not weights:
        raise ValueError("SFT adapter must contain adapter_config.json and adapter weights")
    files = sorted(file for file in path.iterdir() if file.is_file() and
                   (file.name == "adapter_config.json" or file.name.startswith("adapter_model")))
    manifest = "\n".join("%s:%s" % (file.name, D.file_sha256(file)) for file in files)
    return hashlib.sha256(manifest.encode()).hexdigest()


def validate_sft(adapter, splits, model):
    meta_path = Path(adapter).parent / "run_meta.json"
    if not meta_path.is_file():
        raise ValueError("SFT run_meta.json is required beside the adapter directory")
    meta = json.loads(meta_path.read_text())
    for key, expected in [("source_sha256", splits["source_sha256"]),
                          ("split_seed", splits["seed"]), ("model", model)]:
        if meta.get(key) != expected:
            raise ValueError("SFT provenance mismatch: %s" % key)
    if not meta.get("finished_utc"):
        raise ValueError("SFT run has no finished_utc; use a completed SFT adapter")
    config = json.loads((Path(adapter) / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != model:
        raise ValueError("SFT adapter base model disagrees with --model")
    return meta


def load_pairs(path, splits=None, records=None, min_gap=3):
    """Reject contamination or fabricated prefixes before any GPU/model load."""
    splits = splits or D.load_splits()
    records = records or {r["dialogue_id"]: r for r in D.load_dialogues()}
    allowed = set(splits["pair_gen"])
    seen, pairs = set(), []
    with open(path, encoding="utf-8") as fh:
        for line, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            did, turn = row.get("scenario_id"), row.get("turn_index")
            if type(did) is not int or did not in allowed:
                raise ValueError("line %d: scenario is not in the pair_gen split" % line)
            if type(turn) is not int:
                raise ValueError("line %d: turn_index must be an integer" % line)
            if (did, turn) in seen:
                raise ValueError("line %d: duplicate scenario/turn" % line)
            seen.add((did, turn))
            for key, expected in [("source_sha256", splits["source_sha256"]),
                                  ("split_seed", splits["seed"])]:
                if row.get(key) != expected:
                    raise ValueError("line %d: pair provenance mismatch: %s" % (line, key))
            turns = D.parse_turns(records[did]["dialogue"])
            eligible = D.agent_turn_indices(turns)[1:]
            if turn not in eligible:
                raise ValueError("line %d: prefix is not an eligible non-greeting agent turn" % line)
            expected_messages = D.build_messages(records[did], turns, turn)
            if row.get("messages") != expected_messages:
                raise ValueError("line %d: messages disagree with committed source/prompt" % line)
            for key in ("prompt", "chosen", "rejected"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError("line %d: %s must be nonempty text" % (line, key))
            norm = lambda text: " ".join(text.lower().split())
            if norm(row["chosen"]) == norm(row["rejected"]):
                raise ValueError("line %d: chosen and rejected are identical" % line)
            gap = row.get("gap")
            if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not min_gap <= gap <= 9:
                raise ValueError("line %d: gap must be between %s and 9" % (line, min_gap))
            scores = row.get("scores")
            if not isinstance(scores, dict) or set(scores) != {"0", "1", "2", "3"}:
                raise ValueError("line %d: scores require four original candidate indices" % line)
            if any(type(score) is not int or not 1 <= score <= 10 for score in scores.values()):
                raise ValueError("line %d: scores must be integer ratings 1..10" % line)
            chosen_index, rejected_index = row.get("chosen_index"), row.get("rejected_index")
            if type(chosen_index) is not int or type(rejected_index) is not int:
                raise ValueError("line %d: chosen/rejected candidate indices are required" % line)
            if str(chosen_index) not in scores or str(rejected_index) not in scores:
                raise ValueError("line %d: chosen/rejected indices are outside scores" % line)
            if (scores[str(chosen_index)] != max(scores.values()) or
                    scores[str(rejected_index)] != min(scores.values()) or
                    gap != scores[str(chosen_index)] - scores[str(rejected_index)]):
                raise ValueError("line %d: claimed gap/preference disagrees with judge scores" % line)
            pairs.append(row)
    if not pairs:
        raise ValueError("No usable preference pairs; training would do nothing")
    return pairs


def check_rendering_and_lengths(pairs, tokenizer, max_length):
    lengths = []
    for row in pairs:
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                                add_generation_prompt=True)
        if row["prompt"] != rendered:
            raise ValueError("Stored prompt disagrees with the current Qwen chat template")
        prompt_len = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        for key in ("chosen", "rejected"):
            # TRL appends EOS to each completion. Reserve one token even if already present.
            completion_len = len(tokenizer(row[key], add_special_tokens=False)["input_ids"]) + 1
            if completion_len < 2:
                raise ValueError("Completion has no trainable text tokens")
            lengths.append(prompt_len + completion_len)
    if max(lengths) > max_length:
        raise ValueError("Pair token length %d exceeds --max-seq-len %d; raise the limit, "
                         "do not truncate the KB" % (max(lengths), max_length))
    return {"max": max(lengths), "p50": sorted(lengths)[len(lengths)//2],
            "truncated": 0, "max_seq_len": max_length}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=D.REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-a-git-repository"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=None,
                    help="Pinned Hugging Face commit or tag; historical runs may be unpinned")
    ap.add_argument("--sft-adapter", "--adapter", dest="sft_adapter", required=True)
    ap.add_argument("--pairs", default=os.path.join(D.ARTIFACTS, "pairs.jsonl"))
    ap.add_argument("--out-root", default=str(Path(__file__).parent / "outputs"))
    ap.add_argument("--run-name")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--resume-from-checkpoint")
    ap.add_argument("--gcs-output")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for key in ("epochs", "lr", "beta", "batch_size", "grad_accum", "max_seq_len", "save_steps"):
        if getattr(args, key) <= 0:
            ap.error("--%s must be positive" % key.replace("_", "-"))
    if args.max_steps == 0 or args.max_steps < -1:
        ap.error("--max-steps must be -1 or positive")
    # Download the whole SFT run so its provenance metadata travels with the adapter.
    # The generated pairs folder includes the judge gate and statistics as well.
    input_cache = None
    if args.sft_adapter.startswith("gs://") or args.pairs.startswith("gs://"):
        from cloud.artifacts import download_tree
        input_cache = tempfile.TemporaryDirectory(prefix="rlaif-dpo-inputs-")
        if args.sft_adapter.startswith("gs://"):
            uri = args.sft_adapter.rstrip("/")
            if not uri.endswith("/adapter"):
                ap.error("GCS --adapter must end in /adapter beneath an SFT run")
            local = Path(input_cache.name) / "sft"
            download_tree(uri.rsplit("/", 1)[0], str(local))
            args.sft_adapter = str(local / "adapter")
        if args.pairs.startswith("gs://"):
            folder, filename = args.pairs.rsplit("/", 1)
            local = Path(input_cache.name) / "pairs"
            download_tree(folder, str(local))
            args.pairs = str(local / filename)
    splits = D.load_splits()
    pairs = load_pairs(args.pairs, splits=splits)
    sft_meta = validate_sft(args.sft_adapter, splits, args.model)
    identity = {"model": args.model, "model_revision": args.revision,
                "source_sha256": splits["source_sha256"],
                "split_seed": splits["seed"], "pairs_sha256": D.file_sha256(args.pairs),
                "sft_adapter_sha256": adapter_fingerprint(args.sft_adapter),
                "epochs": args.epochs, "learning_rate": args.lr, "beta": args.beta,
                "batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accum,
                "max_seq_len": args.max_seq_len, "max_steps": args.max_steps}
    for row in pairs:
        if row.get("model") != args.model or row.get("adapter_fingerprint") != identity["sft_adapter_sha256"]:
            raise ValueError("Pair sampling policy differs from the supplied SFT checkpoint")
    print("[validated] %d preference pairs; SFT and pair provenance match" % len(pairs), flush=True)
    if args.dry_run:
        print(json.dumps(identity, indent=2))
        print("[dry-run] no model loaded; exact tokenizer/length validation runs on GPU")
        return

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(splits["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This unquantized bf16 training run requires a compatible CUDA GPU")
    run_dir = Path(args.out_root) / (args.run_name or "dpo-qwen2.5-3b-%s" %
                                     time.strftime("%Y%m%d-%H%M%S"))
    meta_path = run_dir / "run_meta.json"
    if args.resume_from_checkpoint:
        if not meta_path.is_file():
            raise ValueError("Resume requires --run-name of the original run with run_meta.json")
        old_meta = json.loads(meta_path.read_text())
        if any(old_meta.get(key) != val for key, val in identity.items()):
            raise ValueError("Resume configuration/provenance changed; use the original arguments")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_stats = check_rendering_and_lengths(pairs, tokenizer, args.max_seq_len)
    print("[tokens]", json.dumps(token_stats), flush=True)
    metadata = dict(identity, n_pairs=len(pairs), git_commit=git_commit(),
                    reference="frozen copy of the same SFT adapter", loss_type="sigmoid",
                    sft_run=sft_meta, token_lengths=token_stats,
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    package_versions={name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "trl", "peft", "datasets", "accelerate")})
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    reference_base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16)
    reference_base.config.use_cache = False
    reference = PeftModel.from_pretrained(
        reference_base, args.sft_adapter, is_trainable=False)
    reference.requires_grad_(False)
    reference.eval()
    if any(p.requires_grad for p in reference.parameters()):
        raise RuntimeError("Reference policy unexpectedly has trainable weights")
    metadata["resolved_model_revision"] = getattr(base.config, "_commit_hash", None)
    meta_path.write_text(json.dumps(metadata, indent=2))
    model.print_trainable_parameters()
    config = DPOConfig(
        output_dir=str(run_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, beta=args.beta, loss_type="sigmoid", lr_scheduler_type="cosine",
        warmup_ratio=0.05, max_grad_norm=0.3, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2, report_to=[],
        seed=splits["seed"], data_seed=splits["seed"], max_length=args.max_seq_len,
        max_prompt_length=None, max_completion_length=None,
        disable_dropout=True, reference_free=False,
    )
    dataset = Dataset.from_list([{k: row[k] for k in ("prompt", "chosen", "rejected")}
                                for row in pairs])
    trainer = DPOTrainer(model=model, ref_model=reference, args=config, processing_class=tokenizer,
                         train_dataset=dataset)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(len(pairs) / args.batch_size) / args.grad_accum * args.epochs)

    class Progress(TrainerCallback):
        def on_train_begin(self, args_, state, control, **kwargs):
            self.t0, self.step0 = time.time(), state.global_step

        def on_step_end(self, args_, state, control, **kwargs):
            elapsed_steps = state.global_step - self.step0
            if elapsed_steps == 20:
                seconds = (time.time() - self.t0) / elapsed_steps
                print("[eta] %.2f seconds/step; %.1f minutes remaining" %
                      (seconds, seconds * max(0, total_steps - state.global_step) / 60), flush=True)
            return control

        def on_save(self, args_, state, control, **kwargs):
            if args.gcs_output:
                from cloud.artifacts import sync_tree
                sync_tree(str(run_dir), args.gcs_output)
            return control

    trainer.add_callback(Progress())
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metadata.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    adapter_dir=str(adapter_dir), metrics=result.metrics)
    meta_path.write_text(json.dumps(metadata, indent=2))
    (run_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    if args.gcs_output:
        from cloud.artifacts import sync_tree
        sync_tree(str(run_dir), args.gcs_output)
    print("[done] DPO adapter -> %s" % adapter_dir, flush=True)


if __name__ == "__main__":
    main()

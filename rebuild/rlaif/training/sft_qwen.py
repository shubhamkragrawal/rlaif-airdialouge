"""LoRA SFT of Qwen2.5-3B-Instruct on the turn-level examples. Runs on one 24GB L4.

    python3 sft_qwen.py                          # defaults from the Day 1 brief
    python3 sft_qwen.py --batch-size 2 --grad-accum 8   # on OOM: halve, double

bf16, no quantization (~6.2GB of the 24GB). The adapter, the tokenizer and a run_meta.json
recording the split seed, the example count and the corpus hash are written to a versioned
directory under outputs/.

Two defences are deliberate and must not be removed:
  * the response template is passed to the collator as the STRING
    "<|im_start|>assistant\\n" so the tokenizer tokenizes it. Hardcoded token ids are how
    the original repo ended up carrying Mistral's [28792, 16289, 28793] into a run;
  * after the first batch is built we assert that label tokens survived masking, and abort
    if not. A template mismatch then fails loudly instead of training on nothing.
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rlaif.shared.data as D

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
SEED = 20260922
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=D.REPO_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "not-a-git-repository"


def load_examples(path, tokenizer):
    """Render each stored example into a single chat-templated training string."""
    from datasets import Dataset

    texts = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            e = json.loads(raw)
            msgs = e["messages"] + [{"role": "assistant", "content": e["completion"]}]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    return Dataset.from_dict({"text": texts})


def assert_labels_survive(collator, dataset, tokenizer, max_len):
    """Build one real batch and check the collator left something to learn from."""
    batch = collator([
        tokenizer(dataset[i]["text"], truncation=True, max_length=max_len)
        for i in range(min(4, len(dataset)))
    ])
    labels = batch["labels"]
    kept = int((labels != -100).sum())
    if kept == 0:
        raise SystemExit(
            "ABORT: the completion-only collator masked every label token. The response "
            "template %r does not occur in the rendered examples - the chat template and "
            "the collator disagree. Training now would fit nothing." % RESPONSE_TEMPLATE)
    total = int(labels.numel())
    print("[check] label tokens surviving masking: %d of %d (%.1f%%)"
          % (kept, total, 100.0 * kept / total))


def make_eta_callback(total_steps, after=20):
    from transformers import TrainerCallback

    class ETA(TrainerCallback):
        def __init__(self):
            self.t0 = None

        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            if state.global_step == after and self.t0:
                per_step = (time.time() - self.t0) / after
                remaining = (total_steps - after) * per_step
                print("[eta] %.2f s/step over %d steps -> ~%.0f min remaining "
                      "(%d steps total)" % (per_step, after, remaining / 60.0, total_steps))
            return control

    return ETA()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=None,
                    help="Pinned Hugging Face commit or tag; required for exact reproduction")
    ap.add_argument("--data", default=os.path.join(D.ARTIFACTS, "sft_examples.jsonl"))
    ap.add_argument("--out-root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"))
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--gcs-output", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and run the collator check, then stop")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig
    from trl import SFTTrainer
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:  # moved in newer trl
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM

    set_seed(SEED)
    run_dir = os.path.join(args.out_root, args.run_name or
                          "sft-qwen2.5-3b-%s" % time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=bool(args.resume_from_checkpoint))

    D.verify_dataset_manifest(require_sft_examples=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_examples(args.data, tokenizer)
    print("[data] %d examples from %s" % (len(dataset), args.data))

    lengths = sorted(len(tokenizer(t, add_special_tokens=False)["input_ids"])
                     for t in dataset["text"])
    initial_over = sum(n > args.max_seq_len for n in lengths)
    if initial_over / len(lengths) > 0.02:
        args.max_seq_len = max(args.max_seq_len, 3072)
    over = sum(n > args.max_seq_len for n in lengths)
    length_stats = {"p50": lengths[len(lengths)//2],
                    "p95": lengths[int(len(lengths)*.95)], "max": lengths[-1],
                    "max_seq_len": args.max_seq_len, "truncated": over,
                    "truncated_fraction": over / len(lengths)}
    print("[tokens]", json.dumps(length_stats), flush=True)
    for lo, hi in [(0,1024),(1024,1536),(1536,2048),(2048,2560),
                   (2560,3072),(3072,4096),(4096,1000000)]:
        print("[histogram] %d-%d: %d" % (lo, hi, sum(lo <= n < hi for n in lengths)))
    if over / len(lengths) > 0.02:
        raise SystemExit("ABORT: more than 2% truncated even at 3072. Review token lengths.")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer)
    assert_labels_survive(collator, dataset, tokenizer, args.max_seq_len)

    effective_batch = args.batch_size * args.grad_accum
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(len(dataset) / args.batch_size) / args.grad_accum * args.epochs)
    print("[plan] effective batch %d -> ~%d optimizer steps" % (effective_batch, total_steps))

    meta = {
        "model": args.model,
        "requested_model_revision": args.revision,
        "git_commit": git_commit(),
        "split_seed": SEED,
        "source_sha256": D.load_splits()["source_sha256"],
        "n_examples": len(dataset),
        "max_seq_len": args.max_seq_len,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "token_lengths": length_stats,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": LORA_TARGETS},
        "response_template": RESPONSE_TEMPLATE,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if args.dry_run:
        print("[dry-run] stopping before model load. meta written to %s" % run_dir)
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS)

    train_kwargs = dict(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        seed=SEED,
    )

    # trl renamed TrainingArguments -> SFTConfig and tokenizer -> processing_class across
    # versions; probe rather than pin, so the script runs on whatever the box has.
    try:
        from trl import SFTConfig
        train_kwargs["max_seq_length"] = args.max_seq_len
        train_kwargs["dataset_text_field"] = "text"
        cfg_fields = set(inspect.signature(SFTConfig.__init__).parameters)
        targs = SFTConfig(**dict((k, v) for k, v in train_kwargs.items() if k in cfg_fields))
        trainer_kwargs = {}
    except ImportError:
        from transformers import TrainingArguments
        targs = TrainingArguments(**train_kwargs)
        trainer_kwargs = {"max_seq_length": args.max_seq_len, "dataset_text_field": "text"}

    sig = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tokenizer
    trainer_kwargs = dict((k, v) for k, v in trainer_kwargs.items() if k in sig)

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=collator,
        **trainer_kwargs
    )
    trainer.add_callback(make_eta_callback(total_steps))
    if args.gcs_output:
        from cloud.artifacts import sync_tree
        from transformers import TrainerCallback

        class UploadCheckpoint(TrainerCallback):
            def on_save(self, args_, state, control, **kwargs):
                sync_tree(run_dir, args.gcs_output)
                return control

        trainer.add_callback(UploadCheckpoint())
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)

    adapter_dir = os.path.join(run_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["resolved_model_revision"] = getattr(trainer.model.config, "_commit_hash", None)
    meta["adapter_dir"] = adapter_dir
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("[done] adapter + tokenizer -> %s" % adapter_dir)
    if args.gcs_output:
        sync_tree(run_dir, args.gcs_output)


if __name__ == "__main__":
    main()

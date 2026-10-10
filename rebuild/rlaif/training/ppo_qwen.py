"""Bounded on-policy PPO of SFT Qwen with external grounded rewards.

One response is a finite episode; prompts come only from the committed pair_gen
split. Sampling uses the raw policy distribution (temperature=1, no top-p/top-k
warpers). Detached old-policy log probabilities/value predictions are collected
before any update and reused through all PPO epochs. The frozen reference is
the original SFT checkpoint. Neither prompt nor padding tokens enter losses.

Checkpoints include adapter, value head, optimizer, RNG and committed cursor.
Resume restarts from the last complete update; uncommitted tails may replay, but
the durable judge ledger never resets. No cloud job is launched by this script.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import rlaif.shared.data as D
from rlaif.training.dpo_qwen import MODEL_ID, adapter_fingerprint, validate_sft
from rlaif.preferences.gen_pairs import atomic_json, select_turn


def identity_for(args, splits, sft_fingerprint):
    excluded = {"out_dir", "adapter", "gcs_output", "resume_from", "dry_run"}
    hyperparameters = {key: value for key, value in vars(args).items() if key not in excluded}
    root = Path(__file__).resolve().parents[2]
    source_files = {
        "training/ppo_qwen.py": Path(__file__).resolve(),
        "training/ppo_core.py": root / "training" / "ppo_core.py",
        "rewards/ppo_reward.py": root / "rewards" / "ppo_reward.py",
        "shared/prompts.py": root / "shared" / "prompts.py",
        "preferences/gen_pairs.py": root / "preferences" / "gen_pairs.py",
        "evaluation/eval_core.py": root / "evaluation" / "eval_core.py",
        "shared/data.py": root / "shared" / "data.py",
    }
    return {"model": args.model, "source_sha256": splits["source_sha256"],
            "split_seed": splits["seed"], "sft_adapter_sha256": sft_fingerprint,
            "hyperparameters": hyperparameters,
            "source_files": {name: D.file_sha256(path) for name, path in source_files.items()},
            "objective": "clipped token PPO + clipped value loss; signed sampled SFT KL rewards",
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
            "episode": "single response terminated by EOS or fixed token horizon"}


def select_scenarios(splits, records, seed):
    """Same deterministic decision-turn policy as preference generation, never heldout."""
    result = []
    for did in splits["pair_gen"]:
        turns, turn = select_turn(records[did], seed, "decision")
        if turn is not None:
            result.append((did, turn))
    random.Random(seed).shuffle(result)
    return result


def check_config(args):
    for key in ("max_updates", "max_scenarios", "max_rollouts", "max_wall_seconds",
                "prefixes_per_update", "ppo_epochs", "max_new_tokens", "max_prompt_tokens",
                "max_judge_calls", "max_judge_tokens", "max_judge_output_tokens",
                "learning_rate", "value_learning_rate", "kl_coef", "max_reference_kl",
                "max_update_kl", "max_ratio", "max_grad_norm", "clip_range", "value_clip"):
        if getattr(args, key) <= 0:
            raise ValueError("--%s must be positive" % key.replace("_", "-"))
    if args.candidates_per_prefix != 4:
        raise ValueError("This experiment samples exactly four on-policy replies per prefix")
    if args.max_rollouts < args.candidates_per_prefix or args.max_scenarios < args.prefixes_per_update:
        raise ValueError("Caps are too small for one rollout batch")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gamma and GAE lambda must be in [0,1]")
    if not 0 < args.clip_range < 1 or args.value_coef < 0:
        raise ValueError("clip_range must be in (0,1); value_coef must be nonnegative")
    if not 0 <= args.judge_thinking_budget < args.max_judge_output_tokens:
        raise ValueError("Judge thinking budget must be below its output token cap")
    if args.max_new_tokens > 256 or args.max_prompt_tokens > 4096:
        raise ValueError("Response/prompt caps exceed the audited single-A100 limits")


def forward_response(policy, value_head, prompt_ids, response_ids, device):
    """State at position p-1 predicts action token p; EOS is a learned action too."""
    import torch
    from rlaif.training.ppo_core import token_logprobs, finite
    prompt_length = len(prompt_ids)
    if prompt_length < 1 or len(response_ids) < 1:
        raise ValueError("Prompt and response must each have at least one token")
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    # The final response token is a target, not an input needed to predict itself.
    output = policy(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
                    output_hidden_states=value_head is not None, use_cache=False)
    start = prompt_length - 1
    logits = output.logits[0, start:start + len(response_ids)]
    actions = torch.tensor(response_ids, device=device)
    logprobs = token_logprobs(logits, actions)
    values = None
    if value_head is not None:
        hidden = output.hidden_states[-1][0, start:start + len(response_ids)]
        values = value_head(hidden.float()).squeeze(-1)
        finite("value predictions", values)
    finite("response log probabilities", logprobs)
    return logprobs, values


def batch_tensors(rollouts, device):
    import torch
    maximum = max(len(row["response_ids"]) for row in rollouts)
    mask = torch.zeros((len(rollouts), maximum), dtype=torch.bool, device=device)
    old = torch.zeros_like(mask, dtype=torch.float32)
    values, refs = torch.zeros_like(old), torch.zeros_like(old)
    scores = torch.tensor([row["reward"] for row in rollouts], dtype=torch.float32, device=device)
    for i, row in enumerate(rollouts):
        n = len(row["response_ids"])
        mask[i, :n] = True
        old[i, :n] = row["old_logprobs"].to(device)
        values[i, :n] = row["old_values"].to(device)
        refs[i, :n] = row["reference_logprobs"].to(device)
    return old.detach(), values.detach(), refs.detach(), scores, mask


def update_policy(policy, value_head, optimizer, rollouts, args, device):
    """One batch has one immutable old snapshot throughout all PPO epochs."""
    import torch
    from rlaif.training.ppo_core import (finite, generalized_advantage, masked_mean, ppo_loss,
                          token_rewards, whiten_advantages)
    old, old_values, refs, scores, mask = batch_tensors(rollouts, device)
    rewards, signed_kl, k3 = token_rewards(scores, old, refs, mask, args.kl_coef)
    reference_kl = float(masked_mean(k3, mask))
    if reference_kl > args.max_reference_kl:
        raise RuntimeError("Frozen-SFT KL gate exceeded before update: %.6f" % reference_kl)
    advantages, returns = generalized_advantage(rewards, old_values, mask, args.gamma, args.gae_lambda)
    advantages = whiten_advantages(advantages, mask)
    total_tokens = int(mask.sum())
    diagnostics = {"reward_mean": float(scores.mean()), "response_tokens": total_tokens,
                   "reference_kl_k3": reference_kl,
                   "reference_kl_signed": float(masked_mean(signed_kl, mask)),
                   "kl_reward_sum_mean": float((-args.kl_coef * signed_kl).sum(-1).mean()),
                   "epochs": []}
    policy.train()
    value_head.train()
    parameters = [p for p in policy.parameters() if p.requires_grad] + list(value_head.parameters())
    for epoch in range(args.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(rollouts)))
        random.shuffle(order)
        epoch_loss = 0.0
        for i in order:
            row = rollouts[i]
            logprobs, values = forward_response(policy, value_head, row["prompt_ids"],
                                               row["response_ids"], device)
            n = len(row["response_ids"])
            loss, before = ppo_loss(logprobs.unsqueeze(0), values.unsqueeze(0),
                old[i:i+1, :n], old_values[i:i+1, :n], advantages[i:i+1, :n],
                returns[i:i+1, :n], mask[i:i+1, :n], args.clip_range, args.value_clip,
                args.value_coef)
            if epoch == 0 and max(abs(float(before["ratio_min"]) - 1),
                                  abs(float(before["ratio_max"]) - 1)) > 1e-3:
                raise RuntimeError("Old-policy snapshot disagrees before the first optimizer step")
            weighted_loss = loss * n / total_tokens
            weighted_loss.backward()
            epoch_loss += float(weighted_loss.detach())
        for parameter in parameters:
            if parameter.grad is not None:
                finite("gradient", parameter.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        finite("gradient norm", grad_norm)
        optimizer.step()
        # Re-measure after every optimizer step. A failed update is never committed.
        combined = {}
        with torch.no_grad():
            for i, row in enumerate(rollouts):
                new, values = forward_response(policy, value_head, row["prompt_ids"], row["response_ids"], device)
                n = len(row["response_ids"])
                _, measures = ppo_loss(new.unsqueeze(0), values.unsqueeze(0), old[i:i+1, :n],
                    old_values[i:i+1, :n], advantages[i:i+1, :n], returns[i:i+1, :n],
                    mask[i:i+1, :n], args.clip_range, args.value_clip, args.value_coef)
                for key, value in measures.items():
                    number = float(value)
                    if key == "ratio_min":
                        combined[key] = min(combined.get(key, float("inf")), number)
                    elif key == "ratio_max":
                        combined[key] = max(combined.get(key, 0), number)
                    else:
                        combined[key] = combined.get(key, 0) + number * n / total_tokens
        combined.update(epoch=epoch + 1, loss=epoch_loss, pre_clip_grad_norm=float(grad_norm))
        diagnostics["epochs"].append(combined)
        if (combined["approx_kl_old_new"] > args.max_update_kl or
                combined["ratio_max"] > args.max_ratio or combined["ratio_min"] < 1 / args.max_ratio):
            raise RuntimeError("PPO update KL/ratio gate exceeded: " + json.dumps(combined))
    return diagnostics


def capture_rng():
    import torch
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(root, policy, value_head, optimizer, tokenizer, state, identity):
    """Atomic directory publish; checkpoints are committed only between full updates."""
    import torch
    destination = root / ("checkpoint-%04d" % state["updates"])
    temporary = root / (destination.name + ".tmp")
    if temporary.exists():
        import shutil
        shutil.rmtree(temporary)
    temporary.mkdir()
    policy.save_pretrained(str(temporary / "adapter"))
    tokenizer.save_pretrained(str(temporary / "adapter"))
    torch.save({"value_head": value_head.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": capture_rng(), "state": state, "identity": identity}, temporary / "training_state.pt")
    atomic_json(temporary / "checkpoint_meta.json", {"state": state, "identity": identity})
    os.replace(temporary, destination)
    atomic_json(root / "latest_checkpoint.json", {"path": destination.name, "updates": state["updates"]})
    # Keep the starting checkpoint and the last two accepted updates locally.
    # Previously uploaded historical checkpoints remain available in GCS.
    checkpoints = sorted(p for p in root.glob("checkpoint-*") if p.is_dir() and
                         p.name.removeprefix("checkpoint-").isdigit() and p.name != "checkpoint-0000")
    for stale in checkpoints[:-2]:
        import shutil
        shutil.rmtree(stale)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=None,
                        help="Pinned Hugging Face commit or tag; historical run was unpinned")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gcs-output")
    parser.add_argument("--resume-from", help="Previous run directory or gs:// URI; caps/config must match")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--project", default="portfolioproject-505823")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--judge-model", default="gemini-2.5-pro")
    parser.add_argument("--max-updates", type=int, default=50)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--max-rollouts", type=int, default=200)
    parser.add_argument("--max-wall-seconds", type=int, default=5400)
    parser.add_argument("--prefixes-per-update", type=int, default=1)
    parser.add_argument("--candidates-per-prefix", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--value-learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--max-reference-kl", type=float, default=1.0)
    parser.add_argument("--max-update-kl", type=float, default=0.05)
    parser.add_argument("--max-ratio", type=float, default=4.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--max-judge-calls", type=int, default=400)
    parser.add_argument("--max-judge-tokens", type=int, default=4000000)
    parser.add_argument("--max-judge-output-tokens", type=int, default=2048)
    parser.add_argument("--judge-thinking-budget", type=int, default=1024)
    args = parser.parse_args()
    check_config(args)
    root = args.out_dir.resolve()
    if args.resume_from:
        if args.resume_from.startswith("gs://"):
            from cloud.artifacts import download_tree
            # Fetch the committed checkpoint and cumulative journals, rather than
            # every historical optimizer snapshot ever uploaded to this run.
            uri = args.resume_from.rstrip("/")
            download_tree(uri + "/run_meta.json", root)
            download_tree(uri + "/latest_checkpoint.json", root)
            latest = json.loads((root / "latest_checkpoint.json").read_text())
            checkpoint_name = latest["path"]
            if not checkpoint_name.startswith("checkpoint-") or not checkpoint_name.removeprefix("checkpoint-").isdigit():
                raise ValueError("Unsafe checkpoint pointer in resume source")
            download_tree(uri + "/" + checkpoint_name, root / checkpoint_name)
            try:
                download_tree(uri + "/rewards", root / "rewards")
            except FileNotFoundError:
                if latest["updates"] != 0:
                    raise ValueError("Resume requires the cumulative reward/compute usage ledgers")
            try:
                download_tree(uri + "/training_log.jsonl", root)
            except FileNotFoundError:
                if latest["updates"] != 0:
                    raise ValueError("Resume requires the committed training log")
        elif Path(args.resume_from).resolve() != root:
            import shutil
            shutil.copytree(args.resume_from, root, dirs_exist_ok=False)
    elif root.exists() and any(root.iterdir()):
        parser.error("Fresh run output directory must be empty; use --resume-from explicitly")
    root.mkdir(parents=True, exist_ok=True)
    cache = None
    if args.adapter.startswith("gs://"):
        from cloud.artifacts import download_tree
        uri = args.adapter.rstrip("/")
        if not uri.endswith("/adapter"):
            parser.error("--adapter GCS URI must end /adapter below a completed SFT run")
        cache = tempfile.TemporaryDirectory(prefix="rlaif-ppo-sft-")
        local = Path(cache.name)
        download_tree(uri, local / "adapter")
        download_tree(uri.rsplit("/", 1)[0] + "/run_meta.json", local)
        args.adapter = str(local / "adapter")
    splits = D.load_splits()
    records = {r["dialogue_id"]: r for r in D.load_dialogues()}
    validate_sft(args.adapter, splits, args.model)
    scenarios = select_scenarios(splits, records, args.seed)
    if args.max_scenarios > len(scenarios):
        parser.error("--max-scenarios exceeds the eligible disjoint training prefixes")
    identity = identity_for(args, splits, adapter_fingerprint(args.adapter))
    metadata_file = root / "run_meta.json"
    if args.resume_from:
        if not metadata_file.is_file() or json.loads(metadata_file.read_text())["identity"] != identity:
            parser.error("Resume config/source/SFT provenance changed; use the original arguments")
    else:
        atomic_json(metadata_file, {"identity": identity, "started_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if args.dry_run:
        print(json.dumps({"eligible_scenarios": len(scenarios), "identity": identity}, indent=2))
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    from rlaif.training.ppo_core import reference_kl, finite
    from rlaif.rewards.ppo_reward import GroundedRewardAdapter, PPOVertexJudge
    from cloud.artifacts import sync_tree
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("PPO requires the audited CUDA bf16 GPU runtime")
    set_seed(args.seed)
    device = torch.device("cuda:0")
    resume_checkpoint, restored = None, None
    if args.resume_from:
        relative = json.loads((root / "latest_checkpoint.json").read_text())["path"]
        resume_checkpoint = (root / relative).resolve()
        if not resume_checkpoint.is_relative_to(root):
            raise ValueError("Unsafe checkpoint path")
        restored = torch.load(resume_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if restored["identity"] != identity:
            raise ValueError("Checkpoint identity disagrees with run metadata")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, str(resume_checkpoint / "adapter") if restored
                                       else args.adapter, is_trainable=True).to(device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.enable_input_require_grads()
    reference = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16), args.adapter,
        is_trainable=False).to(device)
    reference.requires_grad_(False)
    reference.eval()
    reference.config.use_cache = False
    # Dropout would make the old/new ratio depend on RNG rather than policy change.
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    value_head = torch.nn.Linear(policy.config.hidden_size, 1, dtype=torch.float32, device=device)
    torch.nn.init.zeros_(value_head.weight)
    torch.nn.init.zeros_(value_head.bias)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in policy.parameters() if p.requires_grad], "lr": args.learning_rate},
        {"params": list(value_head.parameters()), "lr": args.value_learning_rate}], weight_decay=0.0)
    state = {"updates": 0, "next_scenario": 0, "rollouts": 0, "dropped_rewards": 0,
             "wall_elapsed_seconds": 0.0}
    if restored:
        value_head.load_state_dict(restored["value_head"])
        optimizer.load_state_dict(restored["optimizer"])
        state = restored["state"]
        restore_rng(restored["rng"])
    started = time.monotonic()
    compute_usage_path = root / "rewards" / "rollout_usage.json"
    compute_usage = json.loads(compute_usage_path.read_text()) if compute_usage_path.exists() else {
        "attempts": 0, "wall_elapsed_seconds": 0.0,
        "max_rollouts": args.max_rollouts, "max_wall_seconds": args.max_wall_seconds}
    if (compute_usage["max_rollouts"] != args.max_rollouts or
            compute_usage["max_wall_seconds"] != args.max_wall_seconds):
        raise ValueError("Compute caps disagree with the durable run ledger")
    prior_elapsed = max(state["wall_elapsed_seconds"], compute_usage["wall_elapsed_seconds"])

    def bounded_time():
        state["wall_elapsed_seconds"] = prior_elapsed + time.monotonic() - started
        compute_usage["wall_elapsed_seconds"] = state["wall_elapsed_seconds"]
        atomic_json(compute_usage_path, compute_usage)
        if state["wall_elapsed_seconds"] >= args.max_wall_seconds:
            raise RuntimeError("PPO wall-time cap reached; last complete checkpoint preserved")

    def sync_reward():
        if args.gcs_output:
            sync_tree(root / "rewards", args.gcs_output.rstrip("/") + "/rewards")

    reward_dir = root / "rewards"
    reward_dir.mkdir(exist_ok=True)
    judge = PPOVertexJudge(project=args.project, location=args.location, model=args.judge_model,
        usage_path=reward_dir / "judge_usage.json", max_calls=args.max_judge_calls,
        max_tokens=args.max_judge_tokens, max_output_tokens=args.max_judge_output_tokens,
        thinking_budget=args.judge_thinking_budget, sync=sync_reward)
    rewards_adapter = GroundedRewardAdapter(judge, reward_dir, sync=sync_reward)
    generation_config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        num_beams=1, repetition_penalty=1.0, max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True)
    if not restored:
        save_checkpoint(root, policy, value_head, optimizer, tokenizer, state, identity)
        if args.gcs_output:
            sync_tree(root, args.gcs_output)
    completed = False
    failure = None
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    def worker_terminated(signum, frame):
        raise RuntimeError("PPO worker terminated; the last accepted checkpoint is resumable")
    signal.signal(signal.SIGTERM, worker_terminated)
    try:
        while (state["updates"] < args.max_updates and state["next_scenario"] < args.max_scenarios
               and compute_usage["attempts"] + args.candidates_per_prefix <= args.max_rollouts):
            bounded_time()
            rollouts = []
            batch_start_cursor = state["next_scenario"]
            policy.eval()
            value_head.eval()
            prefixes = 0
            while (prefixes < args.prefixes_per_update and state["next_scenario"] < args.max_scenarios
                   and compute_usage["attempts"] + args.candidates_per_prefix <= args.max_rollouts):
                bounded_time()
                did, turn = scenarios[state["next_scenario"]]
                turns = D.parse_turns(records[did]["dialogue"])
                messages = D.build_messages(records[did], turns, turn)
                prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                if len(prompt_ids) > args.max_prompt_tokens:
                    raise RuntimeError("Prompt would truncate the KB; raise token limit after memory review")
                inputs = torch.tensor([prompt_ids], device=device)
                policy_version = identity["sft_adapter_sha256"] + ":update%d" % state["updates"]
                for candidate in range(args.candidates_per_prefix):
                    bounded_time()
                    # Reserve compute before generation, including uncommitted tails.
                    # This ledger travels with the reward journals on resume.
                    compute_usage["attempts"] += 1
                    atomic_json(compute_usage_path, compute_usage)
                    sync_reward()
                    rollout_id = "update%d:scenario%d:candidate%d" % (state["updates"], did, candidate)
                    with torch.no_grad():
                        generated = policy.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                                    generation_config=generation_config)
                        response_ids = generated[0, len(prompt_ids):].tolist()
                        if not response_ids:
                            raise RuntimeError("Policy produced no response tokens")
                        old_logprobs, old_values = forward_response(policy, value_head, prompt_ids, response_ids, device)
                        ref_logprobs, _ = forward_response(reference, None, prompt_ids, response_ids, device)
                        finite("old-policy values", old_values)
                        reference_kl(old_logprobs, ref_logprobs, torch.ones_like(old_logprobs, dtype=torch.bool))
                    text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    result = rewards_adapter.score(records[did], messages, text, policy_version, rollout_id)
                    state["rollouts"] += 1
                    if result["status"] != "complete" or result.get("reward") is None:
                        state["dropped_rewards"] += 1
                        continue
                    reward = float(result["reward"])
                    if not -1 <= reward <= 1:
                        raise RuntimeError("Grounded reward is nonfinite or outside the declared [-1,1] scale")
                    rollouts.append({"scenario_id": did, "turn_index": turn,
                        "rollout_id": rollout_id, "policy_version": policy_version,
                        "prompt_ids": prompt_ids, "response_ids": response_ids,
                        "response": text, "reward": reward,
                        "old_logprobs": old_logprobs.detach().cpu().clone(),
                        "old_values": old_values.detach().cpu().clone(),
                        "reference_logprobs": ref_logprobs.detach().cpu().clone(),
                        "ended_with_eos": response_ids[-1] == tokenizer.eos_token_id,
                        "reward_fingerprint": result.get("fingerprint")})
                state["next_scenario"] += 1
                prefixes += 1
            if len(rollouts) < 2:
                raise RuntimeError("Fewer than two valid rollouts; reward parsing cannot support a PPO update")
            if state["dropped_rewards"] / max(1, state["rollouts"]) >= 0.1:
                raise RuntimeError("Reward drop rate reached 10%; inspect judge evidence before continuing")
            # Immutable snapshots allow auditing actual old-policy likelihoods/actions.
            torch.save(rollouts, root / ("rollouts-%04d.pt" % (state["updates"] + 1)))
            measures = update_policy(policy, value_head, optimizer, rollouts, args, device)
            bounded_time()
            state["updates"] += 1
            measures.update(update=state["updates"], scenarios_start=batch_start_cursor,
                            scenarios_end=state["next_scenario"], valid_rollouts=len(rollouts),
                            dropped_rewards=state["dropped_rewards"], wall_seconds=state["wall_elapsed_seconds"])
            with (root / "training_log.jsonl").open("a") as stream:
                stream.write(json.dumps(measures) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print("[ppo-update]", json.dumps(measures), flush=True)
            save_checkpoint(root, policy, value_head, optimizer, tokenizer, state, identity)
            if args.gcs_output:
                sync_tree(root, args.gcs_output)
        if state["updates"] == 0:
            raise RuntimeError("Caps allowed no completed PPO updates")
        policy.save_pretrained(str(root / "adapter"))
        tokenizer.save_pretrained(str(root / "adapter"))
        torch.save(value_head.state_dict(), root / "value_head.pt")
        completed = True
    except BaseException as exc:
        failure = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
        state["wall_elapsed_seconds"] = prior_elapsed + time.monotonic() - started
        compute_usage["wall_elapsed_seconds"] = state["wall_elapsed_seconds"]
        atomic_json(compute_usage_path, compute_usage)
        metadata = json.loads(metadata_file.read_text())
        metadata.update(state=state, completed=completed, failure=failure,
            compute_usage=compute_usage, reward_protocol_sha256=rewards_adapter.protocol_hash,
            finished_utc=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), package_versions={name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate")})
        atomic_json(metadata_file, metadata)
        if args.gcs_output:
            sync_tree(root, args.gcs_output)
    print("[done] PPO adapter ->", root / "adapter", flush=True)


if __name__ == "__main__":
    main()

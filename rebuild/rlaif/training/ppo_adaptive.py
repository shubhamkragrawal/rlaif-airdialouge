"""Atomic bounded PPO retries; rejected updates cannot mutate committed state."""
import copy


class UpdateRejected(RuntimeError):
    """A measured update violated hard gates or stricter acceptance headroom."""


def validate_parent(metadata, splits, sft_hash, model, seed, starting_update):
    identity, state = metadata["identity"], metadata["state"]
    if (identity["model"] != model or identity["source_sha256"] != splits["source_sha256"] or
            identity["split_seed"] != splits["seed"] or identity["sft_adapter_sha256"] != sft_hash or
            identity["hyperparameters"]["seed"] != seed):
        raise ValueError("Parent model/data/split/SFT/ordering provenance mismatch")
    if state["updates"] != starting_update or state["next_scenario"] != starting_update:
        raise ValueError("Parent must have19 accepted updates over the first19 ordered prefixes")


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def transactional_update(policy, value_head, optimizer, rollouts, args, device,
                         updater, record_attempt=lambda evidence: None, check_time=lambda: None,
                         prior_attempts=0):
    """Restore policy/value/Adam moments/RNG before each retry of one old snapshot.

    Snapshot only trainable policy parameters: frozen model/reference weights are
    never modified by the optimizer. CPU snapshots avoid an extra GPU model copy.
    Retry LR decay applies to both optimizer groups; lower LR persists after an
    accepted retry. Every non-acceptance restores state before recording/raising.
    """
    import torch
    from rlaif.training.ppo_qwen import capture_rng, restore_rng
    parameters = {name: parameter for name, parameter in policy.named_parameters() if parameter.requires_grad}
    policy_state = {name: cpu_copy(parameter) for name, parameter in parameters.items()}
    value_state, optimizer_state, rng = cpu_copy(value_head.state_dict()), cpu_copy(optimizer.state_dict()), capture_rng()
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    evidence = []

    def rollback():
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(policy_state[name])
        value_head.load_state_dict(value_state)
        # load_state_dict can retain CPU tensor references. A fresh clone prevents
        # Adam's in-place moment updates from mutating the rollback snapshot.
        optimizer.load_state_dict(cpu_copy(optimizer_state))
        optimizer.zero_grad(set_to_none=True)
        restore_rng(rng)

    if not 0 <= prior_attempts < args.max_update_attempts:
        raise UpdateRejected("Durable update retry cap exhausted; committed state unchanged")
    for attempt in range(prior_attempts, args.max_update_attempts):
        rollback()
        check_time()
        scale = args.retry_lr_decay ** attempt
        lrs = [lr * scale for lr in initial_lrs]
        if lrs[0] < args.min_learning_rate:
            raise UpdateRejected("Retry learning rate floor reached; committed state restored")
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr
        try:
            # Persist a reservation before computing the update, so interruption
            # cannot reset its retry counter on a later checkpoint resume.
            record_attempt({"attempt": attempt + 1, "status": "reserved", "learning_rates": lrs})
            result = updater(policy, value_head, optimizer, rollouts, args, device)
            check_time()
        except BaseException as exc:
            rollback()
            item = {"attempt": attempt + 1, "status": "rolled_back", "learning_rates": lrs,
                    "error": "%s: %s" % (type(exc).__name__, exc)}
            evidence.append(item)
            record_attempt(item)
            if not isinstance(exc, UpdateRejected):
                raise
            if attempt + 1 == args.max_update_attempts:
                raise UpdateRejected("Bounded PPO retries exhausted; committed state restored") from exc
        else:
            item = {"attempt": attempt + 1, "status": "accepted", "learning_rates": lrs}
            evidence.append(item)
            try:
                record_attempt(item)
            except BaseException:
                rollback()
                raise
            result["adaptive_attempts"] = evidence
            result["effective_learning_rates"] = lrs
            return result
    raise AssertionError("Unreachable bounded retry state")

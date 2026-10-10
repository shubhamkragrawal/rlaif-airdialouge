"""Auditable token-level PPO primitives. Prompt and padding tokens never enter losses."""
import torch


def finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Nonfinite " + name)


def masked_mean(value, mask):
    mask = mask.to(dtype=torch.bool)
    if mask.sum() == 0:
        raise ValueError("Empty response-token mask")
    # Select rather than multiply: masked NaNs must not poison the loss.
    return value[mask].mean()


def token_logprobs(logits, actions):
    """Log probability of sampled actions; all vocabulary entries enter the denominator."""
    logits = logits.float()
    return logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


def reference_kl(old_logprobs, reference_logprobs, mask):
    """Unbiased sampled log-ratio KL and nonnegative k3 diagnostic under old policy.

    For actions sampled from pi_old, E[log pi_old(a)-log pi_ref(a)] is
    KL(pi_old || pi_ref). The signed per-action estimator shapes rewards. The
    k3 estimator exp(log pi_ref-log pi_old)-1-(log pi_ref-log pi_old) has the
    same expectation and is logged for a stable nonnegative diagnostic.
    """
    finite("policy log probabilities", old_logprobs[mask.bool()])
    finite("reference log probabilities", reference_logprobs[mask.bool()])
    signed = old_logprobs - reference_logprobs
    finite("reference log ratio", signed[mask.bool()])
    k3 = torch.expm1(-signed) + signed
    finite("reference KL estimator", k3[mask.bool()])
    return signed.masked_fill(~mask.bool(), 0), k3.masked_fill(~mask.bool(), 0)


def token_rewards(score, old_logprobs, reference_logprobs, mask, kl_coef):
    if kl_coef < 0:
        raise ValueError("KL coefficient must be nonnegative")
    signed, k3 = reference_kl(old_logprobs, reference_logprobs, mask)
    reward = -kl_coef * signed
    for row in range(mask.shape[0]):
        indices = mask[row].nonzero().flatten()
        if len(indices) == 0:
            raise ValueError("Every rollout requires response tokens")
        reward[row, indices[-1]] += score[row]
    finite("token rewards", reward[mask.bool()])
    return reward, signed, k3


def generalized_advantage(rewards, old_values, mask, gamma=1.0, lam=0.95):
    """GAE over finite response episodes, zero bootstrap after EOS/token horizon.

    A response capped at max_new_tokens is an explicitly defined finite episode,
    rather than a continuation whose unobserved future value is fabricated.
    """
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0,1]")
    finite("old values", old_values[mask.bool()])
    advantages = torch.zeros_like(old_values)
    for row in range(mask.shape[0]):
        indices = mask[row].nonzero().flatten()
        if len(indices) == 0 or not torch.equal(indices, torch.arange(len(indices), device=mask.device)):
            raise ValueError("Response masks must be a nonempty contiguous prefix")
        last_advantage = old_values.new_zeros(())
        next_value = old_values.new_zeros(())
        for col in reversed(indices.tolist()):
            delta = rewards[row, col] + gamma * next_value - old_values[row, col]
            last_advantage = delta + gamma * lam * last_advantage
            advantages[row, col] = last_advantage
            next_value = old_values[row, col]
    returns = (advantages + old_values).masked_fill(~mask.bool(), 0)
    finite("advantages", advantages[mask.bool()])
    finite("returns", returns[mask.bool()])
    return advantages.detach(), returns.detach()


def whiten_advantages(advantages, mask):
    selected = advantages[mask.bool()]
    if len(selected) < 2:
        raise ValueError("At least two response tokens required to normalize advantages")
    normalized = (advantages - selected.mean()) / torch.sqrt(selected.var(unbiased=False) + 1e-8)
    return normalized.masked_fill(~mask.bool(), 0).detach()


def ppo_loss(new_logprobs, new_values, old_logprobs, old_values, advantages, returns,
             mask, clip_range=0.2, value_clip=0.2, value_coef=0.5):
    """Clipped policy surrogate and clipped value regression with detached snapshots."""
    old_logprobs, old_values = old_logprobs.detach(), old_values.detach()
    advantages, returns = advantages.detach(), returns.detach()
    # Eliminate padding BEFORE exponentiation/regression; NaN*0 can otherwise
    # poison gradients even when a later masked mean excludes the padded position.
    masked_out = ~mask.bool()
    new_logprobs = new_logprobs.masked_fill(masked_out, 0)
    new_values = new_values.masked_fill(masked_out, 0)
    old_logprobs = old_logprobs.masked_fill(masked_out, 0)
    old_values = old_values.masked_fill(masked_out, 0)
    advantages = advantages.masked_fill(masked_out, 0)
    returns = returns.masked_fill(masked_out, 0)
    log_ratio = new_logprobs - old_logprobs
    finite("new log probabilities", new_logprobs[mask.bool()])
    finite("new values", new_values[mask.bool()])
    finite("log ratio", log_ratio[mask.bool()])
    if log_ratio[mask.bool()].abs().max() > 20:
        raise FloatingPointError("Policy likelihood ratio exceeds the numerical safety bound")
    ratio = log_ratio.exp()
    policy_unclipped = -advantages * ratio
    policy_clipped = -advantages * ratio.clamp(1 - clip_range, 1 + clip_range)
    policy_loss = masked_mean(torch.maximum(policy_unclipped, policy_clipped), mask)
    clipped_values = old_values + (new_values - old_values).clamp(-value_clip, value_clip)
    value_loss = 0.5 * masked_mean(torch.maximum((new_values - returns).square(),
                                                (clipped_values - returns).square()), mask)
    loss = policy_loss + value_coef * value_loss
    finite("PPO loss", loss)
    diagnostics = {
        "policy_loss": policy_loss.detach(), "value_loss": value_loss.detach(),
        "approx_kl_old_new": masked_mean(torch.expm1(log_ratio) - log_ratio, mask).detach(),
        "ratio_mean": masked_mean(ratio, mask).detach(),
        "ratio_min": ratio[mask.bool()].min().detach(),
        "ratio_max": ratio[mask.bool()].max().detach(),
        "clip_fraction": masked_mean(((ratio - 1).abs() > clip_range).float(), mask).detach(),
        "value_clip_fraction": masked_mean(((new_values - old_values).abs() > value_clip).float(), mask).detach(),
    }
    return loss, diagnostics

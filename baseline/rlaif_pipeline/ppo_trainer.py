 
import logging
import os
from typing import List
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
 
logger = logging.getLogger(__name__)
 
PPO_LR              = 1e-5
PPO_CLIP_EPS        = 0.2
PPO_VALUE_COEF      = 0.5
PPO_KL_COEF         = 0.05
PPO_GAMMA           = 0.99
PPO_GRAD_CLIP       = 1.0 
PPO_VALUE_GRAD_CLIP = 0.1
PPO_ACCUM_STEPS     = 3
LOG_RATIO_CLAMP     = 10.0
 

class ValueHead(nn.Module):
    """Single linear layer: hidden_dim => 1 scalar per token position."""
 
    def __init__(self, hidden_dim: int, dtype: torch.dtype):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=False).to(dtype)
        
        # Better to have small init!! 
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)
 
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states).squeeze(-1)
 
 
class PPOTrainer:
 
    def __init__(self, agent_model: PreTrainedModel, ref_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str = "cuda"):
        self.device = device
         # reads from model (bfloat16 if supported, float16 otherwise)!!! Same as agent setting!
        self.dtype  = agent_model.dtype
        logger.info(f"[PPO INIT] model dtype = {self.dtype}")

        
        self.tokenizer = tokenizer
        self.policy    = agent_model
        self.policy.train()
 
        
        try:
            self.policy.enable_input_require_grads()
            logger.info("[PPO INIT] enable_input_require_grads() succeeded")

        except Exception as e:
            logger.warning(f"[PPO INIT] enable_input_require_grads() failed ({e}) — continuing with manual lora_ loop")
 
        lora_params_found = 0
        # Loop just to detect the lora (Low Rank Adaptation) parameters and update only those!!!
        for name, param in self.policy.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                lora_params_found += 1
        logger.info(f"[PPO INIT] Re-enabled requires_grad on {lora_params_found} LoRA parameter tensors")
 

        hidden_dim = agent_model.config.hidden_size
        self.value_head = ValueHead(hidden_dim, self.dtype).to(device)
        self.value_head.train()
        logger.info(f"[PPO INIT] ValueHead created: hidden_dim={hidden_dim}, dtype={self.dtype}")
 
        # Reference model (frozen SFT)...Will mainlybe used for KL divergence calculation later!!!
        self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        logger.info("[PPO INIT] Reference model frozen")
 
        # Collect trainable params
        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        value_params  = list(self.value_head.parameters())
 
        logger.info(f"[PPO INIT] Trainable policy tensors : {len(policy_params)}")
        logger.info(f"[PPO INIT] Trainable policy params  : {sum(p.numel() for p in policy_params):,}")
        logger.info(f"[PPO INIT] Value head params         : {sum(p.numel() for p in value_params):,}")
 
        if not policy_params:
            raise RuntimeError(
                "[PPO INIT] FATAL: no trainable policy parameters found after enabling LoRA grads. "
                "The model must be a PeftModel with LoRA adapters."
            )
 
        # Separate optimizers
        # Separate so value head can be clipped independently without
        # interfering with policy gradient norm. It was erroring out to to exploding gradients!
        self.policy_optimizer = torch.optim.AdamW(policy_params, lr=PPO_LR)
        self.value_optimizer  = torch.optim.AdamW(value_params,  lr=PPO_LR * 0.1)
        self._accum_count      = 0
        self._value_grad_count = 0

        # To keep track for updates!!! Just like a coutner!!
        self._value_grad_count = 0
 
        logger.info(f"[PPO INIT] PPOTrainer ready | device={device} | accum_steps={PPO_ACCUM_STEPS}")
 
    def step(self, episode_logprobs: List[torch.Tensor], reward: float, input_ids_list: List[torch.Tensor] = None) -> dict:
 
        if not episode_logprobs:
            logger.warning("[PPO STEP] Called with empty logprobs — skipping.")
            return {}
        
        # Normalizing rewards!!!
        reward_normalised = (float(reward) - 5.5) / 4.5
        logger.info(f"[PPO STEP] raw_score={reward} → normalised={reward_normalised:+.4f}")
 
        # Old log probs from rollout => float16 on GPU
        old_logprobs = torch.cat(episode_logprobs, dim=0).to(device=self.device, dtype=self.dtype)
        T = old_logprobs.shape[0]
        logger.info(f"[PPO STEP] T={T} tokens across {len(episode_logprobs)} agent turns")
 
        if T == 0:
            logger.warning("[PPO STEP] T=0 after cat — skipping.")
            return {}
 
        rewards_per_token = _discount_rewards(reward_normalised, T, PPO_GAMMA, self.device, self.dtype)
 
        
        value_pred = self._compute_values(input_ids_list, T)
        value_pred_clean = value_pred.detach()
        if torch.isnan(value_pred_clean).any():
            value_pred_clean = torch.zeros(T, device=self.device, dtype=self.dtype)
            logger.warning("[PPO STEP] value baseline had nan — using zero baseline")
 
        advantages = rewards_per_token.clone() - value_pred_clean
        adv_std = advantages.std()
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        logger.info(f"[PPO STEP] advantage std={adv_std.item():.4f} mean={rewards_per_token.mean().item():.4f} value_baseline_mean={value_pred_clean.mean().item():.4f}")
 
        
        new_logprobs = self._compute_new_logprobs(input_ids_list, T)
        logger.info(f"[PPO STEP] new_logprobs: mean={new_logprobs.mean().item():.4f} requires_grad={new_logprobs.requires_grad}")
 
        # PPO ratio (clamped to prevent float16 overflow)
        log_ratio = (new_logprobs - old_logprobs.detach()).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        ratio     = torch.exp(log_ratio)
        logger.info(f"[PPO STEP] ratio: mean={ratio.mean().item():.4f} max={ratio.max().item():.4f}")
 
        # PPO clipped loss
        unclipped   = ratio * advantages
        clipped     = ratio.clamp(1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # KL LOSS!!! to know how much the new model has deviated from the old responses!!!
        kl_loss = self._kl_penalty(new_logprobs)
        logger.info(f"[PPO STEP] policy_loss={policy_loss.item():.4f} kl_loss={kl_loss.item():.4f}")
 
        # Policy backward (value excluded to prevent nan Erroes)!!!
        policy_total = (policy_loss + PPO_KL_COEF * kl_loss) / PPO_ACCUM_STEPS
        policy_total.backward()
        logger.info(f"[PPO STEP] Policy backward done")
 
        
        # linear layer receives gradients here!!!
        value_target  = rewards_per_token.detach()
        w_norm = self.value_head.linear.weight.norm().item()
        logger.info(f"[PPO STEP] value_head weight_norm={w_norm:.4f}")

        # value_pred already computed above for advantage baseline - reuse it
        value_loss_val = float("nan")
        value_loss_ok  = False

        # Chekc if NaN is generated!!!
        nan_pred = torch.isnan(value_pred).any() or torch.isinf(value_pred).any()
        logger.info(f"[PPO STEP] value_pred: mean={value_pred.mean().item():.4f} nan={nan_pred}") if not nan_pred else logger.info(f"[PPO STEP] value_pred: mean=nan nan={nan_pred}")
 
        if not nan_pred:
            # If not generated then go ahead and calculate loss for Value!!!
            value_loss = F.mse_loss(value_pred, value_target)
            value_loss_val = value_loss.item()
            nan_loss = torch.isnan(value_loss) or torch.isinf(value_loss)
            logger.info(f"[PPO STEP] value_loss={value_loss_val:.4f} nan={nan_loss}")
            if not nan_loss:
                (PPO_VALUE_COEF * value_loss / PPO_ACCUM_STEPS).backward()
                value_loss_ok = True
                self._value_grad_count += 1
                logger.info(f"[PPO STEP] Value backward done (_value_grad_count={self._value_grad_count})")
            else:
                logger.warning("[PPO STEP] value_loss is nan/inf — skipping value backward")
        else:
            logger.warning("[PPO STEP] value_pred contains nan/inf — skipping value backward")
 
        self._accum_count += 1
        did_update = False
 
        if self._accum_count >= PPO_ACCUM_STEPS:

            # If we are at the step whcih we decide to accumlate and update gradients....then update them!!
            policy_grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], PPO_GRAD_CLIP)
            
            # Policy backProp! Update weights!
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad()
            logger.info(f"[PPO UPDATE] policy grad_norm={policy_grad_norm.item():.4f}")
 
            # Value step (only if at least one backward fired this cycle)
            if self._value_grad_count > 0:
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.value_head.parameters()),
                    PPO_VALUE_GRAD_CLIP,
                )
                self.value_optimizer.step()
                logger.info(f"[PPO UPDATE] value grad_norm={value_grad_norm.item():.4f} (from {self._value_grad_count} backwards)")
                # Detect if optimizer step corrupted value head weights
                w = self.value_head.linear.weight
                if torch.isnan(w).any() or torch.isinf(w).any():
                    logger.error("[PPO UPDATE] value_head weights became nan/inf after step — reinitialising")
                    nn.init.normal_(w, mean=0.0, std=0.001)
            else:
                logger.warning("[PPO UPDATE] No value backward this cycle - skipping value step")
 
            self.value_optimizer.zero_grad()
            self._accum_count      = 0
            self._value_grad_count = 0
            did_update = True
 
        metrics = {
            "raw_score":   reward,
            "reward": round(reward_normalised, 4),
            "policy_loss": round(policy_loss.item(), 4),
            "value_loss":  round(value_loss_val, 4) if value_loss_ok else "nan",
            "kl_loss": round(kl_loss.item(), 4),
            "total_loss": round((policy_loss + kl_loss).item(), 4),
            "ratio_mean": round(ratio.mean().item(), 4),
            "ratio_max": round(ratio.max().item(), 4),
            "did_update": did_update,
        }
        logger.info(
            f"[PPO STEP] policy={metrics['policy_loss']:.4f} "
            f"value={metrics['value_loss']} "
            f"kl={metrics['kl_loss']:.4f} "
            f"ratio_mean={metrics['ratio_mean']:.4f} "
            f"ratio_max={metrics['ratio_max']:.4f} "
            f"{'[UPDATED]' if did_update else f'[accum {self._accum_count}/{PPO_ACCUM_STEPS}]'}"
        )
        return metrics

    def flush(self):
        """
        Finalizes any pending weight updates that haven't reached the 
        full accumulation threshold. Ensures the very last leftover
        episodes of training are processed and applied to the model 
        before shutdown.
        """
        if self._accum_count > 0:
            logger.info(f"[PPO FLUSH] Flushing {self._accum_count} accumulated steps")
 
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad],
                PPO_GRAD_CLIP,
            )
            self.policy_optimizer.step()
            self.policy_optimizer.zero_grad()
            logger.info(f"[PPO FLUSH] policy grad_norm={policy_grad_norm.item():.4f}")
 
            if self._value_grad_count > 0:
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.value_head.parameters()),
                    PPO_VALUE_GRAD_CLIP,
                )
                self.value_optimizer.step()
                logger.info(f"[PPO FLUSH] value grad_norm={value_grad_norm.item():.4f}")
            self.value_optimizer.zero_grad()
 
            self._accum_count      = 0
            self._value_grad_count = 0
            logger.info("[PPO FLUSH] Done.")
 
    def save(self, path: str):
        # Save model!
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.pt"))
        logger.info(f"[PPO SAVE] Checkpoint saved to {path}")

    def _compute_new_logprobs(self, input_ids_list: List[torch.Tensor], T: int) -> torch.Tensor:
        """
        Re-runs the conversation to calculate the current probability of the 
        actions taken.
        """
        if input_ids_list is None:
            logger.warning("[PPO] _compute_new_logprobs: input_ids_list is None — returning zeros (ratio=1, no gradient)")
            return torch.zeros(T, device=self.device, dtype=self.dtype)
 
        all_new_logprobs = []
        for turn_idx, ids in enumerate(input_ids_list):
            if ids is None or len(ids) < 2:
                logger.warning(f"[PPO] _compute_new_logprobs: turn {turn_idx} has ids=None or len<2 — skipping")
                continue
            ids        = ids.to(self.device)
            input_seq  = ids[:-1].unsqueeze(0)
            target_seq = ids[1:]
 
            outputs   = self.policy(input_seq, output_hidden_states=False)
            logits    = outputs.logits.squeeze(0)
            log_probs = torch.log_softmax(logits, dim=-1)
            token_lps = log_probs[torch.arange(len(target_seq), device=self.device), target_seq]
            all_new_logprobs.append(token_lps)
 
        if not all_new_logprobs:
            logger.warning("[PPO] _compute_new_logprobs: no valid turns — returning zeros")
            return torch.zeros(T, device=self.device, dtype=self.dtype)
 
        result = torch.cat(all_new_logprobs, dim=0)
        logger.info(f"[PPO] new_logprobs shape={result.shape} vs T={T}")
 
        if result.shape[0] < T:
            pad    = torch.zeros(T - result.shape[0], device=self.device, dtype=self.dtype)
            result = torch.cat([result, pad])
        elif result.shape[0] > T:
            result = result[:T]
        return result
 
    def _compute_values(self, input_ids_list: List[torch.Tensor], T: int) -> torch.Tensor:
        """
        Generates the 'Critic's' estimate of the expected future reward for each word in the conversation.
        """
        if input_ids_list is None:
            return torch.zeros(T, device=self.device, dtype=self.dtype)
 
        all_values = []
        for turn_idx, ids in enumerate(input_ids_list):
            if ids is None or len(ids) == 0:
                continue
            ids = ids.to(self.device).unsqueeze(0)
 
            with torch.no_grad():
                outputs = self.policy(ids, output_hidden_states=True)
 
            hidden = outputs.hidden_states[-1].detach()
            values = self.value_head(hidden).squeeze(0)
            all_values.append(values)
 
        if not all_values:
            return torch.zeros(T, device=self.device, dtype=self.dtype)
 
        result = torch.cat(all_values, dim=0)
        if result.shape[0] < T:
            pad    = torch.zeros(T - result.shape[0], device=self.device, dtype=self.dtype)
            result = torch.cat([result, pad])
        elif result.shape[0] > T:
            result = result[:T]
        return result
 
    def _kl_penalty(self, new_logprobs: torch.Tensor) -> torch.Tensor:
        
        return torch.clamp(-new_logprobs.mean(), min=0.0)

def _discount_rewards(reward: float, length: int, gamma: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Spread episode reward backwards. Last token gets full credit."""
    rewards = torch.zeros(length, device=device, dtype=dtype)
    running = reward
    for t in reversed(range(length)):
        rewards[t] = running
        running    = running * gamma
    return rewards
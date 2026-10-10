"""
rlaif_pipeline.py
=====================
Implemented Reinforce + Baseline algorithm with Direct-RLAIF (d-RLAIF) architecture.  
Referenced from publication: "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" — Lee et al., ICML 2024.

Algorithm:
------------------------------------------------------
  1. For each training example (intent + KB):
       a. Run the current policy (Mistral LoRA) to generate an agent response inside a full Gemini-simulated conversation.
       b. Call the Gemini Judge DIRECTLY as the reward signal. The judge returns a scalar score in [-1, 1].
       c. Compute the REINFORCE policy-gradient loss with a learned value baseline:
       d. Add a KL-divergence penalty against the frozen SFT reference:
       e. Update the value head with MSE loss

Architecture:
-------------
  - Policy  : Mistral-7B base  +  LoRA adapter  (from SFT)
  - Reference: same Mistral-7B base  +  frozen SFT LoRA  (for KL)
  - Value head: linear layer on top of last hidden state  (trained from scratch)
  - Simulator : Gemini (Vertex AI) used as a user silmulator, to generate a conversation with Policy agent.
  - Judge   : Gemini (Vertex AI) called live per rollout — no Reward Model training needed


GCS flags
--------------------------------------------------
  --project          GCP project id
  --bucket           GCS bucket name (no gs:// prefix)
  --data_blob        path inside bucket to train .jsonl
  --sft_model_prefix GCS prefix of the SFT LoRA adapter
  --base_model_id    HuggingFace base model id
  --output_prefix    GCS prefix to save trained LoRA adapter
  --gemini_version   Vertex AI Gemini model version
  --beta             KL penalty coefficient  (default 0.05)
  --lr               learning rate           (default 1e-5)
  --epochs           number of passes over the data (default 8)
  --max_turns        max conversation turns per rollout (default 10)
  --max_samples      cap examples for debug runs
  --gradient_accum   gradient accumulation steps (default 4)
  --overwrite        delete existing GCS output before starting

  --project=274690076649
  --bucket=rlaif-data-bucket-rlaif-project
  --data_blob=airdialogue_sampled_5k.jsonl
  --sft_model_prefix=models/sft-mistral-v6-1-epoch
  --base_model_id=mistralai/Mistral-7B-v0.1
  --base_model_prefix=models/mistral-base
  --output_prefix=models/d-rlaif-rfb-v1
  --max_samples=2
  --gemini_version=gemini-2.5-flash
  --beta=0.05
  --lr=1e-5
  --epochs=1
  --max_turns=5
  --gradient_accum=4
  --overwrite
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
import argparse
import time
from datetime import datetime, timezone
from typing import Optional

import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import storage
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from trl import create_reference_model

from simulator import run_conversation, format_flight_kb
from judge import score_transcript
from config import (
    CHECKPOINT_EVERY,
    CHECKPOINT_LOCAL,
    CHECKPOINT_BLOB,
    OUTPUT_LOCAL,
    OUTPUT_BLOB,
)
from utils import (
    get_secret,
    download_from_gcs,
    download_dir_from_gcs,
    upload_to_gcs,
    upload_dir_to_gcs,
    save_checkpoint,
    load_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLING_TEMPERATURE = 0.9   
KL_BETA_DEFAULT      = 0.05 
LR_DEFAULT           = 1e-5 
EPOCHS_DEFAULT       = 8 

def check_gpu_memory():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    inactive_reserved = reserved - allocated
    free_on_gpu = total_memory - reserved

    logger.info(f"--- GPU Memory Report ---")
    logger.info(f"Total:    {total_memory / 1e9:.2f} GB")
    logger.info(f"Allocated:{allocated / 1e9:.2f} GB (Real Tensors)")
    logger.info(f"Reserved: {reserved / 1e9:.2f} GB (PyTorch Pool)")
    logger.info(f"Fragmented: {inactive_reserved / 1e9:.2f} GB (Internal Cache)")
    logger.info(f"Untouched: {free_on_gpu / 1e9:.2f} GB (Available to Pool)")


#  VALUE HEAD
class ValueHead(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args: hidden_states: [batch, seq_len, hidden_size], from the last layer of policy network
        Returns: value: [batch]  — scalar estimated expected future reward per sequence
        """
        last_hidden = hidden_states[:, -1, :]  
        return self.dense(last_hidden).squeeze(-1) 



# MODEL LOADING
def load_policy_and_reference(
    sft_adapter_path: str,
    base_model_id: str,
) -> tuple:
    """
    Load the trainable policy and the frozen SFT reference model.

    Returns:
        policy_model    — PeftModel with trainable LoRA adapter
        ref_model       — PeftModel with all params frozen  (SFT reference)
        tokenizer       — shared AutoTokenizer
        hidden_size     — LM hidden dimension (for ValueHead)
    """
    logger.info(f"Loading base model: {base_model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # shared mistral base
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    hidden_size = base_model.config.hidden_size

    # policy (trainable LoRA adapter)
    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        adapter_name="reference",
        is_trainable=False,
    )

    # Loading the exact same adapter again, naming it "policy", to save memory
    model.load_adapter(
        sft_adapter_path,
        adapter_name="policy",
        is_trainable=True,
    )

    model.to(torch.bfloat16)

    # Only the policy adapter requires gradients
    for name, param in model.named_parameters():
        if "lora" in name and "policy" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    logger.info("Multi-adapter model loaded (policy & reference sharing base weights).")

    return model, tokenizer, hidden_size


# tokenize a cinversation transcript
def tokenise_transcript(
    transcript: list,
    kb_text: str,
    tokenizer,
    max_length: int = 4096,
) -> dict:
    """
    Re-build the Mistral [INST] prompt from the conversation transcript and
    tokenise the full sequence. Returns:
        input_ids       — [1, seq_len]
        attention_mask  — [1, seq_len]
        response_mask   — [1, seq_len]
        (policy gradient is applied only over generated tokens)
    """
    system_block = (
        "You are a professional flight booking agent. "
        "Help the customer find and book a flight that matches their requirements. "
        "You must ONLY reference flights listed in the database below — "
        "never invent or hallucinate flight details.\n\n"
        f"Available Flights:\n{kb_text}"
    )

    all_input_ids = []
    all_response_masks = []

    all_input_ids.append(tokenizer.bos_token_id)
    all_response_masks.append(False)

    for i, turn in enumerate(transcript):
        if turn["role"] == "customer":

            content = f"{system_block}\n\n{turn['text']}" if i == 0 else turn['text']
            text = f"[INST] {content} [/INST] "
            
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_input_ids.extend(ids)
            all_response_masks.extend([False] * len(ids))
            
        elif turn["role"] == "agent":

            text = f"{turn['text']}</s>"
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_input_ids.extend(ids)

            all_response_masks.extend([True] * len(ids))

    input_ids = torch.tensor([all_input_ids])
    response_mask = torch.tensor([all_response_masks])

    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]
        response_mask = response_mask[:, :max_length]

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.to("cuda"),
        "attention_mask": attention_mask.to("cuda"),
        "response_mask": response_mask.to("cuda"),
    }


#  direct reward from Gemini
def get_direct_reward(
    judge_gemini: GenerativeModel,
    transcript_row: dict,
) -> float:
    """
    Query Gemini Judge DIRECTLY during RL to get a reward.
    The judge returns a structured JSON with a score in [-1.0, +1.0].
    """
    judgment = score_transcript(judge_gemini, transcript_row)
    reward   = float(judgment.get("score", 0.0))
    outcome  = judgment.get("outcome", "unknown")
    justification = judgment.get("justification", "")
    return reward, outcome, justification


#  KL divergence - between policy and SFT reference
def compute_kl_divergence(
    policy_log_probs: torch.Tensor,   # [B, T-1, V]
    ref_log_probs:    torch.Tensor,   # [B, T-1, V]
    response_mask:    torch.Tensor,   # [B, T]
) -> torch.Tensor:
    """ KL averaged over response tokens. """
    mask = response_mask[:, 1:].float() 
 
    kl_per_token = (
        torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
    ).sum(dim=-1)
 
    return (kl_per_token * mask).sum() / (mask.sum() + 1e-8)


#  reinforce policy-gadient loss with value baseline
def compute_pg_loss(
    token_log_prob: torch.Tensor,   # [B, T-1] 
    response_mask:  torch.Tensor,   # [B, T]   
    advantage:      torch.Tensor,   # [B]      
) -> torch.Tensor:
    """REINFORCE PG loss"""
    mask         = response_mask[:, 1:].float()
    seq_log_prob = (token_log_prob * mask).sum(dim=1)
    return -(seq_log_prob * advantage.detach()).mean()

class _LastHiddenStateCapture:
    """
    Registers a forward hook on the final transformer layer norm so we
    get the last hidden state without enabling output_hidden_states=True.
    """
    def __init__(self, model: PeftModel):
        self._norm  = self._find_norm(model)
        self._hook  = None
        self.hidden = None
 
    @staticmethod
    def _find_norm(model):

        for attr in ("base_model", "model", "model"):
            if hasattr(model, attr):
                model = getattr(model, attr)
            else:
                break
 
        if hasattr(model, "norm"):
            return model.norm
 
        norm_module = None
        for name, module in model.named_modules():
            if name.endswith(".norm") or name == "norm":
                norm_module = module 
        if norm_module is not None:
            return norm_module
 
        raise AttributeError(
            "Could not locate the final layer norm in the model. "
        )
 
    def __enter__(self):
        def hook_fn(module, input, output):
            self.hidden = output
        self._hook = self._norm.register_forward_hook(hook_fn)
        return self
 
    def __exit__(self, *_):
        if self._hook is not None:
            self._hook.remove()

#  FULL d-RLAIF OBJECTIVE
def compute_drlaif_loss(
    model:    PeftModel,
    value_head:      ValueHead,
    tokenised:       dict,
    reward:          float,
    beta:            float,
) -> tuple:
    """
    Full d-RLAIF objective:
    Implementation:
      1. Forward pass through policy   → logits, hidden states
      2. Forward pass through reference  → reference logits  (no grad)
      3. Compute value baseline from policy hidden state
      4. Advantage calculation
      5. Policy gradient loss calculation
      6. KL divergence loss calculation
      7. Value (baseline) regression loss
      8. Combined loss calculation

    Returns:
        total_loss, pg_loss, kl_loss, value_loss, advantage
    """
    input_ids      = tokenised["input_ids"]
    attention_mask = tokenised["attention_mask"]
    response_mask  = tokenised["response_mask"]

    # Reference forward pass
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        ref_out = model(input_ids=input_ids, attention_mask=attention_mask)

        ref_log_probs = F.log_softmax(
            ref_out.logits[:, :-1, :].float(), dim=-1
        ).to(torch.bfloat16)  
        del ref_out
    torch.cuda.empty_cache()

    # Policy forward pass
    model.set_adapter("policy")
    model.train()
    with _LastHiddenStateCapture(model) as cap:
        policy_out = model(input_ids=input_ids, attention_mask=attention_mask)

        policy_logits_shifted = policy_out.logits[:, :-1, :]  
        del policy_out 
 
    hidden_states = cap.hidden

    reward_tensor = torch.tensor(reward, dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Shared log-softmax
    policy_log_probs = F.log_softmax(
        policy_logits_shifted.float(), dim=-1
    ).to(torch.bfloat16)
 
    targets = input_ids[:, 1:]
    token_log_prob = policy_log_probs.gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)
 
    del policy_logits_shifted
 
    # Value baseline
    value_estimate = value_head(hidden_states)
    del hidden_states
 
    advantage = reward_tensor - value_estimate.detach()
 
    # PG loss
    pg_loss = compute_pg_loss(
        token_log_prob=token_log_prob,
        response_mask=response_mask,
        advantage=advantage,
    )
 
    # KL loss
    kl_loss = compute_kl_divergence(
        policy_log_probs=policy_log_probs,
        ref_log_probs=ref_log_probs,
        response_mask=response_mask,
    )
    del policy_log_probs, ref_log_probs
 
    # Value loss
    value_loss = F.mse_loss(value_estimate.squeeze(), reward_tensor)
 
    # Combined objective
    total_loss = (1.0 - beta) * pg_loss + beta * kl_loss + value_loss
 
    return total_loss, pg_loss, kl_loss, value_loss, advantage.mean().item()



#  Training Loop
def train(args):
    logger.info("=" * 60)
    logger.info("Direct-RLAIF  (d-RLAIF) training starting")
    logger.info("=" * 60)

    logger.info("Fetching secrets ...")
    hf_token = get_secret(args.project, "hf-token")
    os.environ["HF_TOKEN"]                = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"]  = hf_token
    vertexai.init(project="rlaif-project", location="us-central1")


    simulator_gemini = GenerativeModel(
        args.gemini_version,
        generation_config=GenerationConfig(
            temperature=0.8,
            max_output_tokens=2048
        ),
    )

    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
        "thinking_config": {
            "include_thoughts": False,
            "thinking_budget": 1024
        }
    }

    judge_gemini = GenerativeModel(
        args.gemini_version,
        generation_config=generation_config,
    )
    logger.info("Gemini ready — simulator(T=0.8), judge(T=0.0)")

    # downloading SFT adapter + loading models
    local_adapter = "/tmp/sft_adapter"
    logger.info(f"Downloading SFT adapter from gs://{args.bucket}/{args.sft_model_prefix}")
    download_dir_from_gcs(args.bucket, args.sft_model_prefix, local_adapter)

    local_base_model = "./tmp/local_mistral"
    logger.info(f"Downloading Mistral from gs://{args.bucket}/{args.base_model_prefix}")
    download_dir_from_gcs(args.bucket, args.base_model_prefix, local_base_model)


    combined_model, tokenizer, hidden_size = load_policy_and_reference(
        sft_adapter_path=local_adapter,
        base_model_id=local_base_model,
    )

    # value head
    value_head = ValueHead(hidden_size).to("cuda:0").to(torch.bfloat16)
    logger.info(f"ValueHead initialised (hidden_size={hidden_size})")

    trainable_params = list(combined_model.parameters()) + list(value_head.parameters())
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, trainable_params),
        lr=args.lr,
        weight_decay=0.0,
    )

    local_data = "/tmp/train.jsonl"
    download_from_gcs(args.bucket, args.data_blob, local_data)

    rows = []
    with open(local_data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max_samples:
        rows = rows[:args.max_samples]
    logger.info(f"Loaded {len(rows)} training examples")

    if args.overwrite:
        logger.info("Overwrite flag set — removing existing GCS artefacts ...")
        gcs_client  = storage.Client(project=args.project)
        gcs_bucket  = gcs_client.bucket(args.bucket)
        for blob_path in [CHECKPOINT_BLOB, OUTPUT_BLOB]:
            b = gcs_bucket.blob(blob_path)
            if b.exists():
                b.delete()
                logger.info(f"  Deleted gs://{args.bucket}/{blob_path}")

    checkpoint = load_checkpoint(args.bucket)
    if checkpoint:
        processed_indices = set(checkpoint["processed_indices"])
        all_rewards       = checkpoint["scores"]
        start_epoch       = checkpoint.get("epoch", 0)
    else:
        processed_indices = set()
        all_rewards       = []
        start_epoch       = 0

    step              = 0
    accum_loss        = 0.0
    accum_pg          = 0.0
    accum_kl          = 0.0
    accum_vl          = 0.0
    optimizer.zero_grad()

    log_file = open(OUTPUT_LOCAL, "a" if checkpoint else "w")


    # EPOCH LOOP
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'═'*60}")
        logger.info(f"EPOCH {epoch+1}/{args.epochs}")
        logger.info(f"{'═'*60}")

        for idx, row in enumerate(rows):

            global_step_id = epoch * len(rows) + idx
            if global_step_id in processed_indices:
                continue

            intent = row.get("intent", {})
            kb     = row.get("kb", [])
            if not intent or not kb:
                logger.warning(f"Row {idx} — missing intent/KB, skipping")
                processed_indices.add(global_step_id)
                continue

            logger.info(f"\n[E{epoch+1} | {idx+1}/{len(rows)}]  Running rollout ...")

            # STEP 1: ROLLOUT
            # Generate a full conversation using the current policy.
            try:
                combined_model.set_adapter("policy")
                result = run_conversation(
                    row=row,
                    agent_model=combined_model,
                    agent_tokenizer=tokenizer,
                    gemini_model=simulator_gemini,
                    max_turns=args.max_turns,
                )
            except Exception as e:
                logger.error(f"  Rollout failed: {e}")
                processed_indices.add(global_step_id)
                continue

            transcript = result["transcript"]
            kb_text    = format_flight_kb(kb)

            # STEP 2: DIRECT REWARD FROM GEMINI
            transcript_row = {
                "intent":     result["intent"],
                "kb":         result["kb"],
                "transcript": transcript,
                "outcome":    result["outcome"],
            }
            try:
                reward, outcome, justification = get_direct_reward(
                    judge_gemini, transcript_row
                )
            except Exception as e:
                logger.error(f"  Judge failed: {e} — assigning 0.0 reward")
                reward, outcome, justification = 0.0, "judge_error", str(e)

            logger.info(
                f"  reward={reward:+.2f}  outcome={outcome}  "
                f"turns={result['num_turns']}"
            )
            logger.info(f"  justification: {justification[:100]}")

            all_rewards.append(reward)

            # STEP 3: TOKENISE TRANSCRIPT
            try:
                tokenised = tokenise_transcript(transcript, kb_text, tokenizer)
            except Exception as e:
                logger.error(f"  Tokenisation failed: {e}")
                processed_indices.add(global_step_id)
                continue

            if not tokenised["response_mask"][0].any():
                logger.warning("  No response tokens found — skipping gradient step")
                processed_indices.add(global_step_id)
                continue
            
            check_gpu_memory()
            
            # STEP 4: COMPUTE LOSS
            try:
                total_loss, pg_loss, kl_loss, value_loss, advantage = compute_drlaif_loss(
                    model=combined_model,
                    value_head=value_head,
                    tokenised=tokenised,
                    reward=reward,
                    beta=args.beta,
                )
            except Exception as e:
                logger.error(f"  Loss computation failed: {e}")
                processed_indices.add(global_step_id)
                del tokenised
                torch.cuda.empty_cache()
                continue

            # STEP 5: GRADIENT ACCUMULATION + OPTIMISER STEP
            scaled_loss = total_loss / args.gradient_accum
            scaled_loss.backward()

            accum_loss += total_loss.item()
            accum_pg   += pg_loss.item()
            accum_kl   += kl_loss.item()
            accum_vl   += value_loss.item()

            step += 1

            if step % args.gradient_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, trainable_params),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                n = args.gradient_accum
                logger.info(
                    f"  [step {step}] "
                    f"loss={accum_loss/n:.4f}  "
                    f"pg={accum_pg/n:.4f}  "
                    f"kl={accum_kl/n:.4f}  "
                    f"vl={accum_vl/n:.4f}  "
                    f"adv={advantage:+.4f}"
                )
                accum_loss = accum_pg = accum_kl = accum_vl = 0.0

            # STEP 6: LOG TRAINING RECORD
            record = {
                "epoch":         epoch,
                "idx":           idx,
                "step":          step,
                "reward":        reward,
                "outcome":       outcome,
                "advantage":     advantage,
                "total_loss":    total_loss.item(),
                "pg_loss":       pg_loss.item(),
                "kl_loss":       kl_loss.item(),
                "value_loss":    value_loss.item(),
                "num_turns":     result["num_turns"],
                "justification": justification,
                "timestamp":     datetime.now(timezone.utc).isoformat(),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            del scaled_loss, total_loss, pg_loss, kl_loss, value_loss, tokenised
            torch.cuda.empty_cache()

            processed_indices.add(global_step_id)

            # STEP 7: PERIODIC CHECKPOINT
            if len(processed_indices) % args.chk_freq == 0:
                _save_training_checkpoint(
                    args, combined_model, value_head,
                    processed_indices, all_rewards, epoch,
                )
                logger.info(f"Checkpoint saved at global step {global_step_id}  ")

            # rate-limit buffer between Gemini calls
            time.sleep(1.0)

        if step % args.gradient_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, trainable_params),
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        logger.info(f"Epoch {epoch+1} complete.")

    # SAVE FINAL MODEL
    log_file.close()
    _save_final_model(args, combined_model, value_head)

    # summary
    if all_rewards:
        n        = len(all_rewards)
        avg      = sum(all_rewards) / n
        perfect  = sum(1 for r in all_rewards if r > +0.6)
        partial  = sum(1 for r in all_rewards if r > +0.2 and r <= +0.6)
        neutral  = sum(1 for r in all_rewards if r > -0.2 and r <= +0.2)
        wrong    = sum(1 for r in all_rewards if r > -0.6 and r <= -0.2)
        critical = sum(1 for r in all_rewards if r >= -1.0 and r <= -0.6 )

        logger.info("\n" + "=" * 60)
        logger.info("d-RLAIF TRAINING COMPLETE")
        logger.info(f"  Total rollouts      : {n}")
        logger.info(f"  Average reward      : {avg:+.3f}")
        logger.info(f"  (1.0 to 0.6) perfect        : {perfect}  ({100*perfect/n:.1f}%)")
        logger.info(f"  (0.6 to 0.2) partial        : {partial}  ({100*partial/n:.1f}%)")
        logger.info(f"  (0.2 to -0.2) neutral        : {neutral}  ({100*neutral/n:.1f}%)")
        logger.info(f"  (-0.2 to -0.6) wrong booking  : {wrong}    ({100*wrong/n:.1f}%)")
        logger.info(f"  (-0.6 to -1.0) critical fail  : {critical} ({100*critical/n:.1f}%)")
        logger.info(f"  Model saved         : gs://{args.bucket}/{args.output_prefix}")
        logger.info("=" * 60)

        try:
            plt.figure(figsize=(10, 6))
            
            window = max(1, n // 5)
            rolling_avg = [sum(all_rewards[max(0, i-window):i+1]) / len(all_rewards[max(0, i-window):i+1]) for i in range(n)]
            
            plt.plot(all_rewards, alpha=0.3, color='blue', label='Individual Reward')
            plt.plot(rolling_avg, color='red', linewidth=2, label=f'Rolling Avg (w={window})')
            
            plt.title(f"d-RLAIF Training Rewards (Avg: {avg:+.3f})")
            plt.xlabel("Rollout Number")
            plt.ylabel("Reward Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            client = storage.Client()
            bucket = client.get_bucket(args.bucket)
            plot_blob = bucket.blob(f"{args.output_prefix}/reward_plot.png")
            plot_blob.upload_from_file(buf, content_type='image/png')
            
            logger.info(f"  Reward plot saved   : gs://{args.bucket}/{args.output_prefix}/reward_plot.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate or upload plot: {e}")



def _save_training_checkpoint(
    args,
    model,
    value_head: ValueHead,
    processed_indices: set,
    all_rewards: list,
    epoch: int,
) -> None:
    """Save LoRA adapter + value head + training state to GCS."""
    local_ckpt_dir = "/tmp/drlaif_checkpoint"
    os.makedirs(local_ckpt_dir, exist_ok=True)

    model.save_pretrained(local_ckpt_dir, selected_adapters=["policy"])

    torch.save(
        value_head.state_dict(),
        os.path.join(local_ckpt_dir, "value_head.pt"),
    )

    ckpt_prefix = f"{args.output_prefix}/checkpoint"
    upload_dir_to_gcs(args.bucket, local_ckpt_dir, ckpt_prefix)

    save_checkpoint(
        args.bucket,
        list(processed_indices),
        all_rewards,
        len(processed_indices),
    )
    logger.info(f"Checkpoint saved → gs://{args.bucket}/{ckpt_prefix}")


def _save_final_model(args, model, value_head: ValueHead) -> None:
    """Save final trained LoRA adapter + value head to GCS."""
    local_final = "/tmp/drlaif_final"
    os.makedirs(local_final, exist_ok=True)

    model.save_pretrained(local_final, selected_adapters=["policy"])
    torch.save(
        value_head.state_dict(),
        os.path.join(local_final, "value_head.pt"),
    )

    upload_dir_to_gcs(args.bucket, local_final, args.output_prefix)
    upload_to_gcs(args.bucket, OUTPUT_LOCAL, OUTPUT_BLOB)

    logger.info(f"Final model saved → gs://{args.bucket}/{args.output_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct-RLAIF (d-RLAIF) training with REINFORCE + value baseline"
    )

    # infrastructure
    parser.add_argument("--project",          required=True,
                        help="GCP project ID")
    parser.add_argument("--bucket",           required=True,
                        help="GCS bucket name (no gs:// prefix)")
    parser.add_argument("--data_blob",        required=True,
                        help="Path inside bucket to train.jsonl")
    parser.add_argument("--sft_model_prefix", default="models/sft-mistral-baseline",
                        help="GCS prefix of the SFT LoRA adapter to initialise from")
    parser.add_argument("--base_model_id",    default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace base model ID")
    parser.add_argument("--base_model_prefix",    required=True,
                        help="HuggingFace base model downloaded")
    parser.add_argument("--output_prefix",    default="models/drlaif-mistral",
                        help="GCS prefix to save the trained d-RLAIF adapter")
    parser.add_argument("--gemini_version",   default="gemini-2.5-flash",
                        help="Vertex AI Gemini model version")
    parser.add_argument("--chk_freq",   type=int, default=10,
                        help="checkpoint saving freq")

    # d-RLAIF hyperparameters
    parser.add_argument("--beta",             type=float, default=KL_BETA_DEFAULT,
                        help="KL divergence penalty coefficient")
    parser.add_argument("--lr",               type=float, default=LR_DEFAULT,
                        help="AdamW learning rate")
    parser.add_argument("--epochs",           type=int,   default=EPOCHS_DEFAULT,
                        help="Training epochs")
    parser.add_argument("--gradient_accum",   type=int,   default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_turns",        type=int,   default=10,
                        help="Max conversation turns per rollout")

    parser.add_argument("--max_samples",      type=int,   default=None,
                        help="Maximum examples — use 100 for a debug run")
    parser.add_argument("--overwrite",        action="store_true",
                        help="Delete existing GCS output before starting")

    args = parser.parse_args()
    train(args)

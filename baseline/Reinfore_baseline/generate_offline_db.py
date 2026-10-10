"""
generate_offline_db.py
======================
Generates an offline RL database of N conversations (default 1000) using
the initial SFT Mistral policy. Each record stores:

  - intent          : the original AirDialogue intent dict
  - kb              : the flight knowledge base
  - transcript      : the full conversation (list of {role, text} dicts)
  - outcome         : "confirmed" | "failed" | "max_turns_reached"
  - num_turns       : number of agent turns
  - reward          : scalar in [-1.0, +1.0] from Gemini Judge
  - outcome_judge   : judge's outcome label
  - justification   : one-line judge justification
  - tokenized       : {
        input_ids        : list[int]   (CPU, not cuda tensor — JSON-serializable)
        attention_mask   : list[int]
        response_mask    : list[bool]
    }

Output:
  Local : /tmp/offline_rl_db.jsonl
  GCS   : gs://<bucket>/<output_blob>

Usage:
python generate_offline_db.py \
    --project        274690076649 \
    --bucket         rlaif-data-bucket-rlaif-project \
    --data_blob      airdialogue_sampled_5k.jsonl \
    --sft_model_prefix  models/sft-mistral-v6-1-epoch \
    --base_model_prefix models/mistral-base \
    --base_model_id  mistralai/Mistral-7B-v0.1 \
    --output_blob    offline_rl/offline_rl_db.jsonl \
    --gemini_version gemini-2.5-flash \
    --n_conversations 1000 \
    --max_turns 10
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
import argparse
import time
from datetime import datetime, timezone

import torch
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import storage

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from simulator import run_conversation, format_flight_kb
from judge import score_transcript
from config import judge_retry_fallback_response
from utils import (
    get_secret,
    download_from_gcs,
    download_dir_from_gcs,
    upload_to_gcs,
    parse_gcs_uri,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_LOCAL  = "/tmp/offline_rl_db.jsonl"
CHECKPOINT_LOCAL = "/tmp/offline_db_checkpoint.json"
MAX_SEQ_LEN   = 4096



#  MODEL LOADING (SFT policy only)
def load_sft_policy(sft_adapter_path: str, base_model_id: str):
    """Load the SFT LoRA adapter on top of a 4-bit quantized Mistral base."""
    logger.info(f"Loading base model from: {base_model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        is_trainable=False,
    )
    model.eval()

    logger.info("SFT policy loaded successfully.")
    return model, tokenizer


# TOKENISATION
def tokenise_transcript(
    transcript: list,
    kb_text: str,
    tokenizer,
    max_length: int = MAX_SEQ_LEN,
) -> dict:
    """
    Returns dictionary with keys:
        input_ids
        attention_mask
        response_mask
    """
    system_block = (
        "You are a professional flight booking agent. "
        "Help the customer find and book a flight that matches their requirements. "
        "You must ONLY reference flights listed in the database below — "
        "never invent or hallucinate flight details.\n\n"
        f"Available Flights:\n{kb_text}"
    )

    all_input_ids: list[int] = []
    all_response_masks: list[bool] = []

    all_input_ids.append(tokenizer.bos_token_id)
    all_response_masks.append(False)

    for i, turn in enumerate(transcript):
        if turn["role"] == "customer":
            content = f"{system_block}\n\n{turn['text']}" if i == 0 else turn["text"]
            text = f"[INST] {content} [/INST] "
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_input_ids.extend(ids)
            all_response_masks.extend([False] * len(ids))

        elif turn["role"] == "agent":
            text = f"{turn['text']}</s>"
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_input_ids.extend(ids)
            all_response_masks.extend([True] * len(ids))

    if len(all_input_ids) > max_length:
        all_input_ids      = all_input_ids[:max_length]
        all_response_masks = all_response_masks[:max_length]

    attention_mask = [1] * len(all_input_ids)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": attention_mask,
        "response_mask":  all_response_masks,
    }


def judge_transcript(judge_gemini: GenerativeModel, transcript_row: dict) -> tuple:
    for attempt in range(2):
        try:
            judgment = score_transcript(judge_gemini, transcript_row)
            reward        = float(judgment.get("score", 0.0))
            outcome       = judgment.get("outcome", "unknown")
            justification = judgment.get("justification", "")
            return reward, outcome, justification
        except Exception as e:
            if attempt == 0:
                logger.warning(f"Judge failed (attempt 1): {e} — retrying in 5s...")
                time.sleep(5)
            else:
                logger.error(f"Judge failed (attempt 2): {e} — using fallback.")

    fb = judge_retry_fallback_response
    return float(fb["score"]), fb["outcome"], fb["justification"]


def save_progress_checkpoint(
    bucket: str,
    output_blob: str,
    n_done: int,
    n_confirmed: int,
    n_failed: int,
    all_rewards: list,
) -> None:
    """Save a JSON checkpoint so generation can resume after crash."""
    checkpoint = {
        "n_done":      n_done,
        "n_confirmed": n_confirmed,
        "n_failed":    n_failed,
        "all_rewards": all_rewards,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_LOCAL, "w") as f:
        json.dump(checkpoint, f)

    ckpt_blob = output_blob.replace(".jsonl", "_checkpoint.json")
    try:
        upload_to_gcs(bucket, CHECKPOINT_LOCAL, ckpt_blob)
        logger.info(f"Checkpoint saved → gs://{bucket}/{ckpt_blob}")
    except Exception as e:
        logger.warning(f"Checkpoint GCS upload failed: {e}")


def load_progress_checkpoint(bucket: str, output_blob: str) -> dict | None:
    """Download and return checkpoint dict if it exists"""
    ckpt_blob = output_blob.replace(".jsonl", "_checkpoint.json")
    try:
        download_from_gcs(bucket, ckpt_blob, CHECKPOINT_LOCAL)
        with open(CHECKPOINT_LOCAL) as f:
            ckpt = json.load(f)
        logger.info(
            f"Resuming from checkpoint: {ckpt['n_done']} conversations already done."
        )
        return ckpt
    except Exception:
        return None


# MAIN GENERATION LOOP
def generate(args):
    logger.info("=" * 60)
    logger.info("Offline RL Database Generation")
    logger.info(f"  Target conversations : {args.n_conversations}")
    logger.info(f"  Max turns per convo  : {args.max_turns}")
    logger.info("=" * 60)

    hf_token = get_secret(args.project, "hf-token")
    os.environ["HF_TOKEN"]               = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    vertexai.init(project="rlaif-project", location="us-central1")

    simulator_gemini = GenerativeModel(
        args.gemini_version,
        generation_config=GenerationConfig(
            temperature=0.8,
            max_output_tokens=2048,
        ),
    )

    judge_gemini = GenerativeModel(
        args.gemini_version,
        generation_config=GenerationConfig(
            temperature=0.0,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    logger.info("Gemini ready — simulator(T=0.8), judge(T=0.0)")


    local_adapter    = "/tmp/sft_adapter"
    local_base_model = "/tmp/local_mistral"

    logger.info(f"Downloading SFT adapter from gs://{args.bucket}/{args.sft_model_prefix}")
    download_dir_from_gcs(args.bucket, args.sft_model_prefix, local_adapter)

    logger.info(f"Downloading base model from gs://{args.bucket}/{args.base_model_prefix}")
    download_dir_from_gcs(args.bucket, args.base_model_prefix, local_base_model)

    model, tokenizer = load_sft_policy(
        sft_adapter_path=local_adapter,
        base_model_id=local_base_model,
    )

    # load training source data
    local_data = "/tmp/source_data.jsonl"
    download_from_gcs(args.bucket, args.data_blob, local_data)

    rows = []
    with open(local_data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    logger.info(f"Total source rows loaded: {len(rows)}")

    end = args.end_idx if args.end_idx is not None else args.start_idx + args.n_conversations * 2
    slice_rows = rows[args.start_idx : end]

    import random
    random.seed(42)
    random.shuffle(slice_rows)
    source_rows = slice_rows

    logger.info(
        f"Using source rows [{args.start_idx} : {end}]  "
        f"({len(source_rows)} rows after slicing, before any skip filtering)"
    )

    # resume from checkpoint if available
    ckpt = load_progress_checkpoint(args.bucket, args.output_blob)
    if ckpt:
        n_done      = ckpt["n_done"]
        n_confirmed = ckpt["n_confirmed"]
        n_failed    = ckpt["n_failed"]
        all_rewards = ckpt["all_rewards"]

        log_file = open(OUTPUT_LOCAL, "a")
        logger.info(f"Resuming: {n_done} conversations already complete.")
    else:
        n_done = n_confirmed = n_failed = 0
        all_rewards = []
        log_file = open(OUTPUT_LOCAL, "w")

    # main generation loop
    row_idx = n_done

    while n_done < args.n_conversations and row_idx < len(source_rows):
        row = source_rows[row_idx]
        row_idx += 1

        intent = row.get("intent", {})
        kb     = row.get("kb", [])

        if not intent or not kb:
            logger.warning(f"Row {row_idx} — missing intent/KB, skipping.")
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(
            f"Generating conversation {n_done + 1}/{args.n_conversations}  "
            f"(source row {row_idx})"
        )
        logger.info(f"{'─'*60}")

        # Run conversation
        try:
            result = run_conversation(
                row=row,
                agent_model=model,
                agent_tokenizer=tokenizer,
                gemini_model=simulator_gemini,
                max_turns=args.max_turns,
            )
        except Exception as e:
            logger.error(f"  Conversation failed: {e} — skipping.")
            time.sleep(2)
            continue

        transcript = result["transcript"]
        kb_text    = format_flight_kb(kb)

        # Judge transcript for reward
        transcript_row = {
            "intent":     result["intent"],
            "kb":         result["kb"],
            "transcript": transcript,
            "outcome":    result["outcome"],
        }

        reward, judge_outcome, justification = judge_transcript(
            judge_gemini, transcript_row
        )

        logger.info(
            f"  reward={reward:+.2f}  outcome={result['outcome']}  "
            f"judge={judge_outcome}  turns={result['num_turns']}"
        )

        try:
            tokenized = tokenise_transcript(transcript, kb_text, tokenizer)
        except Exception as e:
            logger.error(f"  Tokenisation failed: {e} — skipping.")
            continue

        has_response_tokens = any(tokenized["response_mask"])
        if not has_response_tokens:
            logger.warning("  No response tokens found after tokenisation — skipping.")
            continue

        # Write database record
        record = {

            "intent":        intent,
            "kb":            kb,
            "transcript":    transcript,
            "outcome":       result["outcome"],
            "num_turns":     result["num_turns"],
            "reward":        reward,
            "outcome_judge": judge_outcome,
            "justification": justification,

            "tokenized": {
                "input_ids":      tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "response_mask":  tokenized["response_mask"],
            },

            "seq_len":    len(tokenized["input_ids"]),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "db_index":   n_done,
        }

        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        all_rewards.append(reward)
        n_done += 1
        if result["outcome"] == "confirmed":
            n_confirmed += 1
        elif result["outcome"] == "failed":
            n_failed += 1

        if n_done % args.save_freq == 0:
            log_file.close()
            upload_to_gcs(args.bucket, OUTPUT_LOCAL, args.output_blob)
            save_progress_checkpoint(
                args.bucket, args.output_blob, n_done,
                n_confirmed, n_failed, all_rewards,
            )
            log_file = open(OUTPUT_LOCAL, "a")
            logger.info(f"Progress save: {n_done}/{args.n_conversations} complete.")


        time.sleep(1.0)

    log_file.close()

    # GCS upload
    upload_to_gcs(args.bucket, OUTPUT_LOCAL, args.output_blob)
    logger.info(f"Database saved → gs://{args.bucket}/{args.output_blob}")

    # summary
    incomplete = n_done - n_confirmed - n_failed
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    avg_seq_len_approx = "see jsonl"

    logger.info("\n" + "=" * 60)
    logger.info("DATABASE GENERATION COMPLETE")
    logger.info(f"  Total conversations  : {n_done}")
    logger.info(f"  Booking confirmed    : {n_confirmed} ({100*n_confirmed/max(n_done,1):.1f}%)")
    logger.info(f"  Booking failed       : {n_failed}    ({100*n_failed/max(n_done,1):.1f}%)")
    logger.info(f"  Incomplete           : {incomplete}")
    logger.info(f"  Average reward       : {avg_reward:+.3f}")
    logger.info(f"  Output (GCS)         : gs://{args.bucket}/{args.output_blob}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate offline RL database of N Mistral-SFT conversations "
                    "with Gemini judge rewards"
    )

    parser.add_argument("--project",            required=True,
                        help="GCP project ID")
    parser.add_argument("--bucket",             required=True,
                        help="GCS bucket name")
    parser.add_argument("--data_blob",          required=True,
                        help="GCS path to source train.jsonl inside bucket")
    parser.add_argument("--sft_model_prefix",   required=True,
                        help="GCS prefix of the SFT LoRA adapter to use as initial policy")
    parser.add_argument("--base_model_prefix",  required=True,
                        help="GCS prefix of the downloaded Mistral base model")
    parser.add_argument("--base_model_id",      default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace base model ID")
    parser.add_argument("--output_blob",        required=True,
                        help="GCS path for the output database .jsonl")
    parser.add_argument("--start_idx", type=int, default=0,
                    help="Start index into source data rows")
    parser.add_argument("--end_idx",   type=int, default=None,
                    help="End index into source data rows "
                         "Defaults to start_idx + n_conversations.")
    parser.add_argument("--save_freq",   type=int, default=10,
                    help="checkpoint saving")

    parser.add_argument("--gemini_version",     default="gemini-2.5-flash",
                        help="Vertex AI Gemini model version for simulator + judge")
    parser.add_argument("--n_conversations",    type=int, default=1000,
                        help="Number of conversations to generate")
    parser.add_argument("--max_turns",          type=int, default=10,
                        help="Maximum turns per conversation")

    args = parser.parse_args()
    generate(args)

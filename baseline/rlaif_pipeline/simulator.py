"""
Simulates a customer interacting with the fine-tuned Mistral-7B flight booking agent.
Uses Gemini API to drive the customer side of the conversation based on hidden intent.
"""
import os
import json
import argparse
import logging
import time
from constitution import SIMULATOR_CONSTITUTION
from config import intent_field_map
from utils import download_from_gcs, upload_to_gcs, parse_gcs_uri
 
import torch
from google.cloud import storage
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
 
vertexai.init(project="rlaif-project", location="us-central1")
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
 
 
def format_intent(intent: dict) -> str:
    if not intent or not isinstance(intent, dict):
        return "No specific constraints — find a reasonable flight."
    lines = []
    for field, label in intent_field_map.items():
        val = intent.get(field)
        if val is not None and val != "" and val != "none":
            lines.append(f"  - {label}: {val}")
    for k, v in intent.items():
        if k not in intent_field_map and v is not None:
            lines.append(f"  - {k.replace('_', ' ').title()}: {v}")
    return "\n".join(lines) if lines else "No specific constraints."
 
 
def format_flight_kb(kb: list) -> str:
    if not kb:
        return "No flights available."
    lines = []
    for i, flight in enumerate(kb, 1):
        if not isinstance(flight, dict):
            continue
        parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in flight.items()]
        lines.append(f"Flight {i}: " + " | ".join(parts))
    return "\n".join(lines) if lines else "No flights available."
 
 
def load_agent(model_path: str, base_model_id: str = "mistralai/Mistral-7B-v0.1"):
    logger.info(f"Loading agent from {model_path} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    logger.info("Agent loaded successfully.")
    return model, tokenizer
 

def _hard_cutoff(text: str, max_sentences: int = 3) -> str:
    """
    Cut the agent response to at most max_sentences sentences.
    This prevents the model from rambling on for paragraphs.
    """
    # Cut at hallucination signals first
    for marker in ["[INST]", "Customer:", "CUSTOMER:", "[/INST]"]:
        if marker in text:
            text = text[:text.index(marker)]
 
    # Cut at double newline (paragraph boundary)
    if "\n\n" in text:
        text = text[:text.index("\n\n")]
 
    # Cut to max_sentences using ./question/exclamation as boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) > max_sentences:
        text = " ".join(sentences[:max_sentences])
 
    return text.strip()
 
# shorter (128) = more focused replies!! Agent was rambling for higher setting!!
def get_agent_response(model, tokenizer, conversation_history: list, kb_text: str, max_new_tokens: int = 128) -> tuple:
    """
    Generate agent response and return (text, logprobs, token_ids).
    """
    system_block = (
        "You are a professional flight booking agent. "
        "Help the customer find and book a flight. "
        "Give SHORT, focused replies — 1 to 3 sentences only. "
        "ONLY reference flights listed in the database below. "
        "Never invent flight details.\n\n"
        f"Available Flights:\n{kb_text}"
    )
 
    prompt = ""
    for i, turn in enumerate(conversation_history):
        if turn["role"] == "customer":
            if i == 0:
                prompt += f"<s>[INST] {system_block}\n\n{turn['text']} [/INST] "
            else:
                prompt += f"[INST] {turn['text']} [/INST] "
        elif turn["role"] == "agent":
            prompt += f"{turn['text']} </s>"
 
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
 
    # temperature=0.0 + do_sample=False = greedy decoding
    # This stops the hallucination/repetition loops seen in logs.
    # The model picks the single most likely token at each step.
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
 
    new_token_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    raw_text      = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    response      = _hard_cutoff(raw_text)
 
    # Guard: if model generates empty response (EOS immediately, common in well-trained models),
    # send a safe fallback so Gemini doesn't crash with "value must not be empty"
    if not response.strip():
        logger.warning("[AGENT] Empty response after _hard_cutoff — using fallback phrase")
        response = "I'm checking the available flights for you. Could you confirm your travel dates?"
 
    # Compute log probs for the tokens that made it into the final response
    scores        = torch.stack(outputs.scores, dim=1)
    log_probs_all = torch.log_softmax(scores, dim=-1)
    token_logprobs = log_probs_all[ 0, torch.arange(len(new_token_ids)), s]
 
    # Only keep logprobs for tokens that survived the cutoff
    clean_ids = tokenizer(
        response, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    n = min(len(clean_ids), len(new_token_ids))
 
    token_logprobs = token_logprobs[:n].detach().cpu()
    token_ids_cpu  = new_token_ids[:n].detach().cpu()
 
    return response, token_logprobs, token_ids_cpu
 
 
def get_simulator_response(gemini_model, chat_session, agent_message: str) -> str:
    try:
        response = chat_session.send_message(agent_message)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini API error: {e}. Retrying in 5s...")
        time.sleep(5)
        response = chat_session.send_message(agent_message)
        return response.text.strip()
 
 
def is_terminal(simulator_message: str) -> tuple:
    if "BOOKING_CONFIRMED" in simulator_message:
        return True, "confirmed"
    if "BOOKING_FAILED" in simulator_message:
        return True, "failed"
    return False, ""
 
 
def run_conversation(row: dict, agent_model, agent_tokenizer, gemini_model, max_turns: int) -> dict:
    intent      = row.get("intent", {})
    kb          = row.get("kb", [])
    kb_text     = format_flight_kb(kb)
    intent_text = format_intent(intent)
 
    system_prompt = SIMULATOR_CONSTITUTION.format(
        intent_block=intent_text,
        max_turns=max_turns,
    )
 
    chat_session = gemini_model.start_chat(history=[], response_validation=False)
    chat_session.send_message(
        f"[SYSTEM SETUP - acknowledge with 'Ready']\n{system_prompt}"
    )
 
    conversation_history = []
    outcome      = "incomplete"
    all_logprobs = []
    all_token_ids = []
 
    opening = chat_session.send_message(
        "Begin the conversation. Send your opening message to the booking agent."
    ).text.strip()
 
    conversation_history.append({"role": "customer", "text": opening})
    logger.info(f"  Customer: {opening}")
 
    for turn in range(max_turns):
        # get_agent_response
        agent_response, token_logprobs, token_ids = get_agent_response(
            agent_model, agent_tokenizer, conversation_history, kb_text
        )
        all_logprobs.append(token_logprobs)
        all_token_ids.append(token_ids)
 
        conversation_history.append({"role": "agent", "text": agent_response})
        logger.info(f"  Agent:    {agent_response}")
 
        sim_response = get_simulator_response(gemini_model, chat_session, agent_response)
        terminal, outcome = is_terminal(sim_response)
 
        clean_response = (
            sim_response
            .replace("BOOKING_CONFIRMED", "")
            .replace("BOOKING_FAILED", "")
            .strip()
        )
        conversation_history.append({"role": "customer", "text": clean_response})
        logger.info(f"  Customer: {clean_response}")
 
        if terminal:
            logger.info(f"  Conversation ended: {outcome}")
            break
    else:
        outcome = "max_turns_reached"
        logger.info("  Max turns reached without resolution.")
 
    return {
        "intent": intent,
        "kb": kb,
        "transcript": conversation_history,
        "outcome":    outcome,
        "num_turns":  len([t for t in conversation_history if t["role"] == "agent"]),
        "logprobs":   all_logprobs,
        "token_ids":  all_token_ids,
    }
 
 
def main(args):
    # Trigger point!!!
    gemini_model = GenerativeModel(
        args.gemini_version,
        generation_config=GenerationConfig(
            temperature=0.8,
            max_output_tokens=512,
        ),
    )
    logger.info("Gemini simulator initialized.")
 
    agent_bucket, agent_prefix = parse_gcs_uri(args.agent_model)
    local_model_dir = "/tmp/agent_model"
    os.makedirs(local_model_dir, exist_ok=True)
 
    client = storage.Client()
    blobs  = client.bucket(agent_bucket).list_blobs(prefix=agent_prefix)
    for blob in blobs:
        local_path = os.path.join(local_model_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
    logger.info(f"Agent adapter files downloaded to {local_model_dir}")
 
    agent_model, agent_tokenizer = load_agent(
        model_path=local_model_dir,
        base_model_id=args.base_model_id,
    )
 
    data_bucket, data_blob = parse_gcs_uri(args.test_data)
    local_data = "/tmp/test.jsonl"
    download_from_gcs(data_bucket, data_blob, local_data)
 
    rows = []
    with open(local_data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
 
    if args.max_samples:
        rows = rows[:args.max_samples]
 
    logger.info(f"Running simulator on {len(rows)} test examples ...")
 
    transcripts = []
    for i, row in enumerate(rows):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Conversation {i+1}/{len(rows)}")
        logger.info(f"{'─'*60}")
        try:
            result = run_conversation(
                row=row,
                agent_model=agent_model,
                agent_tokenizer=agent_tokenizer,
                gemini_model=gemini_model,
                max_turns=args.max_turns,
            )
            transcripts.append(result)
        except Exception as e:
            logger.error(f"Conversation {i+1} failed: {e}")
            continue
        time.sleep(1)
 
    local_out = "/tmp/transcripts.jsonl"
    with open(local_out, "w") as f:
        for t in transcripts:
            record = {k: v for k, v in t.items() if k not in ("logprobs", "token_ids")}
            f.write(json.dumps(record) + "\n")
 
    out_bucket, out_blob = parse_gcs_uri(args.output)
    upload_to_gcs(out_bucket, local_out, out_blob)
 
    outcomes   = [t["outcome"] for t in transcripts]
    confirmed  = outcomes.count("confirmed")
    failed     = outcomes.count("failed")
    incomplete = len(outcomes) - confirmed - failed
 
    logger.info(f"\n{'═'*60}")
    logger.info(f"Simulator complete.")
    logger.info(f"  Total conversations : {len(transcripts)}")
    if transcripts:
        logger.info(f"  Booking confirmed : {confirmed} ({confirmed/len(transcripts)*100:.1f}%)")
        logger.info(f"  Booking failed : {failed} ({failed/len(transcripts)*100:.1f}%)")
        logger.info(f"  Incomplete : {incomplete}")
    logger.info(f"  Transcripts saved   : gs://{out_bucket}/{out_blob}")
    logger.info(f"{'═'*60}")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirDialogue User Simulator")
    parser.add_argument("--agent_model",    required=True)
    parser.add_argument("--base_model_id",  default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--test_data",      required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gemini_key", required=True)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gemini_version", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()
    main(args)
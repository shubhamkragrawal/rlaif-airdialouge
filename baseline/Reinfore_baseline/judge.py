# While conversation, Agent should not provide any fabricated information, everything should be referenced truely from the database

"""Reads completed conversation transcripts produced by simulator.py,
scores each one against the original intent using Gemini API,
and outputs a reward signal.
"""

import os
import json
import argparse
import logging
import time
from typing import Optional
from constitution import JUDGE_CONSTITUTION
from config import intent_field_map, judge_retry_fallback_response
from utils import (
    download_from_gcs,
    upload_to_gcs,
    parse_gcs_uri
    )
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

vertexai.init(project="rlaif-project", location="us-central1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# formating our intent block for the judge same as the user simulator's intent
def format_intent(intent: dict) -> str:
    if not intent or not isinstance(intent, dict):
        return "No specific constraints provided."
    lines = []

    for field, label in intent_field_map.items():
        val = intent.get(field)
        if val is not None and val != "" and val != "none":
            lines.append(f"  {label}: {val}")
    for k, v in intent.items():
        if k not in intent_field_map and v is not None:
            lines.append(f"  {k.replace('_',' ').title()}: {v}")
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


def format_transcript(transcript: list) -> str:
    if not transcript:
        return "No conversation recorded."
    lines = []
    for turn in transcript:
        role = turn.get("role", "unknown").upper()
        text = turn.get("text", "").strip()
        lines.append(f"{role}: {text}")
    return "\n".join(lines)



# builidng judge's prompt which decides it's behaviour
def build_judge_prompt(transcript_row: dict) -> str:

    intent_text    = format_intent(transcript_row.get("intent", {}))
    kb_text        = format_flight_kb(transcript_row.get("kb", []))
    transcript_text = format_transcript(transcript_row.get("transcript", []))

    return f"""
            {JUDGE_CONSTITUTION}


            --- CUSTOMER'S HIDDEN INTENT ---
            {intent_text}

            --- AVAILABLE FLIGHT DATABASE ---
            {kb_text}

            --- CONVERSATION TRANSCRIPT ---
            {transcript_text}

            ==========================================
            Now evaluate the agent's performance and output your JSON score.
            ==========================================
            """

def _extract_json(raw: str) -> str:
    """
    Robustly extract a JSON object from a Gemini response string.
    Handles markdown fences, leading text, trailing text, and newlines.
    """
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    start = raw.find("{")
    end   = raw.rfind("}")

    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)

    return raw[start:end+1]

def score_transcript(gemini_model, transcript_row: dict, max_retries: int = 3,) -> dict:
    """
    Returns the full judgment dict with score and reasoning.
    """
    prompt = build_judge_prompt(transcript_row)

    for attempt in range(max_retries):
        try:
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
                "thinking_config": {
                    "include_thoughts": False,
                    "thinking_budget": 1024
                }
            }
            response = gemini_model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            raw = response.text.strip()
            
            collapsed_raw = raw.replace('\n', '\\n')
            logger.info(f"collapsed_raw: {collapsed_raw}")
            # logger.info(f"Raw: {raw}")

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            raw = _extract_json(raw)
            judgment = json.loads(raw)

            # Validating required fields
            assert "score" in judgment, "Missing 'score' field"
            assert "outcome" in judgment, "Missing 'outcome' field"
            assert isinstance(judgment["score"], (int, float)), "Score must be numeric"
            assert -1.0 <= judgment["score"] <= 1.0, f"Score out of range: {judgment['score']}"

            return judgment

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt+1}: JSON parse error — {e}")
            logger.warning(f"Raw response: {raw[:1000]}...")
        except AssertionError as e:
            logger.warning(f"Attempt {attempt+1}: Validation error — {e}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}: Gemini error — {e}")
        # Exponential backoff
        time.sleep(3 * (attempt + 1))  

    # Fallback if all retries fail
    logger.error("All retry attempts failed. Assigning fallback neutral score.")
    return judge_retry_fallback_response


def main(args):

    gemini_model=GenerativeModel(
    args.gemini_version,
    generation_config=GenerationConfig(
        temperature=0.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )
)
    logger.info("Gemini Judge initialized (temperature=0 for determinism).")

    # load transcripts
    t_bucket, t_blob = parse_gcs_uri(args.transcripts)
    local_transcripts = "/tmp/transcripts.jsonl"
    download_from_gcs(t_bucket, t_blob, local_transcripts)

    rows = []
    with open(local_transcripts) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.max_samples:
        rows = rows[:args.max_samples]

    logger.info(f"Scoring {len(rows)} transcripts ....")

    # score each trascript
    results = []
    score_distribution = {1.0: 0, 0.5: 0, 0.0: 0, -0.3: 0, -0.5: 0, -1.0: 0, "error": 0}

    for i, row in enumerate(rows):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Judging transcript {i+1}/{len(rows)}")
        logger.info(f"  Outcome from simulator: {row.get('outcome', 'unknown')}")

        judgment = score_transcript(gemini_model, row)
        score = judgment.get("score", 0.0)
        outcome = judgment.get("outcome", "unknown")

        logger.info(f"  Judge score  : {score:+.1f}")
        logger.info(f"  Judge outcome: {outcome}")
        logger.info(f"  Justification: {judgment.get('justification', '')}")

        rounded = round(score * 2) / 2 
        if judgment.get("error"):
            score_distribution["error"] += 1
        elif rounded in score_distribution:
            score_distribution[rounded] += 1

        result = {
            **row,
            "reward": score,
            "judgment": judgment,
        }
        results.append(result)

        time.sleep(0.5)

    local_out = "/tmp/rewards.jsonl"
    with open(local_out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    out_bucket, out_blob = parse_gcs_uri(args.output)
    upload_to_gcs(out_bucket, local_out, out_blob)

    # summarizing stats
    scores = [r["reward"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    logger.info(f"\n{'═'*60}")
    logger.info(f"Judge complete.")
    logger.info(f"  Total scored       : {len(results)}")
    logger.info(f"  Average reward     : {avg_score:+.3f}")
    logger.info(f"  Score distribution :")
    for score_val, count in sorted(score_distribution.items(), reverse=True):
        if count > 0:
            pct = count / len(results) * 100
            logger.info(f"    {str(score_val):>5} : {count:>4} ({pct:.1f}%)")
    logger.info(f"  Results saved      : gs://{out_bucket}/{out_blob}")
    logger.info(f"{'═'*60}")

    # Warn if hallucination rate is high
    hallucination_count = sum(
        1 for r in results
        if r.get("judgment", {}).get("reasoning", {}).get("hallucination_detected", False)
    )
    if hallucination_count / len(results) > 0.2:
        logger.warning(
            f"⚠ High hallucination rate: {hallucination_count}/{len(results)} "
            f"({hallucination_count/len(results)*100:.1f}%). "
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirDialogue AI Judge")

    parser.add_argument("--transcripts",  required=True,
                        help="GCS URI to transcripts JSONL from simulator.py")
    parser.add_argument("--output",       required=True,
                        help="GCS URI for output rewards JSONL")
    parser.add_argument("--gemini_key",   required=True,
                        help="Google Gemini API key")
    parser.add_argument("--max_samples",  type=int, default=None,
                        help="Maximum number of transcripts to score (for debugging)")
    parser.add_argument(
        "--gemini_version", 
        type=str, 
        default="gemini-2.5-flash", 
        help="The Vertex AI Gemini model version to use for the Simulator and Judge"
    )

    args = parser.parse_args()
    main(args)

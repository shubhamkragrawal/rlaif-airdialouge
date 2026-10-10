# """
# Reads completed conversation transcripts produced by simulator.py,
# scores each one against the original intent using Gemini API,
# and outputs a reward signal for the PPO training loop.
# """

# import os
# import json
# import argparse
# import logging
# import time
# from typing import Optional
# from constitution import JUDGE_CONSTITUTION
# from config import intent_field_map, judge_retry_fallback_response
# from utils import (
#     download_from_gcs,
#     upload_to_gcs,
#     parse_gcs_uri
#     )
# import vertexai
# from vertexai.generative_models import GenerativeModel, GenerationConfig

# # project config on VERTEX AI : project name: rlaif-project | server location: us-central1
# vertexai.init(project="rlaif-project", location="us-central1")

# # initiating logger 
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )
# logger = logging.getLogger(__name__)


# # formating our intent block for the judge same as the user simulator's intent
# def format_intent(intent: dict) -> str:
#     if not intent or not isinstance(intent, dict):
#         return "No specific constraints provided."
#     lines = []

#     for field, label in intent_field_map.items():
#         val = intent.get(field)
#         if val is not None and val != "" and val != "none":
#             lines.append(f"  {label}: {val}")
#     for k, v in intent.items():
#         if k not in intent_field_map and v is not None:
#             lines.append(f"  {k.replace('_',' ').title()}: {v}")
#     return "\n".join(lines) if lines else "No specific constraints."


# def format_flight_kb(kb: list) -> str:
#     if not kb:
#         return "No flights available."
#     lines = []
#     for i, flight in enumerate(kb, 1):
#         if not isinstance(flight, dict):
#             continue
#         parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in flight.items()]
#         lines.append(f"Flight {i}: " + " | ".join(parts))
#     return "\n".join(lines) if lines else "No flights available."


# def format_transcript(transcript: list) -> str:
#     if not transcript:
#         return "No conversation recorded."
#     lines = []
#     for turn in transcript:
#         role = turn.get("role", "unknown").upper()
#         text = turn.get("text", "").strip().replace('"', "'")
#         lines.append(f"{role}: {text}")
#     return "\n".join(lines)



# # builidng judge's prompt which decides it's behaviour
# # using Gemini 2.5 flash via API call from VERTEX AI playground as judge
# def build_judge_prompt(transcript_row: dict) -> str:
#     """
#     Build the full evaluation prompt for the Judge.
#     Combines the constitution with the specific conversation data.
#     """
#     intent_text    = format_intent(transcript_row.get("intent", {}))
#     kb_text        = format_flight_kb(transcript_row.get("kb", []))
#     transcript_text = format_transcript(transcript_row.get("transcript", []))

#     return f"""
#             {JUDGE_CONSTITUTION}


#             --- CUSTOMER'S HIDDEN INTENT ---
#             {intent_text}

#             --- AVAILABLE FLIGHT DATABASE ---
#             {kb_text}

#             --- CONVERSATION TRANSCRIPT ---
#             {transcript_text}

#             ==========================================
#             Now evaluate the agent's performance and output your JSON score.
#             ==========================================
#             """


# def score_transcript(gemini_model, transcript_row: dict, max_retries: int = 3,) -> dict:
#     """
#     Send transcript to Gemini Judge and parse the JSON response.
#     Returns the full judgment dict with score and reasoning.
#     """
#     prompt = build_judge_prompt(transcript_row)

#     for attempt in range(max_retries):
#         try:
#             response = gemini_model.generate_content(prompt)
#             raw = response.text.strip()
#             # logger.info(f"DEBUG [Attempt {attempt+1}] Raw Response: {raw}")
#             # Strip markdown fences if present (defensive)
#             if raw.startswith("```"):
#                 raw = raw.split("```")[1]
#                 if raw.startswith("json"):
#                     raw = raw[4:]
#                 raw = raw.strip()

#             judgment = json.loads(raw)

#             # Validate required fields
#             assert "score" in judgment, "Missing 'score' field"
#             assert "outcome" in judgment, "Missing 'outcome' field"
#             assert isinstance(judgment["score"], (int, float)), "Score must be numeric"
#             assert -1.0 <= judgment["score"] <= 1.0, f"Score out of range: {judgment['score']}"

#             return judgment

#         except json.JSONDecodeError as e:
#             logger.warning(f"Attempt {attempt+1}: JSON parse error — {e}")
#             logger.warning(f"Raw response: {raw}...")
#         except AssertionError as e:
#             logger.warning(f"Attempt {attempt+1}: Validation error — {e}")
#         except Exception as e:
#             logger.warning(f"Attempt {attempt+1}: Gemini error — {e}")
#         # Exponential backoff
#         time.sleep(3 * (attempt + 1))  

#     # Fallback if all retries fail — critical failure score, flagged for review
#     logger.error("All retry attempts failed. Assigning fallback neutral score.")
#     return judge_retry_fallback_response


# def main(args):

#     gemini_model=GenerativeModel(
#     args.gemini_version,
#     generation_config=GenerationConfig(
#         temperature=0.0,
#         max_output_tokens=8192,
#         response_mime_type="application/json",
#     )
# )
#     logger.info("Gemini Judge initialized (temperature=0 for determinism).")

#     # load transcripts
#     t_bucket, t_blob = parse_gcs_uri(args.transcripts)
#     local_transcripts = "/tmp/transcripts.jsonl"
#     download_from_gcs(t_bucket, t_blob, local_transcripts)

#     rows = []
#     with open(local_transcripts) as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 rows.append(json.loads(line))

#     if args.max_samples:
#         rows = rows[:args.max_samples]

#     logger.info(f"Scoring {len(rows)} transcripts ...")

#     # score each trascript
#     results = []
#     score_distribution = {1.0: 0, 0.5: 0, 0.0: 0, -0.3: 0, -0.5: 0, -1.0: 0, "error": 0}

#     for i, row in enumerate(rows):
#         logger.info(f"\n{'─'*60}")
#         logger.info(f"Judging transcript {i+1}/{len(rows)}")
#         logger.info(f"  Outcome from simulator: {row.get('outcome', 'unknown')}")

#         judgment = score_transcript(gemini_model, row)
#         score = judgment.get("score", 0.0)
#         outcome = judgment.get("outcome", "unknown")

#         logger.info(f"  Judge score  : {score:+.1f}")
#         logger.info(f"  Judge outcome: {outcome}")
#         logger.info(f"  Justification: {judgment.get('justification', '')}")

#         # Track distribution
#         rounded = round(score * 2) / 2  # rounding to nearest 0.5
#         if judgment.get("error"):
#             score_distribution["error"] += 1
#         elif rounded in score_distribution:
#             score_distribution[rounded] += 1

#         # Merge judgment into the transcript row
#         result = {
#             **row,
#             "reward": score,
#             "judgment": judgment,
#         }
#         results.append(result)

#         # Rate limit buffer
#         time.sleep(0.5)

#     local_out = "/tmp/rewards.jsonl"
#     with open(local_out, "w") as f:
#         for r in results:
#             f.write(json.dumps(r) + "\n")

#     out_bucket, out_blob = parse_gcs_uri(args.output)
#     upload_to_gcs(out_bucket, local_out, out_blob)

#     # summarizing stats
#     scores = [r["reward"] for r in results]
#     avg_score = sum(scores) / len(scores) if scores else 0.0

#     logger.info(f"\n{'═'*60}")
#     logger.info(f"Judge complete.")
#     logger.info(f"  Total scored       : {len(results)}")
#     logger.info(f"  Average reward     : {avg_score:+.3f}")
#     logger.info(f"  Score distribution :")
#     for score_val, count in sorted(score_distribution.items(), reverse=True):
#         if count > 0:
#             pct = count / len(results) * 100
#             logger.info(f"    {str(score_val):>5} : {count:>4} ({pct:.1f}%)")
#     logger.info(f"  Results saved      : gs://{out_bucket}/{out_blob}")
#     logger.info(f"{'═'*60}")

#     # Warn if hallucination rate is high : signals a prompt engineering problem
#     hallucination_count = sum(
#         1 for r in results
#         if r.get("judgment", {}).get("reasoning", {}).get("hallucination_detected", False)
#     )
#     if hallucination_count / len(results) > 0.2:
#         logger.warning(
#             f"⚠ High hallucination rate: {hallucination_count}/{len(results)} "
#             f"({hallucination_count/len(results)*100:.1f}%). "
#             f"Consider adding more grounding examples to SFT training data."
#         )



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="AirDialogue AI Judge")

#     parser.add_argument("--transcripts",  required=True,
#                         help="GCS URI to transcripts JSONL from simulator.py")
#     parser.add_argument("--output",       required=True,
#                         help="GCS URI for output rewards JSONL")
#     parser.add_argument("--gemini_key",   required=True,
#                         help="Google Gemini API key")
#     parser.add_argument("--max_samples",  type=int, default=None,
#                         help="Cap number of transcripts to score (for debugging)")
#     parser.add_argument(
#         "--gemini_version", 
#         type=str, 
#         default="gemini-2.5-flash", 
#         help="The Vertex AI Gemini model version to use for the Simulator and Judge"
#     )

#     args = parser.parse_args()
#     main(args)



import os
import json
import re
import argparse
import logging
import time
from constitution import JUDGE_CONSTITUTION
from config import intent_field_map, judge_retry_fallback_response
from utils import download_from_gcs, upload_to_gcs, parse_gcs_uri
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
        text = turn.get("text", "").strip().replace('"', "'")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)
 
 
def build_judge_prompt(transcript_row: dict) -> str:
    intent_text     = format_intent(transcript_row.get("intent", {}))
    kb_text         = format_flight_kb(transcript_row.get("kb", []))
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
 

def _safe_parse_score(raw: str) -> dict:
    """
    Extract score/outcome/justification from any Gemini response shape.
 
    Layer 1: json.loads on cleaned string
    Layer 2: sanitise control chars (fixes literal newlines in strings), json.loads
    Layer 3: regex targeting TOP-LEVEL fields using rfind
             (avoids matching score/outcome inside the reasoning block)
    Layer 4: hard fallback score=1 if response was truncated entirely
 
    The key insight for Layer 3: Gemini's response structure is:
      { "reasoning": { ...nested stuff... }, "score": X, "outcome": "...", ... }
    The top-level "score" always appears AFTER the reasoning block closes.
    Using rfind("score") finds the LAST occurrence = top-level, not reasoning.
    """
    # Layer 1: strip markdown fences, try json.loads
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
 
    try:
        parsed = json.loads(cleaned)
        score  = parsed.get("score")
        if score is not None:
            return _build_result(score, parsed.get("outcome"), parsed.get("justification"))
    except (json.JSONDecodeError, Exception):
        pass
 
    # Layer 2: sanitise ALL control characters (fixes literal \n inside strings)
    sanitised = re.sub(r"[\x00-\x1f\x7f]", " ", cleaned)
    try:
        parsed = json.loads(sanitised)
        score  = parsed.get("score")
        if score is not None:
            logger.warning("_safe_parse_score: layer 2 (control char sanitise) used")
            return _build_result(score, parsed.get("outcome"), parsed.get("justification"))
    except (json.JSONDecodeError, Exception):
        pass
 
    # Layer 3: regex on raw string — finds LAST (top-level) score occurrence
    logger.warning("_safe_parse_score: layer 3 (regex) used")
 
    # All score matches — take the LAST one (top-level comes after reasoning block)
    all_scores = re.findall(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
    score_val  = all_scores[-1] if all_scores else None
 
    # Find outcome and justification from AFTER the last score position
    # This ensures we get top-level fields, not reasoning sub-fields
    score_pos    = raw.rfind('"score"')
    outcome_val  = "unknown"
    just_val     = ""
    if score_pos >= 0:
        after_score = raw[score_pos:]
        o = re.search(r'"outcome"\s*:\s*"([^"]{1,80})"', after_score)
        j = re.search(r'"justification"\s*:\s*"([^"]{1,400})"', after_score)
        if o:
            outcome_val = o.group(1)
        if j:
            just_val = j.group(1)
 
    # Layer 4: if no score found at all, response was truncated
    if score_val is None:
        logger.warning("_safe_parse_score: layer 4 — response appears truncated, using fallback score=1")
        return _build_result(1, "judge_truncated", "Gemini response was truncated before score was written")
 
    return _build_result(score_val, outcome_val, just_val)
 
 
def _build_result(score, outcome, justification) -> dict:
    """Clamp score to 1-10 integer and return canonical dict."""
    try:
        clamped = max(1, min(10, int(round(float(score)))))
    except (TypeError, ValueError):
        clamped = 1
    return {
        "score":         clamped,
        "outcome":       str(outcome or "unknown"),
        "justification": str(justification or ""),
    }
 

def score_transcript(gemini_model, transcript_row: dict, max_retries: int = 3) -> dict:
    prompt = build_judge_prompt(transcript_row)
 
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            raw = response.text.strip()
 
            # Log full response as ONE line so Cloud Logging doesn't split it
            logger.info(f"Judge raw (attempt {attempt+1}): {raw.replace(chr(10), '|')}")
 
            result = _safe_parse_score(raw)
 
            # Validate score is in range
            assert 1 <= result["score"] <= 10, f"Score out of range: {result['score']}"
 
            logger.info(
                f"  Judge score  : {result['score']}/10 | "
                f"outcome: {result['outcome']} | "
                f"{result['justification'][:80]}"
            )
            return result
 
        except AssertionError as e:
            logger.warning(f"Attempt {attempt+1}: Validation error — {e}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}: Judge error — {e}")
 
        time.sleep(3 * (attempt + 1))
 
    logger.error("All retry attempts failed. Using fallback.")
    return judge_retry_fallback_response
 

def main(args):
    gemini_model = GenerativeModel(
        args.gemini_version,
        generation_config=GenerationConfig(
            temperature=0.0,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    logger.info("Gemini Judge initialized.")
 
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
 
    logger.info(f"Scoring {len(rows)} transcripts ...")
 
    results = []
    score_distribution = {s: 0 for s in range(1, 11)}
    score_distribution["error"] = 0
 
    for i, row in enumerate(rows):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Judging transcript {i+1}/{len(rows)}")
        logger.info(f"  Outcome from simulator: {row.get('outcome', 'unknown')}")
 
        judgment = score_transcript(gemini_model, row)
        score    = judgment.get("score", 1)
        outcome  = judgment.get("outcome", "unknown")
 
        logger.info(f"  Judge score  : {score}/10")
        logger.info(f"  Judge outcome: {outcome}")
        logger.info(f"  Justification: {judgment.get('justification', '')}")
 
        if judgment.get("error"):
            score_distribution["error"] += 1
        else:
            bucket = max(1, min(10, int(round(float(score)))))
            score_distribution[bucket] = score_distribution.get(bucket, 0) + 1
 
        result = {**row, "reward": score, "judgment": judgment}
        results.append(result)
        time.sleep(0.5)
 
    local_out = "/tmp/rewards.jsonl"
    with open(local_out, "w") as f:
        for r in results:
            record = {k: v for k, v in r.items() if k != "logprobs"}
            f.write(json.dumps(record) + "\n")
 
    out_bucket, out_blob = parse_gcs_uri(args.output)
    upload_to_gcs(out_bucket, local_out, out_blob)
 
    scores    = [r["reward"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
 
    logger.info(f"\n{'═'*60}")
    logger.info(f"Judge complete.")
    logger.info(f"  Total scored   : {len(results)}")
    logger.info(f"  Average score  : {avg_score:.2f} / 10")
    logger.info(f"  Score distribution:")
    for score_val in range(10, 0, -1):
        count = score_distribution.get(score_val, 0)
        if count > 0:
            pct = count / len(results) * 100
            logger.info(f"    {score_val:>2} : {count:>4} ({pct:.1f}%)")
    errors = score_distribution.get("error", 0)
    if errors:
        logger.info(f"  Errors         : {errors}")
    logger.info(f"  Results saved  : gs://{out_bucket}/{out_blob}")
    logger.info(f"{'═'*60}")
 
    hallucination_count = sum(
        1 for r in results
        if r.get("judgment", {}).get("reasoning", {}).get("hallucination_detected", False)
    )
    if results and hallucination_count / len(results) > 0.2:
        logger.warning(
            f"High hallucination rate: {hallucination_count}/{len(results)} "
            f"({hallucination_count/len(results)*100:.1f}%). "
            f"Consider adding more grounding examples to SFT training data."
        )
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirDialogue AI Judge")
    parser.add_argument("--transcripts",    required=True)
    parser.add_argument("--output",         required=True)
    parser.add_argument("--gemini_key",     required=True)
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--gemini_version", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()
    main(args)
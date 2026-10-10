"""Full held-out customer rollouts for base, SFT, DPO and PPO; resume at turn boundaries.

The customer simulator and optional secondary judge use Gemini. The headline task
metric is evaluated locally, never by Gemini. See eval_NOTES.md for restricted scope.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

from rlaif.evaluation.eval_core import (aggregate, check_outcome, fingerprint, make_manifest, named_flights,
                       paired_comparison, parse_marker)
from rlaif.shared.prompts import system_message, render_kb
from rlaif.shared.data import load_dialogues, load_splits, file_sha256

SIMULATOR_VERSION = "customer-v3-known-flight-offers"
MAX_SIMULATOR_SEMANTIC_REPAIRS = 2
SIMULATOR_PROMPT = """Act as the customer in a flight-booking chat. Your only private information
is CUSTOMER INTENT below. Do not invent preferences or change that intent. The airport
called return_airport in the intent is the destination. max_connections is an upper
bound; -1 means no preference. Dates use month and day, no year. Reveal your name exactly
as given when needed. Respond briefly and naturally to the latest agent message, answering
its questions. Mention missing constraints proactively if the agent is about to book
without asking about them. If the agent offers a specific flight and describes facts
consistent with your intent, agree to that flight explicitly; do not choose or invent a
flight ID on your own. If the agent says nothing matches, accept its answer and close.
You have no access to the flight database or the correct outcome. Do not pretend to
verify facts the agent did not state. Never write BOOKING_CONFIRMED or BOOKING_FAILED.
Return only JSON: {"text": "your next customer message", "accepted_flight": null}.
Set accepted_flight to an integer only when THIS message explicitly agrees to book that
specific flight already offered by the agent. In all other cases it must be null.
"""


class BudgetLimitError(RuntimeError):
    pass


class EvalGeminiClient:
    """Persist API attempts and conservatively reserve tokens before each request.

    Input UTF-8 byte count is a conservative tokenizer bound. Failed requests retain
    that reservation because their actual billed usage is unknown. Successful calls
    use reported usage when available. This is a resource cap, not a dollar invoice.
    """
    def __init__(self, model, usage_path, max_calls=1200, max_tokens=2000000,
                 max_output_tokens=512, max_retries=3):
        from google import genai
        from google.genai import types
        self.model, self.path = model, Path(usage_path)
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.max_output_tokens, self.max_retries = max_output_tokens, max_retries
        self.usage = json.loads(self.path.read_text()) if self.path.exists() else {
            "calls_attempted": 0, "calls_successful": 0, "errors": 0,
            "reported_total_tokens": 0, "reserved_unreported_tokens": 0}
        options = types.HttpOptions(timeout=60000)
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            self.client = genai.Client(api_key=key, http_options=options)
        else:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError("Set GOOGLE_CLOUD_PROJECT for Vertex ADC, or GOOGLE_API_KEY")
            self.client = genai.Client(vertexai=True, project=project,
                                      location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                                      http_options=options)

    def rank(self, prompt):
        from google.genai import types
        reserve = len(prompt.encode("utf-8")) + self.max_output_tokens
        config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                             max_output_tokens=self.max_output_tokens,
                                             thinking_config=types.ThinkingConfig(thinking_budget=0))
        for attempt in range(self.max_retries):
            u = self.usage
            used = u["reported_total_tokens"] + u["reserved_unreported_tokens"]
            if u["calls_attempted"] >= self.max_calls or used + reserve > self.max_tokens:
                atomic_json(self.path, u)
                raise BudgetLimitError("Gemini call/token cap reached; usage saved to %s" % self.path)
            u["calls_attempted"] += 1
            u["reserved_unreported_tokens"] += reserve
            atomic_json(self.path, u)
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                metadata = getattr(response, "usage_metadata", None)
                total = getattr(metadata, "total_token_count", None)
                if isinstance(total, int) and total >= 0:
                    u["reserved_unreported_tokens"] -= reserve
                    u["reported_total_tokens"] += total
                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("Gemini returned no text")
                u["calls_successful"] += 1
                atomic_json(self.path, u)
                return text
            except Exception as exc:
                u["errors"] += 1
                u["last_error"] = "%s: %s" % (type(exc).__name__, exc)
                atomic_json(self.path, u)
                if attempt + 1 == self.max_retries:
                    return None
                time.sleep(2 ** (attempt + 1))


def simulator_prompt(record, transcript):
    # Deliberately excludes KB, labels, gold dialogue, and evaluator matching set.
    return SIMULATOR_PROMPT + "\nCUSTOMER INTENT:\n" + json.dumps(record["intent"], sort_keys=True) + \
        "\nCHAT:\n" + "\n".join("%s: %s" % (t["role"], t["content"]) for t in transcript)


def simulator_repair_prompt(record, transcript, invalid_raw, validation_error):
    """Ask the same simulator to repair invalid JSON without advancing the chat."""
    return simulator_prompt(record, transcript) + """

Your previous JSON response was rejected by the evaluator and was not added to the chat.
Return a corrected response for the exact same latest agent message. Keep the customer
intent unchanged. Do not claim consent to any flight number unless that exact number was
already offered by the agent in the chat above. If no offered flight can be accepted,
set accepted_flight to null and respond naturally without inventing a flight.
VALIDATION ERROR:
%s
REJECTED JSON:
%s
Return only corrected JSON.
""" % (validation_error, invalid_raw)


def parse_customer(raw, transcript, known_flights=None):
    obj = json.loads(raw)
    text, flight = obj.get("text"), obj.get("accepted_flight")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Simulator returned no customer text")
    if "BOOKING_CONFIRMED" in text or "BOOKING_FAILED" in text:
        raise ValueError("Simulator returned an agent marker")
    if flight is not None:
        if isinstance(flight, bool) or not isinstance(flight, int):
            raise ValueError("accepted_flight must be integer or null")
        assistant_text = [t["content"] for t in transcript if t["role"] == "assistant"]
        offers = set().union(*(named_flights(text) for text in assistant_text))
        # Model replies often present a valid KB flight as a bare table row or say
        # "the flight is 1010". Match only known flight IDs so dates, times and prices
        # cannot be mistaken for offers.
        for known in known_flights or ():
            if any(re.search(r"(?<!\d)%s(?!\d)" % re.escape(str(known)), text)
                   for text in assistant_text):
                offers.add(int(known))
        if flight not in offers:
            raise ValueError("Simulator consent references an unoffered flight")
    return text.strip(), flight


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


class Policy:
    def __init__(self, model_name, adapter=None, revision=None, max_new_tokens=160):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Evaluation requires a CUDA GPU; use --dry-run for data checks")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    def reply(self, record, transcript):
        messages = [{"role": "system", "content": system_message(record["kb"])}] + transcript
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        limit = getattr(self.model.config, "max_position_embeddings", 32768)
        if prompt_tokens + self.max_new_tokens > limit:
            raise ValueError("Rollout exceeds model context; refusing to silently truncate")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True).strip()


def secondary_judge_prompt(record, transcript):
    return """Grade this completed flight-booking chat against its database and customer intent.
Score grounding, factual accuracy, respecting constraints, progress and concise helpful
conversation on a scale from 1 (serious invention) to 10 (correct and useful). Do not use
an outcome label or assume a confirmation marker proves success. Return only JSON:
{"score": <number 1-10>, "reasoning": "brief explanation"}.
""" + "\n" + render_kb(record["kb"]) + "\nCUSTOMER INTENT:\n" + json.dumps(record["intent"]) + \
        "\nCHAT:\n" + "\n".join("%s: %s" % (t["role"], t["content"]) for t in transcript)


def run_scenario(record, state, policy, simulator, max_turns, save, judge=None):
    if state.get("status") == "complete":
        return state
    state["status"] = "running"
    state.pop("error", None)
    state.setdefault("transcript", [{"role": "user", "content": "Hello, I would like to book a flight."}])
    state.setdefault("accepted_flights", [])
    state.setdefault("simulator_calls", [])
    while True:
        transcript = state["transcript"]
        turns = sum(t["role"] == "assistant" for t in transcript)
        if transcript[-1]["role"] == "assistant":
            marker, _ = parse_marker(transcript[-1]["content"])
            if marker is not None or turns >= max_turns:
                state["termination"] = "marker" if marker is not None else "max_turns"
                break
            prompt = simulator_prompt(record, transcript)
            text = accepted = None
            last_error = None
            for semantic_attempt in range(MAX_SIMULATOR_SEMANTIC_REPAIRS + 1):
                raw = simulator.rank(prompt)
                call = {"phase": "primary" if semantic_attempt == 0 else "semantic_repair",
                        "semantic_attempt": semantic_attempt,
                        "prompt_sha256": fingerprint(prompt), "raw": raw}
                state["simulator_calls"].append(call)
                if not raw:
                    save(state)
                    raise RuntimeError("Customer simulator failed; scenario remains resumable")
                try:
                    text, accepted = parse_customer(
                        raw, transcript, {int(f["flight_number"]) for f in record["kb"]})
                    save(state)
                    break
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    last_error = exc
                    call["validation_error"] = "%s: %s" % (type(exc).__name__, exc)
                    save(state)
                    if semantic_attempt < MAX_SIMULATOR_SEMANTIC_REPAIRS:
                        prompt = simulator_repair_prompt(record, transcript, raw, call["validation_error"])
            if text is None:
                raise RuntimeError("Invalid customer simulator output after bounded repair: %s" % last_error)
            transcript.append({"role": "user", "content": text})
            if accepted is not None:
                state["accepted_flights"].append(accepted)
            save(state)
        else:
            reply = policy.reply(record, transcript)
            if not reply:
                raise RuntimeError("Policy returned an empty message")
            transcript.append({"role": "assistant", "content": reply})
            save(state)
    state["metrics"] = check_outcome(record, state)
    state["judge_score"] = None
    if judge:
        raw = judge.rank(secondary_judge_prompt(record, state["transcript"]))
        state["judge_raw"] = raw
        try:
            score = json.loads(raw)["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 10:
                raise ValueError("Invalid score")
            state["judge_score"] = float(score)
        except (ValueError, TypeError, KeyError):
            state["judge_error"] = "Missing or invalid secondary judge score"
    state["status"] = "complete"
    save(state)
    return state


def result_files(directory):
    return [json.loads(p.read_text()) for p in sorted((Path(directory) / "scenarios").glob("*.json"))]


def compare(directories):
    manifests = [json.loads((Path(d) / "eval_manifest.json").read_text()) for d in directories]
    if len({m["manifest_sha256"] for m in manifests}) != 1:
        raise ValueError("Comparisons require identical evaluation manifests")
    results = [result_files(d) for d in directories]
    expected = {s["dialogue_id"] for s in manifests[0]["scenarios"]}
    for rows in results:
        if {r["dialogue_id"] for r in rows if r.get("status") == "complete"} != expected:
            raise ValueError("Finish every planned scenario before final paired comparison")
    return {"manifest_sha256": manifests[0]["manifest_sha256"],
            "comparisons": [{"left": str(directories[i]), "right": str(directories[j]),
                             **paired_comparison(results[i], results[j])}
                            for i in range(len(results)) for j in range(i + 1, len(results))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter")
    parser.add_argument("--arm", choices=["base", "sft", "dpo", "ppo"], default="base")
    parser.add_argument("--out", default=str(Path(__file__).parent / "artifacts" / "eval-base"))
    parser.add_argument("--scenario-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-agent-turns", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--simulator-model", default="gemini-2.5-flash")
    parser.add_argument("--secondary-judge", action="store_true")
    parser.add_argument("--max-gemini-calls", type=int, default=1200)
    parser.add_argument("--max-gemini-tokens", type=int, default=2000000)
    parser.add_argument("--max-gemini-output-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gcs-output")
    parser.add_argument("--resume-from", help="Restore a prior eval output tree from gs://")
    parser.add_argument("--compare", nargs="+", metavar="EVAL_DIR")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(args.compare), indent=2))
        return
    if (args.arm == "base") != (not args.adapter):
        parser.error("Base takes no adapter; trained arms require --adapter")
    if min(args.max_agent_turns, args.max_new_tokens, args.max_gemini_calls,
           args.max_gemini_tokens, args.max_gemini_output_tokens) < 1:
        parser.error("Turn and token limits must be positive")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        if not args.resume_from.startswith("gs://"):
            parser.error("--resume-from must be a gs:// URI")
        from cloud.artifacts import download_tree
        download_tree(args.resume_from, root)
    def sync():
        if args.gcs_output:
            from cloud.artifacts import sync_tree
            sync_tree(root, args.gcs_output)
    dialogues, splits = load_dialogues(), load_splits()
    manifest = make_manifest(dialogues, splits, args.scenario_count, args.seed)
    manifest_path = root / "eval_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new output directory")
    atomic_json(manifest_path, manifest)
    print("[eval] eligible=%d selected=%d exclusions=%s" %
          (manifest["eligible_count"], args.scenario_count, manifest["exclusion_counts"]), flush=True)
    if args.dry_run:
        sync()
        print("[eval] dry run complete; no model loaded or Gemini calls made")
        return
    adapter = args.adapter
    if adapter and adapter.startswith("gs://"):
        from cloud.artifacts import download_tree
        adapter = download_tree(adapter, root.parent / "input_adapters" / fingerprint(adapter))
    adapter_digest = None
    if adapter:
        files = [p for p in Path(adapter).iterdir() if p.name.startswith("adapter_") and p.is_file()]
        if not any(p.name == "adapter_config.json" for p in files):
            raise ValueError("Adapter path does not contain adapter_config.json")
        adapter_digest = fingerprint({p.name: file_sha256(p) for p in files})
    config = {"arm": args.arm, "model": args.model, "revision": args.revision,
              "adapter_source": args.adapter, "adapter_sha256": adapter_digest,
              "max_agent_turns": args.max_agent_turns, "max_new_tokens": args.max_new_tokens,
              "decoding": "greedy", "simulator_model": args.simulator_model,
              "simulator_version": SIMULATOR_VERSION, "simulator_temperature": 0.0,
              "secondary_judge": args.secondary_judge, "manifest_sha256": manifest["manifest_sha256"],
              "max_gemini_calls": args.max_gemini_calls, "max_gemini_tokens": args.max_gemini_tokens,
              "max_gemini_output_tokens": args.max_gemini_output_tokens,
              "system_prompt_sha256": fingerprint(system_message([]))}
    config_path = root / "eval_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing evaluation config differs; use a new output directory")
    atomic_json(config_path, config)
    policy = Policy(args.model, adapter, args.revision, args.max_new_tokens)
    revision_path = root / "model_revision.json"
    revision = {"resolved_model_revision": policy.resolved_revision}
    if revision_path.exists() and json.loads(revision_path.read_text()) != revision:
        raise ValueError("Model revision changed; pass --revision or use a fresh output directory")
    atomic_json(revision_path, revision)
    simulator = EvalGeminiClient(args.simulator_model, root / "api_usage.json", args.max_gemini_calls,
                                args.max_gemini_tokens, args.max_gemini_output_tokens)
    judge = simulator if args.secondary_judge else None
    by_id = {r["dialogue_id"]: r for r in dialogues}
    errors = 0
    for scenario in manifest["scenarios"]:
        dialogue_id = scenario["dialogue_id"]
        path = root / "scenarios" / ("%05d.json" % dialogue_id)
        state = json.loads(path.read_text()) if path.exists() else {
            "dialogue_id": dialogue_id, "line_hash": scenario["line_hash"], "arm": args.arm,
            "manifest_sha256": manifest["manifest_sha256"], "started_at_unix": time.time()}
        if state.get("manifest_sha256") != manifest["manifest_sha256"] or state.get("line_hash") != scenario["line_hash"]:
            raise ValueError("Saved scenario does not match its held-out manifest")
        if state.get("status") == "complete":
            continue
        def save(value):
            atomic_json(path, value)
            sync()
        try:
            run_scenario(by_id[dialogue_id], state, policy, simulator, args.max_agent_turns, save, judge)
        except Exception as exc:
            state["status"], state["error"] = "error", "%s: %s" % (type(exc).__name__, exc)
            save(state)
            errors += 1
            print("[eval] scenario %d error: %s" % (dialogue_id, exc), flush=True)
            if isinstance(exc, BudgetLimitError):
                atomic_json(root / "summary.json", aggregate(result_files(root), args.scenario_count))
                sync()
                raise
        atomic_json(root / "summary.json", aggregate(result_files(root), args.scenario_count))
        sync()
        print("[eval] %s scenario=%d status=%s" % (args.arm, dialogue_id, state["status"]), flush=True)
        # Stop after persistent failures instead of burning calls over the whole pool.
        if errors >= 3:
            raise RuntimeError("Three scenario errors; inspect saved failures and resume")
    summary = aggregate(result_files(root), args.scenario_count)
    summary["api_usage"] = simulator.usage
    atomic_json(root / "summary.json", summary)
    sync()
    print(json.dumps(summary, indent=2))
    if summary["provisional"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

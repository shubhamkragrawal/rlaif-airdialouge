# Held-out evaluation

**Scope: 50 booking-only scenarios where the stored structured intent completely
recovers the labelled acceptable-flight set. This is a restricted benchmark, not a
result over all AirDialogue tasks. No GPU evaluation has run yet.**

## Why the eligibility gate exists

The held-out split contains 250 scenarios: 187 booking requests and 63 changes or
cancellations. The KB lacks reservation state, so the latter cannot be evaluated
honestly. Of the 187 booking requests, 82 have acceptable-flight labels that disagree
with the flights matching every stored intent field. Those records often carry extra
preferences only in their original customer dialogue:

- Dialogue 8: intent allows flights 1012 and 1022, but the original customer asks for
  an evening return. Their return hours are 12 and 20; only 1022 is labelled acceptable.
- Dialogue 445: intent permits flight 1007, but the original customer asks for afternoon
  departure and return. Its return hour is 10; the label is `no_flight`.
- Dialogue 234: several flights meet stored fields, but the label lists only two.
  The stored intent and customer transcript do not fully explain the restriction.

Day strings are normalized numerically (`01` equals `1`), months and airport strings
case-insensitively, max_price is inclusive, and max_connections is an inclusive upper
bound with `-1` unrestricted. There are no stored cabin/time fields in these 187 intents.
Unknown future intent fields cause an error instead of being ignored. The gate excludes
all disagreements, irrespective of their cause. It leaves 105 scenarios, from which
the same 50 are sampled with seed 20260922 for every arm. All exclusions, source SHA256,
line hashes and the selected IDs are saved in `eval_manifest.json`.

## Metrics and limits

- **Task success:** valid terminal `BOOKING_CONFIRMED <flight>` for a matching flight,
  after the customer gave their full name and explicit consent, or `BOOKING_FAILED`
  when no flight matches. Because of the eligibility gate, matching flights also equal
  the gold acceptable-flight set. This is a transcript outcome proxy: no actual booking
  API exists. Correct no-flight outcomes do not require the name. A lucky decline can
  therefore score correctly; progress and dialogue quality remain separate measures.
- Booking success among scenarios with a valid flight is reported separately so no-flight
  outcomes do not inflate the booking rate. Wilson 95% intervals accompany task success.
- Unknown-flight-ID hallucination detects explicit flight mentions and markers referring
  to IDs absent from the KB. **It does not detect all incorrect prices, airlines, times
  or implicit flight references.** Do not label it a general hallucination rate.
- Average agent turns and average turns on successful scenarios are reported separately.
- Optional Gemini score is secondary, with scored count and missing scores explicit.
  Missing/error responses never become default scores.
- Paired comparison requires every planned scenario completed under identical manifests;
  it reports success-rate differences and seeded paired bootstrap intervals.

The Gemini customer receives only structured intent and the generated transcript,
never the KB, gold agent turns, acceptable-flight labels or evaluator matching set.
Its consent references must be to flights already mentioned by the policy. It can
still make simulation mistakes. All raw responses are retained for inspection. Model
decoding is greedy; Gemini temperature is zero, but hosted responses can vary. Same
scenarios do not mean identical customer responses after different policy replies.
Customer simulation uses the `gemini-2.5-flash` API with thinking disabled, a 512-token
output limit, 60-second request timeout, and at most three attempts per turn.

## Run

From the RLAIF directory, after the parent builds the image with the eval files:

```bash
PYTHONPATH=rebuild python3 -m rlaif.evaluation.eval_qwen --dry-run --out rebuild/artifacts/eval-base

PYTHONPATH=rebuild python3 -m rlaif.evaluation.eval_qwen --arm base --out /tmp/eval-base \
  --gcs-output gs://YOUR_BUCKET/eval/base
PYTHONPATH=rebuild python3 -m rlaif.evaluation.eval_qwen --arm sft --adapter gs://YOUR_BUCKET/SFT_ADAPTER \
  --out /tmp/eval-sft --gcs-output gs://YOUR_BUCKET/eval/sft
PYTHONPATH=rebuild python3 -m rlaif.evaluation.eval_qwen --arm dpo --adapter gs://YOUR_BUCKET/DPO_ADAPTER \
  --out /tmp/eval-dpo --gcs-output gs://YOUR_BUCKET/eval/dpo
PYTHONPATH=rebuild python3 -m rlaif.evaluation.eval_qwen --compare /tmp/eval-base /tmp/eval-sft /tmp/eval-dpo
```

Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION for Vertex ADC, or GOOGLE_API_KEY.
The service account needs Gemini invocation and output-bucket write permissions.
Container dependencies are the pinned torch/transformers/peft stack plus google-genai
and google-cloud-storage. No extra eval dependency is needed. Pass `--secondary-judge`
to enable scores; omit it to minimize API spend. For a first GPU smoke rollout use
`--scenario-count 2 --max-agent-turns 4` in a separate output directory; do not mix this
manifest with the final 50-scenario benchmark.

The shared call/token caps default to `--max-gemini-calls 1200` and
`--max-gemini-tokens 2000000` per arm, including retries and optional judge calls.
`api_usage.json` stores reported tokens and conservative reservations for attempts
without usage metadata. These are resource caps, **not dollar billing guarantees**;
the parent must enforce the overall $150 soft/$200 hard spending limits and job timeouts.
The cap is checked before each API request and failures preserve evidence. There is
also a three-scenario-error stop. A successful retry after an API exception still
consumes the earlier call/token reservation.

Per-turn JSON state is saved atomically and synchronized when --gcs-output is set.
Re-running the same output directory skips complete scenarios and resumes incomplete
ones without replaying saved policy messages. On a fresh managed worker pass
`--resume-from gs://...` to restore the eval output tree into `--out`. Changes to config,
manifest, adapter fingerprint or resolved model revision require a fresh directory.
Pin `--revision` to the recorded resolved_model_revision for later reproducibility.

## Verification completed locally

Nine standard-library tests cover constraint boundaries, strict markers, consent/name,
unknown IDs, unavailable flights, oracle isolation, interrupted-turn resume, the real
held-out gate and disjointness, paired/incomplete reporting, and API caps stopping before
network requests. The manifest dry run
passed with eligible=105, selected=50, excluded=82 hidden/ambiguous intent plus 63
reservation scenarios. GPU loading and live Gemini behavior still require the cloud
smoke rollout; these local checks do not imply measured policy performance.

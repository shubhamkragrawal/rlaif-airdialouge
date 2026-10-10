# RLAIF — Repository Summary
_Audited 2026-09-21. Read-only pass; no files were modified except this one._

> **Note on scope:** the brief asked for a commit-history narrative and a git-tracking
> audit. This directory is **not a git repository** — there is no `.git/` and no
> `.gitignore`. Every git-dependent question (commit history, contributors, branches,
> tracked-vs-ignored, secrets in deleted history) is therefore unanswerable and is
> marked as such throughout. File mtimes are the only chronology available, and the
> Vertex AI log exports in `vertex_ai_logs/` carry real timestamps.

---

## 1. What this project is

A university course project (SUNY Buffalo, Institute for AI and Data Science) that builds
a **flight-booking dialogue agent trained by Reinforcement Learning from AI Feedback**,
with no human annotators anywhere in the loop. A Mistral-7B base model is first
supervised-fine-tuned with LoRA on Google's AirDialogue corpus
(`baseline/fine_tune/new_train_1.py:266`), then improved by online RL in which **Gemini 2.5
Flash plays both the customer** (`baseline/rlaif_pipeline/simulator.py:194`) **and the
reward-assigning judge** (`baseline/rlaif_pipeline/judge.py:436`). Two RL algorithms were
implemented against that same loop: a hand-rolled **PPO** with a value head
(`baseline/rlaif_pipeline/ppo_trainer.py:38`) and a **REINFORCE-with-learned-baseline**
variant following the direct-RLAIF (d-RLAIF) formulation of Lee et al.
(`baseline/Reinfore_baseline/rlaif_pipeline.py:362`).

Everything ran on Vertex AI Custom Training against GCS. The work stopped when the team
exhausted their Google Cloud credits (stated in `final_project_ryadav4_sa69_vshinde.pdf`,
page 5, "Challenges"), which is why every RL run in this repo is 10–48 episodes long
rather than the thousands such a method needs.

## 2. Current state

**Partially built — and the PPO entry point is currently broken.**

The SFT stage is complete and demonstrably ran to completion (a full 2-epoch, 20k-example
run is logged end to end in `vertex_ai_logs/logs-sft-20k_finetune.json`, final
`train_loss: 0.6219`). Both RL loops also genuinely ran and produced real logged metrics.
But the PPO rollout function as committed contains a `NameError` —
`baseline/rlaif_pipeline/simulator.py:155` indexes `log_probs_all[0, torch.arange(...), s]`
where `s` is never assigned anywhere in the file — so `python rlaif_pipeline.py --ppo` dies
on the first agent turn. The sibling implementation at
`baseline/Reinfore_baseline/simulator.py:168-171` has the correct indexing
(`new_token_ids`), which shows this is a regression introduced into the PPO copy after the
runs that produced the logs, not a design gap.

## 3. Architecture

```
                    airdialogue_sampled_5k.jsonl  (5,000 rows: intent + kb + dialogue)
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
           ┌────────▼─────────┐          ┌─────────▼──────────┐
           │  STAGE 1: SFT    │          │  used again as the │
           │ new_train_1.py   │          │  prompt source for │
           │ Mistral-7B + LoRA│          │  every RL rollout  │
           │ 4-bit NF4, r=16  │          └─────────┬──────────┘
           └────────┬─────────┘                    │
                    │ LoRA adapter → GCS           │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │              STAGE 2: RL ROLLOUT LOOP               │
        │                                                     │
        │   Gemini 2.5 Flash          Mistral+LoRA policy     │
        │   "customer" (T=0.8)  ◄────►  "agent" (greedy)      │
        │   hidden intent block         sees KB, 128 tok max  │
        │        │                                            │
        │        └── emits BOOKING_CONFIRMED / BOOKING_FAILED │
        │            → terminates episode                     │
        └──────────────────────────┬──────────────────────────┘
                                   │ full transcript
                    ┌──────────────▼───────────────┐
                    │   Gemini 2.5 Flash JUDGE     │
                    │   T=0.0, JSON mime type      │
                    │   rubric = constitution.py   │
                    │   → soft distribution over   │
                    │     ranks 1..10 → scalar     │
                    └──────────────┬───────────────┘
                                   │ scalar reward
              ┌────────────────────┴────────────────────┐
              │                                         │
   ┌──────────▼───────────┐               ┌─────────────▼────────────┐
   │ PATH A: PPO          │               │ PATH B: REINFORCE+base   │
   │ rlaif_pipeline/      │               │ Reinfore_baseline/       │
   │ ppo_trainer.py       │               │ rlaif_pipeline.py        │
   │ clipped ratio ε=0.2  │               │ (1-β)·PG + β·KL + value  │
   │ separate value optim │               │ two-adapter policy/ref   │
   └──────────────────────┘               └──────────────────────────┘
```

The two paths are **parallel forks, not a shared library**. `config.py`, `constitution.py`,
`judge.py`, `simulator.py` and `utils.py` all exist in duplicate under
`baseline/rlaif_pipeline/` and `baseline/Reinfore_baseline/`, and they have drifted apart
(see §5). Nothing is importable as a package; every module imports its siblings flat
(`from config import ...`), which works only because the Dockerfiles copy all files into a
single `/app` directory (`baseline/Dockerfile.rlaif:41-47`).

## 4. The RLAIF loop

**Prompts / scenarios.** Every episode is seeded from one row of
`airdialogue_sampled_5k.jsonl` (5,000 rows). A row carries an `intent` dict (the customer's
hidden goal) and a `kb` list of ~30 flight dicts that is the ground truth for the judge.
Rows missing either are skipped (`baseline/rlaif_pipeline/rlaif_pipeline.py:214`).

**User simulator.** A prompted LLM, not a script or fixed dataset. Gemini 2.5 Flash at
`temperature=0.8`, `max_output_tokens=512`
(`baseline/rlaif_pipeline/rlaif_pipeline.py:81-87`). It is handed `SIMULATOR_CONSTITUTION`
(`baseline/rlaif_pipeline/constitution.py:400-458`) as a pseudo-system message inside a
chat session, then told "Begin the conversation"
(`baseline/rlaif_pipeline/simulator.py:209`). The constitution instructs it to stay in
character, never front-load its constraints, never change its mind, and to terminate by
emitting the literal token `BOOKING_CONFIRMED` or `BOOKING_FAILED` on its own line; the
driver string-matches those to end the episode (`baseline/rlaif_pipeline/simulator.py:180-185`).

**AI judge.** Gemini 2.5 Flash at `temperature=0.0` with
`response_mime_type="application/json"` (`baseline/rlaif_pipeline/rlaif_pipeline.py:88-95`).
The live rubric is `JUDGE_CONSTITUTION` at
`baseline/rlaif_pipeline/constitution.py:294-386`. It scores **factual accuracy against the
flight KB only** — the file contains two older, commented-out rubrics that scored
constraint satisfaction and conversation quality instead
(`baseline/rlaif_pipeline/constitution.py:17-115` and `190-292`), so the reward definition
was narrowed at least twice. The current rubric asks the judge to:

1. run a chain-of-thought check over flight existence, airline/time/airport/stops/price
   accuracy, hallucination, and omission;
2. place a **probability distribution over ranks 1–10** (soft preference labelling);
3. return the expected rank `E = Σ i·P(i)` as `score`.

Rank anchors run from `[Rank 1] Complete fabrication` to `[Rank 10] Perfect factual
execution`; `[Rank 8]` is "fully accurate but did not book", `[Rank 9]` is "fully accurate
with a confirmed booking".

**Parse-failure handling** is unusually thorough — four layers in
`_safe_parse_score` (`baseline/rlaif_pipeline/judge.py:355-420`): plain `json.loads` after
stripping fences; retry after scrubbing control characters; regex using `rfind('"score"')`
to grab the *last* (top-level) score rather than one nested in the reasoning block; and a
hard fallback of `score=1, outcome="judge_truncated"` when the response was cut off. Three
retries with backoff wrap that (`baseline/rlaif_pipeline/judge.py:439-464`), falling back
to `judge_retry_fallback_response` (`baseline/rlaif_pipeline/config.py:78`).

**Reward construction — there are no preference pairs.** This is the single largest
divergence from what the README advertises. Despite the README's "AI Preference Generation
& Reward Modeling" and "RL Optimization (PPO / DPO)", **no chosen/rejected pair is ever
constructed and no DPO trainer exists anywhere in the repo.** A grep for
`DPO|chosen|rejected` across all source directories returns only two stale comments
(`baseline/rlaif_pipeline/rlaif_pipeline.py:7`,
`baseline/rlaif_pipeline/constitution.py:7`). The judge's scalar is fed straight into the
policy gradient — that is what "direct-RLAIF" means, and it is a deliberate design choice
documented in the poster, but the README was never updated to match. `N_CANDIDATES` was
dropped from 2 to 1 with the inline note "pivoted to refining of 1 response"
(`baseline/rlaif_pipeline/config.py:51`); it is now imported in two places and read in
none, and the "best-of-2" naming survives only in the output filename
`rlaif_best_of_2.jsonl`.

**Reward scaling differs between the two paths** and this is easy to trip over:

| | PPO path | REINFORCE path |
|---|---|---|
| judge asked to return | integer-ish 1–10 (`constitution.py:382`) | already-normalised −1..+1 (`Reinfore_baseline/constitution.py:96`) |
| code then does | `(score − 5.5) / 4.5` (`ppo_trainer.py:114`) | passes through unchanged (`Reinfore_baseline/rlaif_pipeline.py:283`) |
| soft distribution | **discarded** — `int(round(float(score)))` at `judge.py:426` | preserved as a float |

**On-disk datasets — all present, none of them large:**

| File | Rows | Contents |
|---|---|---|
| `baseline/rlaif_pipeline/output.jsonl` | 10 | judged transcripts, Apr 27 PPO run |
| `baseline/output.jsonl` | 15 | judged transcripts, Apr 22 PPO run |
| `baseline/rlaif_best_of_2.jsonl` | 43 | judged transcripts, Apr 24–25 PPO run |
| `baseline/Offline_rl_datasets/rlaif_offline_database_100.jsonl` | 100 | offline RL records + pre-tokenised ids/masks |
| `baseline/Offline_rl_datasets/rlaif_offline_database_200.jsonl` | 100 | second shard, same schema |

Both "offline database" files hold 100 rows with `db_index` 0–99 — the `_200` suffix
denotes the second batch of 100, not 200 rows. They were generated by
`baseline/Reinfore_baseline/generate_offline_db.py`, which targets `--n_conversations 1000`
by default; the poster confirms the 1,000-conversation run was abandoned mid-way for lack
of credits and that the offline-RL model was never trained.

## 5. Training

| | PPO | REINFORCE + baseline (d-RLAIF) | *(DPO)* |
|---|---|---|---|
| **library / implementation** | Hand-written, no TRL. `baseline/rlaif_pipeline/ppo_trainer.py:38`. `trl` appears in `requirements.txt:84` but is imported only by the SFT script. | Hand-written. `baseline/Reinfore_baseline/rlaif_pipeline.py:362`. Imports `create_reference_model`, `get_peft_model`, `LoraConfig`, `TaskType` at lines 80–81 and **uses none of them**. | **Does not exist.** No trainer, no pair builder, no β. Referenced only in two comments. |
| **key hyperparameters** | `lr=1e-5` policy / `1e-6` value (`ppo_trainer.py:97-98`), `clip ε=0.2`, `value_coef=0.5`, `kl_coef=0.05`, `γ=0.99`, `grad_clip=1.0` policy / `0.1` value, `accum=3`, `log-ratio clamp ±10` (`ppo_trainer.py:13-21`) | `--beta 0.05`, `--lr 1e-5`, `--epochs 8` default (`Reinfore_baseline/rlaif_pipeline.py:109-111`), `--gradient_accum 4`, `--max_turns 10`, `grad_clip=1.0` | — |
| **KL handling** | **Not a KL divergence.** `_kl_penalty` returns `clamp(−mean(new_logprobs), min=0)` (`ppo_trainer.py:355-357`) — a negative-log-likelihood/entropy term that never consults the reference model. The frozen `ref_model` is loaded in 4-bit (`rlaif_pipeline.py:121`), stored and frozen (`ppo_trainer.py:74-78`), and then **never read again**. | Genuine full-vocabulary KL: `Σ exp(p)·(p − q)` masked to response tokens (`Reinfore_baseline/rlaif_pipeline.py:292-306`), against a real frozen reference. Combined as `(1−β)·pg + β·kl + value_loss` (line 450). | — |
| **reference model** | Loaded, frozen, unused (see above). Also note it loads the *base* model, not the SFT checkpoint, contradicting the startup banner's "Reference model: Frozen SFT checkpoint" (`rlaif_pipeline.py:142`). | Correct: same base weights, two PEFT adapters named `reference` (frozen) and `policy` (trainable), swapped via `set_adapter` (`Reinfore_baseline/rlaif_pipeline.py:185-199`). Memory-efficient and the better of the two implementations. | — |
| **value head** | Yes — bias-free `Linear(hidden, 1)` per token position, init `N(0, 0.001)` (`ppo_trainer.py:24-35`). Separate optimizer with tight gradient clipping, added specifically because values were exploding (`ppo_trainer.py:94-98`). | Yes — `Linear(hidden, 1)` on the **last** hidden position only, zero-init, one scalar per sequence (`Reinfore_baseline/rlaif_pipeline.py:127-139`). | — |
| **status** | Ran (three logged runs). **Currently broken** at `simulator.py:155`. | Ran to completion once, 48 rollouts. Believed runnable. | Not started. |

**Shared vs duplicated.** Effectively nothing is shared. `utils.py` is near-identical in
both trees (only the `datetime` import style and the checkpoint blob names differ).
`simulator.py` and `judge.py` are forks that have drifted in meaningful ways — the judge's
parse hardening (`_safe_parse_score`) exists only on the PPO side, while the correct
logprob indexing exists only on the REINFORCE side.

**Cross-cutting issue affecting both algorithms.** Rollouts are generated with
`do_sample=False` (`rlaif_pipeline/simulator.py:134`, `Reinfore_baseline/simulator.py:154`),
i.e. greedy decoding. Both files pass a `temperature` alongside it (`0.1` and `0.0`
respectively) which HuggingFace ignores when sampling is off, and the PPO copy's comment
claims `temperature=0.0` while the code says `0.1`. The REINFORCE module further declares
`SAMPLING_TEMPERATURE = 0.9` (`Reinfore_baseline/rlaif_pipeline.py:108`) and never uses it.
A deterministic policy makes the policy-gradient estimator degenerate: the same prompt
always yields the same action, so there is no exploration for either algorithm to learn
from. This is very likely a bigger contributor to the flat reward curves than the episode
count.

**SFT.** `baseline/fine_tune/new_train_1.py`: Mistral-7B-v0.1 in 4-bit NF4 with double
quantisation (line 192), LoRA `r=16, α=32, dropout=0.1` on `q/k/v/o_proj` only (line 214),
`lr=1e-4` cosine with 5% warmup, `max_grad_norm=0.3`, `fp16`, `max_seq_length=4096`,
per-device batch 1 × grad-accum 4, gradient checkpointing on (line 266). Loss is masked to
response tokens only via `DataCollatorForCompletionOnlyLM` with hardcoded Mistral token ids
`[28792, 16289, 28793]` / `[733, 28748, 16289, 28793]` (line 229), guarded by a sanity
check that aborts if the collator masks everything (line 254) — a good defensive touch,
and the comment at line 60 records that an earlier verbose system prompt did exactly that.

## 6. Results

Every number below is copied from the file named. No metric in this table was inferred.

### SFT — `vertex_ai_logs/logs-sft-20k_finetune.json`

| Metric | Value |
|---|---|
| Hardware | 2 × NVIDIA A100-SXM4-40GB |
| Examples | 20,000 (20,000 raw rows, 0 skipped) |
| Epochs | 2 |
| Effective batch | 16 |
| Learning rate | 1e-4 |
| First logged loss | 1.7984 (epoch 0.01) |
| Final logged loss | 0.5161 (epoch 2.0) |
| `train_loss` | 0.6219 |
| `train_runtime` | 15,659.23 s (≈4h21m) |
| `train_samples_per_second` | 2.554 |
| Collator check | 120 / 1257 tokens visible to loss |
| Output | `models/sft-mistral-20k-v3-2epoch` |
| Wall clock | 2026-04-27 13:34 → 18:00 UTC |

> **Discrepancy:** this log reports `GPUs: 2`, `Effective batch: 16`, and the banner
> "Starting SFT training v4 — 2xA100". The script in this repo hardcodes
> `device_map={"": 0}` (single GPU), computes effective batch `1 × 4 = 4`
> (`new_train_1.py:304`), and logs the banner without the "— 2xA100" suffix
> (`new_train_1.py:301`). **The committed SFT script is not the version that produced this
> log.** The multi-GPU variant is not in the repo.

### PPO — three runs

| Run | Artifact (metrics) | Artifact (transcripts) | Episodes | Mean judge score | Optimizer updates |
|---|---|---|---|---|---|
| Apr 22 | `vertex_ai_logs/logs-ppo_50steps.json` | `baseline/output.jsonl` | 15 | **2.60 / 10** | 1 |
| Apr 24–25 | `baseline/rlaif_pipeline/RL_50_convo_log.json` | `baseline/rlaif_best_of_2.jsonl` | 43 | **1.88 / 10** | 8 |
| Apr 27 | `vertex_ai_logs/logs-ppo.json` | `baseline/rlaif_pipeline/output.jsonl` | 10 | **2.40 / 10** | 3 |

Score distribution, 43-episode run (`baseline/rlaif_best_of_2.jsonl`): `1`×29, `3`×10,
`5`×2, `6`×2. Outcome labels: `hallucination` 25, `confused` 11, `incomplete` 3,
`judge_truncated` 3, `constraint_violation` 1.

Score distribution, 15-episode run (`baseline/output.jsonl`): `1`×10, `3`×3, `8`×1, `10`×1.
Outcomes: `hallucination` 5, `confused` 4, `constraint_violation` 2, `judge_truncated` 2,
`incomplete` 1, `confirmed` 1.

Score distribution, 10-episode run (`baseline/rlaif_pipeline/output.jsonl`): `1`×6, `2`×1,
`3`×2, `10`×1. Outcomes: `judge_truncated` 4, `fabrication` 2, `major_inaccuracy` 2,
`hallucination` 1, `perfect_accuracy` 1.

Representative PPO metric rows, copied verbatim:

- Apr 22, first step — `policy_loss 0.6194, value_loss 24.2031, kl_loss 2.977, total_loss 12.8698, ratio_mean 1.146`
- Apr 22, step 11 onward — `value_loss: nan` for the **final 5 of 15 steps**; the value head
  diverged and never recovered in that run.
- Apr 27, first step — `policy_loss 0.67, value_loss 0.5986, kl_loss 3.187, total_loss 3.857, ratio_mean 1.2605, ratio_max 23.2591`
- Apr 27 run shows **no `nan`** — the separate value optimizer with `grad_clip=0.1` fixed it.
- `ratio_max` reaches **52.83** (Apr 27) and **44.26** (Apr 24) — far outside the clip band
  [0.8, 1.2], indicating the policy moved a long way from the behaviour policy on individual
  tokens.

The Apr 24–25 run logged 43 episodes against 8 optimizer updates, implying
`PPO_ACCUM_STEPS = 5` at the time; the file now says `3` (`ppo_trainer.py:20`). The Apr 22
logs also lack the `ratio_max` key that `ppo_trainer.py:235` now emits. **None of the three
logged PPO runs was produced by the code currently in the repo.**

### REINFORCE + baseline — `vertex_ai_logs/logs_reinforce_baseline_50.json`

Final summary block, copied verbatim from the log:

| Metric | Value |
|---|---|
| Total rollouts | 48 |
| **Average reward** | **−0.473** |
| perfect (+1.0 … +0.6) | 4 (8.3%) |
| partial (+0.6 … +0.2) | 3 (6.2%) |
| neutral (+0.2 … −0.2) | 2 (4.2%) |
| wrong booking (−0.2 … −0.6) | 15 (31.2%) |
| critical fail (−0.6 … −1.0) | **24 (50.0%)** |
| Model saved | `models/d-rlaif-rfb-v1` |
| Wall clock | 2026-04-30 04:55 → 09:15 UTC (≈4h20m for 48 rollouts) |

Nine logged optimizer steps (at global steps 5, 10, …, 45), implying `gradient_accum = 5`
for this run — the default in the file is `4` (`Reinfore_baseline/rlaif_pipeline.py:679`).
The poster independently states "10 training steps, one in every 5 steps", confirming 5.

| Step | loss | pg | kl | vl | adv |
|---|---|---|---|---|---|
| 5 | −131.0587 | −138.6414 | 0.0009 | 0.6506 | −0.9766 |
| 10 | −316.4532 | −333.6832 | 0.0013 | 0.5457 | −0.7656 |
| 15 | −70.8369 | −74.8970 | 0.0012 | 0.3151 | +0.5234 |
| 20 | −257.8285 | −272.0507 | 0.0014 | 0.6195 | −0.4102 |
| 25 | −280.3862 | −295.7775 | 0.0012 | 0.6023 | −0.6797 |
| 30 | −168.0817 | −177.2117 | 0.0013 | 0.2693 | −0.7656 |
| 35 | −252.1120 | −265.8881 | 0.0014 | 0.4816 | −0.4805 |
| 40 | +127.6691 | +133.6569 | 0.0012 | 0.6949 | −0.8672 |
| 45 | −255.3980 | −269.4213 | 0.0011 | 0.5521 | −0.4746 |

KL stays at ~0.001 throughout — the policy barely moved from the reference, consistent
with greedy rollouts and 9 optimizer steps.

### Offline RL datasets

| File | Rows | Mean reward | Min | Max | confirmed | failed | max_turns |
|---|---|---|---|---|---|---|---|
| `rlaif_offline_database_100.jsonl` | 100 | **−0.4491** | −1.0 | +1.0 | 26 | 65 | 9 |
| `rlaif_offline_database_200.jsonl` | 100 | **−0.5277** | −1.0 | +1.0 | 27 | 61 | 12 |

Judge-outcome labels in the first file: `major_inaccuracy` 44, `fabrication` 11,
`hallucination` 11, `severe_hallucination` 11, `fully_accurate` 6, `complete_fabrication` 5,
`perfect_accuracy` 4, `perfect_factual_execution` 4, `accurate_with_a_trivial_discrepancy` 2,
`mostly_accurate_with_a_minor_error` 1, `partial_accuracy` 1.
Sequence lengths: min 3,151, max 4,038, mean 3,521 tokens — close enough to the 4,096
truncation ceiling (`generate_offline_db.py:154`) that some transcripts are being clipped.

### PPO-vs-REINFORCE comparison

**There is none, and one cannot be constructed from these artifacts.** The two paths use
different reward scales (1–10 integer vs −1..+1 float), different judge rubrics, different
SFT checkpoints (`sft-mistral-v6-1-epoch` vs `sft-mistral-20k-v3-2epoch`), different episode
counts (43 vs 48) and different source rows. No shared held-out evaluation set, no
evaluation script, and no metric other than the training-time judge score exists anywhere
in the repo. The poster's Results section presents the two side by side but draws no
quantitative comparison either.

### Plots on disk

- `baseline/rlaif_pipeline/plots/` — `reward.png`, `raw_score.png`, `total_loss.png`,
  `policy_loss.png`, `value_loss.png`, `kl_loss.png`, `ratio_mean.png`, `ratio_max.png`,
  `reward_histogram.png`, `did_update.png` (10 files, generated by `plot_all.py`)
- `baseline/rlaif_pipeline/dashboard_24th_April.png` — 3×3 PPO dashboard
- `baseline/rlaif_pipeline/training_curves_PPO_50_old.png`
- `final_project_ryadav4_sa69_vshinde.pdf` — conference poster (p.1) plus 6 pages of
  appendix with the d-RLAIF reward plot, KL/PG/value loss curves and a challenges write-up

### Claims without artifacts

- The poster states a **1,000-conversation** offline RL database was being generated. Only
  200 rows exist on disk, and the poster itself says the run was cut short by credit
  exhaustion.
- The poster reports "550 Gemini calls in 50 conversations" and "around 7 hours to train for
  50 conversations". The log supports ≈4h20m for 48 rollouts; the 7-hour figure has no
  artifact behind it. (The 43-episode PPO run did span ≈7h15m — possibly a conflation.)
- No trained adapter weights are present in this repo for any stage; all model artifacts
  live in the GCS bucket, which this audit cannot see.

## 7. Repo map

| Path | Purpose | Status |
|---|---|---|
| `README.md` | Project blurb + two `gcloud builds submit` commands | Stale — advertises DPO and a reward model, neither of which exists |
| `requirements.txt` | 93 pinned packages | Present; missing `matplotlib` (see §9) |
| `RLAIF_AUDIT_BRIEF.md` | The brief this document answers | Meta |
| `SUMMARY.md` | This file | New |
| `airdialogue_sampled_5k.jsonl` | 5,000 AirDialogue rows, 55 MB | Present; the sole data source |
| `final_project_ryadav4_sa69_vshinde.pdf` | Poster + 6-page appendix | Complete, and the best narrative record in the repo |
| `vertex_ai_logs/logs-sft-20k_finetune.json` | 341 entries, full 2-epoch SFT run | Complete |
| `vertex_ai_logs/logs-ppo.json` | 509 entries, 10 PPO steps, Apr 27 | Complete; **newer than `logs-ppo_50steps.json` despite the name** |
| `vertex_ai_logs/logs-ppo_50steps.json` | 544 entries, **15** PPO steps, Apr 22 | Misnamed — not 50 steps |
| `vertex_ai_logs/logs_reinforce_baseline_50.json` | 1,445 entries, 48 rollouts | Complete, with final summary |
| `baseline/cloudbuild.yaml` | 5-line Cloud Build, parameterised by `_IMAGE`/`_DOCKERFILE` | Complete |
| `baseline/Dockerfile.finetune` | CUDA 12.1 + torch 2.2.2, entrypoint `new_train_1.py` | Complete |
| `baseline/Dockerfile.rlaif` | Same base, entrypoint `rlaif_pipeline.py` | Complete |
| `baseline/fine_tune/new_train_1.py` | SFT entry point (329 ln) | Runnable; **older than the log it's paired with** |
| `baseline/fine_tune/utils.py` | GCS + Secret Manager helpers | Complete; `download_dir_from_gcs` and `parse_gcs_uri` unused here |
| `baseline/rlaif_pipeline/rlaif_pipeline.py` | PPO entry point (359 ln) | **Broken** via `simulator.py:155` |
| `baseline/rlaif_pipeline/ppo_trainer.py` | Hand-rolled PPO + value head (365 ln) | Complete but KL is not a KL; `ref_model` unused |
| `baseline/rlaif_pipeline/simulator.py` | Rollout driver + standalone CLI (354 ln) | **`NameError` at line 155** |
| `baseline/rlaif_pipeline/judge.py` | Judge + 4-layer parser (571 ln) | Working; lines 1–269 are a commented-out prior version |
| `baseline/rlaif_pipeline/constitution.py` | Live rubric + 2 dead ones (457 ln) | Working; ~63% commented out |
| `baseline/rlaif_pipeline/config.py` | Constants (96 ln) | Working; ~half commented out; `N_CANDIDATES` dead |
| `baseline/rlaif_pipeline/utils.py` | GCS + checkpointing | Complete |
| `baseline/rlaif_pipeline/plot_all.py` | 9-panel PPO dashboard from Cloud Logging JSON | Working |
| `baseline/rlaif_pipeline/plot_training_graphs.py` | Single reward curve from output JSONL | Working |
| `baseline/rlaif_pipeline/output.jsonl` | 10 judged transcripts | Data |
| `baseline/rlaif_pipeline/RL_50_convo_log.json` | 43 PPO metric rows | Data |
| `baseline/rlaif_pipeline/plots/*.png` | 10 generated plots | Data |
| `baseline/Reinfore_baseline/rlaif_pipeline.py` | REINFORCE entry point (882 ln) | Complete; the most careful RL code here |
| `baseline/Reinfore_baseline/generate_offline_db.py` | Offline dataset builder (485 ln) | Complete; target 1,000, delivered 200 |
| `baseline/Reinfore_baseline/simulator.py` | Rollout driver fork (396 ln) | Complete; **has the correct logprob indexing** |
| `baseline/Reinfore_baseline/judge.py` | Judge fork (298 ln) | Complete; simpler parser than the PPO fork |
| `baseline/Reinfore_baseline/constitution.py` | Rubric returning −1..+1 (173 ln) | Complete |
| `baseline/Reinfore_baseline/config.py`, `utils.py` | Near-duplicates of the PPO copies | Duplicated |
| `baseline/Offline_rl_datasets/*.jsonl` | 2 × 100 offline records | Data |
| `baseline/output.jsonl`, `baseline/rlaif_best_of_2.jsonl` | 15 + 43 judged transcripts | Data; loose at `baseline/` root |
| `.DS_Store` (×N) | macOS cruft | Should be removed before publishing |

Directory name `Reinfore_baseline` is misspelled (should be `Reinforce_`). Renaming it
would break `Dockerfile.rlaif`-style relative copies if any script referenced it — none do,
so it is safe to rename.

## 8. How to run it

Reconstructed from the code and the two Dockerfiles. **Nothing below was executed** — this
was a read-only audit — so every step is unverified unless noted.

**Prerequisites**

- A GCP project with Vertex AI, GCS, Cloud Build and Secret Manager enabled.
- A GCS bucket (the code defaults to `rlaif-data-bucket-rlaif-project`).
- A Secret Manager secret literally named `hf-token` holding a Hugging Face token with
  access to `mistralai/Mistral-7B-v0.1` — fetched at
  `baseline/rlaif_pipeline/rlaif_pipeline.py:75`. **This is a hard dependency; the pipeline
  raises before doing anything else if the secret is absent.**
- An A100-class GPU. `new_train_1.py:158` raises `RuntimeError("No GPU detected")` outright;
  the RL paths hardcode `.to("cuda")` and `device_map={"": 0}`.
- Upload `airdialogue_sampled_5k.jsonl` to the bucket.
- For the REINFORCE path only: a **copy of the Mistral base model already in GCS**, passed
  via `--base_model_prefix` (`Reinfore_baseline/rlaif_pipeline.py:497`). The PPO path pulls
  the base from Hugging Face instead. This asymmetry is undocumented elsewhere.

**No env vars are read directly**; `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` are *set* by the
code from Secret Manager. GCP auth is ambient (ADC / the Vertex service account).
`vertexai.init(project="rlaif-project", location="us-central1")` is hardcoded at four call
sites, so a different project requires source edits — the `--project` flag only controls
Secret Manager and GCS, not Vertex.

**1 — Build images** (from `baseline/`, exactly as `README.md:27-41` gives):

```bash
gcloud builds submit --project <PROJECT> --config cloudbuild.yaml \
  --substitutions _IMAGE=gcr.io/<PROJECT>/mistral-sft-trainer:v8_3,_DOCKERFILE=Dockerfile.finetune \
  --timeout 30m --machine-type e2-highcpu-8 .
```

**2 — SFT** (Vertex Custom Job; args from `new_train_1.py:322-327`):

```
--bucket=<BUCKET> --data_blob=airdialogue_sampled_5k.jsonl
--output_prefix=models/sft-mistral-20k-v3-2epoch --epochs=2
```
Produces a LoRA adapter at `gs://<BUCKET>/<output_prefix>/`. Expect ≈4h20m on 2×A100 for
20k examples × 2 epochs.

**3a — PPO** (`rlaif_pipeline.py:346-359`). **This will fail as committed** — see §9 item 1.

```
--project=<PROJECT_NUMBER> --bucket=<BUCKET> --data_blob=airdialogue_sampled_5k.jsonl
--model_prefix=models/sft-mistral-20k-v3-2epoch --max_samples=50
--gemini_version=gemini-2.5-flash --overwrite --ppo
```
Writes judged transcripts to `gs://<BUCKET>/rlaif/output/rlaif_best_of_2.jsonl` and the
adapter to `models/ppo-mistral-<UTC timestamp>/`. Resumable — checkpoints every 50 examples
to `rlaif/checkpoints/` (`config.py:50-55`); omit `--overwrite` to resume.

**3b — REINFORCE + baseline** (`Reinfore_baseline/rlaif_pipeline.py:42-56` documents a
working invocation):

```
--project=<PROJECT_NUMBER> --bucket=<BUCKET> --data_blob=airdialogue_sampled_5k.jsonl
--sft_model_prefix=models/sft-mistral-v6-1-epoch --base_model_prefix=models/mistral-base
--output_prefix=models/d-rlaif-rfb-v1 --max_samples=50 --epochs=1 --max_turns=5
--beta=0.05 --lr=1e-5 --gradient_accum=4 --overwrite
```
Uploads adapter + `value_head.pt` + `reward_plot.png` under `--output_prefix`.
Budget ≈5 min/rollout — dominated by sequential Gemini calls, not GPU.

**4 — Offline dataset** (`generate_offline_db.py:26-35` gives a complete example).

**5 — Plots** (local, CPU-only; the only steps runnable without GCP):

```bash
python3 plot_all.py --input RL_50_convo_log.json --output dashboard.png --window 10
python3 plot_training_graphs.py --input ../output.jsonl --output training_curves.png --window 20
```
Both need `matplotlib`, which is **not** in `requirements.txt` — install it separately.

**Unverifiable steps:** every GCP-dependent step above. Also note the standalone
`simulator.py` / `judge.py` CLIs both mark `--gemini_key` as `required=True`
(`simulator.py:350`, `judge.py:568`) but never read it — auth is ADC. Pass any placeholder.

## 9. Gaps and unfinished work

Ordered by how much each would cost someone picking this repo up cold.

1. **The PPO entry point does not run.** `baseline/rlaif_pipeline/simulator.py:155`
   references an undefined `s`. Fix: index with `new_token_ids`, as
   `baseline/Reinfore_baseline/simulator.py:171` already does. One-token change, but it is
   the difference between "runnable" and "not".
2. **DPO is advertised and absent.** `README.md:16` promises "RL Optimization (PPO / DPO)";
   there is no DPO trainer, no preference-pair construction, and no `beta`. Either
   implement it or correct the README — as written, the repo's front door misdescribes it.
3. **PPO's KL penalty is not a KL divergence.** `ppo_trainer.py:355-357` returns
   `clamp(−mean(new_logprobs), 0)`, which penalises low-likelihood output generally rather
   than drift from the reference. The reference model is loaded into 4-bit GPU memory
   (`rlaif_pipeline.py:121`) purely to sit unused. Either wire it in (the REINFORCE path at
   `Reinfore_baseline/rlaif_pipeline.py:292-306` is a correct template) or drop the load and
   rename the term.
4. **Greedy rollouts break the policy gradient.** `do_sample=False` at
   `rlaif_pipeline/simulator.py:134` and `Reinfore_baseline/simulator.py:154`. With a
   deterministic policy there is no exploration and the REINFORCE/PPO estimators are
   degenerate. `SAMPLING_TEMPERATURE = 0.9` at `Reinfore_baseline/rlaif_pipeline.py:108`
   suggests sampling was intended and never plumbed through.
5. **The committed code does not match any logged run.** The SFT script lacks the
   multi-GPU support its own log shows (§6); `PPO_ACCUM_STEPS` is 3 but the runs used 5;
   the Apr 22 log predates the `ratio_max` metric. Anyone trying to reproduce the numbers
   will not, and nothing in the repo warns them.
6. **`intent_field_map` does not match the dataset.** `config.py:59-73` maps `budget`,
   `class`, `num_people`, `departure_time_num`, `return_time_num`; the actual rows carry
   `max_price`, `name`, `goal`, `max_connections` and no `budget` key at all. So
   `format_intent` (`simulator.py:30-41`) never emits a "Maximum budget (USD)" line, and
   leaks `Name: <synthetic name>` into the customer's own hidden-intent block via the
   catch-all loop at line 38 — while the judge rubric is still checking budget compliance.
7. **The judge's soft labels are thrown away on the PPO path.** The rubric spends a whole
   step building `preference_distribution` over ranks 1–10
   (`constitution.py:345-350`), then `_build_result` collapses it with
   `int(round(float(score)))` (`judge.py:426`). The REINFORCE fork keeps the float. If the
   d-RLAIF soft-labelling is the point, the PPO path is discarding it.
8. **No evaluation harness at all.** No held-out split, no eval script, no success-rate or
   hallucination-rate metric computed over a fixed set. The only number anywhere is the
   training-time judge score, which is also the training signal — so there is no
   independent measurement of whether any of this worked.
9. **No tests.** `find -iname "*test*"` returns nothing.
10. **`matplotlib` missing from `requirements.txt`** despite three modules importing it
    (`plot_all.py:9`, `plot_training_graphs.py:5`,
    `Reinfore_baseline/rlaif_pipeline.py:70`). `scipy` and `einops` are likewise installed
    only inside the Dockerfiles.
11. **Dead code and unused imports.** `N_CANDIDATES` imported twice, read never
    (`config.py:51`); `create_reference_model`, `get_peft_model`, `LoraConfig`, `TaskType`
    imported and unused (`Reinfore_baseline/rlaif_pipeline.py:80-81`); `--gemini_key`
    required but ignored in both standalone CLIs; ~270 commented-out lines at the head of
    `rlaif_pipeline/judge.py` and ~290 in `rlaif_pipeline/constitution.py`.
12. **Judge truncation is a live data-quality problem.** 4 of 10 rows in
    `rlaif_pipeline/output.jsonl` and 3 of 43 in `rlaif_best_of_2.jsonl` carry
    `outcome: "judge_truncated"`, meaning the reward for those episodes is the fallback
    `1`, not a real judgment. The comment at `rlaif_pipeline.py:92` flags this as unresolved.
13. **Wholesale duplication between the two trees.** Five modules exist twice and have
    drifted; fixes applied to one silently miss the other (which is exactly how item 1
    arose).

## 10. Pre-publication checklist

Findings only — no values are reproduced below.

**Credential-shaped files:** none. No `.env`, `.pem`, `.key`, `credentials*`,
`*service-account*`, `.p12` or `id_rsa*` anywhere in the tree.

**No secret material found in source.** Scans for HF-token, Google-API-key and
private-key patterns returned only false positives (the substring `sk-` inside
"Task-Oriented" in `README.md`). Both pipelines fetch the HF token from Secret Manager at
runtime (`rlaif_pipeline/utils.py:23-26`) rather than embedding it — the right pattern.

**Hardcoded identifiers in source that will become public.** All are GCP resource names,
not credentials; none grants access on its own, but together they fingerprint the project
and the bucket, and a world-readable bucket name is worth checking before publishing.

| Path:line | Kind of value |
|---|---|
| `baseline/rlaif_pipeline/simulator.py:21` | GCP project id + region, hardcoded in `vertexai.init` |
| `baseline/rlaif_pipeline/judge.py:285` | GCP project id + region |
| `baseline/rlaif_pipeline/judge.py:24` | GCP project id + region (commented-out block) |
| `baseline/rlaif_pipeline/rlaif_pipeline.py:79` | GCP project id + region |
| `baseline/rlaif_pipeline/rlaif_pipeline.py:14,15,17` | GCP **project number**, bucket name, model path (docstring) |
| `baseline/rlaif_pipeline/rlaif_pipeline.py:24,25,27` | GCP project number, bucket name, model path (docstring) |
| `baseline/Reinfore_baseline/simulator.py:26` | GCP project id + region |
| `baseline/Reinfore_baseline/judge.py:24` | GCP project id + region |
| `baseline/Reinfore_baseline/rlaif_pipeline.py:466` | GCP project id + region |
| `baseline/Reinfore_baseline/rlaif_pipeline.py:42,43` | GCP project number, bucket name (docstring) |
| `baseline/Reinfore_baseline/generate_offline_db.py:27,28,239` | GCP project number, bucket name, project id |
| `baseline/fine_tune/new_train_1.py:3` | Bucket name (docstring) |
| `baseline/fine_tune/utils.py:48` | Bucket name (comment) |
| `README.md:28,30,37,39` | GCP project id, GCR image paths |
| `baseline/rlaif_pipeline/plot_training_graphs.py:10` | Bucket name inside an example `gcloud` command |

Also present: the **secret name** `hf-token` at `rlaif_pipeline/utils.py:25` and
`fine_tune/utils.py:16` — the name only, never a value. Harmless, but it tells a reader
exactly which secret to target.

**Email addresses in tracked content:** three `@buffalo.edu` addresses appear in the
poster footer of `final_project_ryadav4_sa69_vshinde.pdf` (page 1). These are the authors'
own and were already published on a conference poster; flagging for awareness only.

**Log exports contain the full GCP resource path.** All four files in `vertex_ai_logs/`
and `baseline/rlaif_pipeline/RL_50_convo_log.json` are raw Cloud Logging exports whose
`logName` and `resource.labels` fields carry the project id, project number, job ids and
region on **every one of ~2,800 entries**. Scrubbing the source files alone will not
remove the identifiers from the repo.

**Data that arguably shouldn't ship:**
- `airdialogue_sampled_5k.jsonl` — 55 MB, and 55 MB of the 60 MB repo. It is a sample of a
  public Google dataset (`huggingface.co/datasets/google/air_dialogue`), so redistribution
  is likely fine, but it is large enough to warrant a download script instead. Note its
  `intent.name` fields contain synthetic person names (e.g. the AirDialogue generator's
  output), not real PII.
- `.DS_Store` files — macOS cruft, no reason to publish.
- The three loose transcript JSONLs and two offline-DB shards are legitimate result
  artifacts and are small; keeping them is reasonable.

**Cannot be assessed without git:** whether any secret was ever committed and later
deleted, and whether `data/` or `outputs/` content is tracked. There is no history to
search. If this directory is a working copy detached from its original repository — the
poster states "Github for code versioning" — **the audit of the real history has not been
done**, and it should be run against that repository before making it public.

## 11. Open questions for the owner

1. Is there a git repository for this project elsewhere? Publishing from this snapshot
   means no history is audited for secrets.
2. Was the multi-GPU SFT script (the one that logged "2xA100", effective batch 16) lost, or
   does it exist outside this directory?
3. Is `PPO_ACCUM_STEPS = 3` (`ppo_trainer.py:20`) intentional, or should it be 5 to match
   the runs that produced the logged results?
4. Was PPO's `_kl_penalty` (`ppo_trainer.py:355`) meant to be a true KL against
   `ref_model`, or is the entropy-style term deliberate?
5. Should PPO's reference model be the SFT checkpoint rather than the base model?
   `rlaif_pipeline.py:121` loads `args.base_model_id`; the startup banner at line 142 claims
   the SFT checkpoint.
6. Was greedy decoding (`do_sample=False`) a deliberate choice for stability, or was
   `SAMPLING_TEMPERATURE = 0.9` meant to be wired into `model.generate`?
7. Is `Offline_rl_datasets/rlaif_offline_database_200.jsonl` rows 100–199 of the same
   intended 1,000-row database, i.e. should the two files be concatenated?
8. Do the trained adapters (`models/sft-mistral-20k-v3-2epoch`, `models/d-rlaif-rfb-v1`,
   `models/ppo-mistral-*`) still exist in the GCS bucket, or has that project been deleted?
9. Is the GCS bucket named in the source still live, and is it private?
10. Should `--gemini_key` be removed from the standalone `simulator.py` / `judge.py` CLIs,
    given both authenticate via ADC and ignore the flag?
11. Is the `rlaif_best_of_2.jsonl` filename vestigial from the abandoned best-of-2 design
    (`N_CANDIDATES` 2→1), or does it still mean something?

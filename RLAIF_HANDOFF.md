# RLAIF — Project Handoff

**For:** a fresh session picking this up cold.
**Owner:** Shubham Kumar Agrawal
**Written:** 21 September 2026
**Timebox:** 3 working days (22–24 Sep). Rerouted — the main portfolio project — starts 30 Sep and must not slip.

---

## 0 · Read this first

There is an existing RLAIF repo from a university course project. It has been audited read-only; the findings are in `SUMMARY.md` at its root. **Read that file before touching anything.**

The short version: the repo does not do what the owner's resume says it does. This project exists to close that gap by actually building the PPO-vs-DPO comparison, on a new base model, with a real held-out evaluation.

We are **not** repairing the old pipeline. We are building a small, clean one beside it and reusing the old repo's data and rubrics.

---

## 1 · Why this is happening

The owner's resume claims:

> *"Evaluating PPO versus DPO using PyTorch and Hugging Face, focusing on stabilizing model training with KL-divergence penalties to prevent semantic drift."*

The audit found:

- **No DPO exists.** No trainer, no preference-pair construction, no β. Two stale comments only (`rlaif_pipeline.py:7`, `constitution.py:7`).
- **The PPO path's "KL penalty" is not a KL divergence.** `ppo_trainer.py:355-357` returns `clamp(−mean(new_logprobs), 0)` — an entropy-style term that never reads the reference model. The frozen `ref_model` is loaded into 4-bit GPU memory and never used again.
- The **REINFORCE path** does have a genuine full-vocabulary KL (`Reinfore_baseline/rlaif_pipeline.py:292-306`) — so the claim is true of a path the resume doesn't name.

So there are two jobs: **fix the resume** (see §8, do this regardless) and **build the thing** (§3–§6).

---

## 2 · State of the existing repo

From the audit. Every item has a file:line reference in `SUMMARY.md`.

**Blocking**
- **Not a git repository.** No `.git`, no `.gitignore`. The poster states GitHub was used for versioning — so a real repo exists elsewhere and **has never been scanned for secrets in history**. Find it before publishing anything.
- **The GCS bucket has been cleaned.** No SFT adapter, no trained weights survive. Everything trains from scratch.
- **PPO entry point is broken.** `baseline/rlaif_pipeline/simulator.py:155` indexes with an undefined `s`. The sibling at `Reinfore_baseline/simulator.py:171` has the correct `new_token_ids`. One-line fix if we want the PPO arm.

**Technically important**
- **Greedy decoding in both paths.** `do_sample=False` at `rlaif_pipeline/simulator.py:134` and `Reinfore_baseline/simulator.py:154`. A deterministic policy makes both policy-gradient estimators degenerate — no exploration, nothing to learn from. `SAMPLING_TEMPERATURE = 0.9` is declared at `Reinfore_baseline/rlaif_pipeline.py:108` and never wired in. This is very likely a bigger cause of the flat reward curves than the low episode count, and it is **also why no preference pairs can exist** — sampling twice returns the same string.
- **`intent_field_map` doesn't match the data.** `config.py:59-73` maps `budget`, `class`, `num_people`; the actual rows carry `max_price`, `name`, `goal`, `max_connections`. So the simulator never emits a budget line while the judge rubric still checks budget compliance.
- **No evaluation harness at all.** No held-out split, no eval script. The only number anywhere is the training-time judge score — which is also the training signal, so nothing independent was ever measured.
- **Committed code doesn't match any logged run.** `PPO_ACCUM_STEPS` is 3, the runs used 5; the SFT script is single-GPU, the log is 2×A100. The old numbers are not reproducible.

**Results as they stand** — PPO mean judge score 1.88–2.60/10 across three runs; REINFORCE average reward −0.473 with 50% critical failures, 48 rollouts. Both runs stopped when the team's GCP credits ran out.

**Context** — this was a **three-person course project** (`final_project_ryadav4_sa69_vshinde.pdf`), SUNY Buffalo. Any public write-up must say so and name the owner's contribution specifically.

---

## 3 · What we are building

A small pipeline that produces a **four-arm comparison on a shared held-out set**.

| Arm | Notes |
|---|---|
| **A** Qwen2.5-3B-Instruct, zero-shot | Costs nothing but eval. Answers the question a reviewer asks first: *did any of this beat just prompting the instruct model?* |
| **B** + SFT (10k × 1 epoch) | The shared starting policy for every RL arm |
| **C** + SFT + DPO | The main deliverable |
| **D** + SFT + PPO | Optional. Only if time allows and the `simulator.py:155` fix goes in |

**The single most important constraint:** arms C and D must start from **the same SFT checkpoint**. The original project's fatal flaw was that PPO ran from `sft-mistral-v6-1-epoch` and REINFORCE from `sft-mistral-20k-v3-2epoch` — different checkpoints, different reward scales, no shared eval, so no comparison was ever possible.

---

## 4 · Decisions already made — do not relitigate

**Base model: `Qwen/Qwen2.5-3B-Instruct`.**
Ungated (removes the Secret Manager `hf-token` dependency entirely), instruct-tuned (SFT becomes light domain adaptation), runs bf16 unquantized on a 24GB L4 at ~6.2GB, and fast generation means more data per dollar. The "3B not 7B" choice is deliberate and defensible: *the bottleneck is rollout throughput, not model capacity.*

**Hardware: single L4** (`g2-standard-8`), not A100. ~4× cheaper per hour and sufficient at 3B.

**Preference pairs are turn-level, not trajectory-level.**
This is the key design decision. `airdialogue_sampled_5k.jsonl` already contains real customer turns, so pair generation does **not** need the user simulator:

1. Take a conversation prefix from an existing dialogue
2. Sample **k=4** agent replies from the SFT policy at temp 0.9, top_p 0.95
3. **One** judge call ranking all four against the flight KB
4. chosen = rank 1, rejected = rank 4 — keep only if the gap is real

One Gemini call per scenario instead of eleven. Generation drops from ~8 hours to under one. It is also more correct: DPO is a single-turn preference method, and applying it to whole trajectories asks it to do cross-turn credit assignment, which it does not do.

**The simulator is for evaluation only** — full multi-turn rollouts to measure end-to-end task success.

**Use TRL's `DPOTrainer`. Do not hand-roll it.** `trl` is already in `requirements.txt:84`. Hand-rolling PPO is how the original ended up with a KL that isn't a KL; repeating that on DPO would be the same mistake twice. The interview answer is better too: *"I hand-rolled PPO to understand the mechanics, then used the reference DPO implementation — the comparison is between the algorithms, not between my two implementations of them."*

**SFT is deliberately light: 10k rows × 1 epoch.** Not 20k × 2. An over-trained SFT policy produces near-identical near-gold samples, the judge ranks them the same, and there is no preference signal left for DPO to learn from. Competent but imperfect is the target.

---

## 5 · Budget and timing

$240 of GCP credits, **expiring 10 October**. This project needs about $10–15, leaving the rest for Rerouted.

| Stage | Time | Cost |
|---|---|---|
| SFT 10k × 1ep, L4 bf16 | ~1.5 hrs | ~$2 |
| Pair generation — 700 scenarios, k=4 | ~1 hr | ~$1.50 |
| DPO — ~450 pairs, 3 epochs | ~15 min | ~$0.50 |
| Eval — 4 arms × 50 held-out, full rollouts | ~1.5 hrs | ~$2 |
| Gemini (≈700 rank calls + ≈2,200 simulator) | — | ~$4 |
| **Total** | **~4.5 GPU-hrs** | **~$10** |

Rates move — confirm before committing. L4 ≈ $0.90/hr, A100 ≈ $3.70/hr as of writing.

**Day 1 (22 Sep)** — Write the Qwen SFT script, launch it (unattended). While it runs, write the pair generator. Then **dry-run pair generation on 20 scenarios and inspect the score spread before spending anything else.**

**Day 2 (23 Sep)** — Full pair generation (~1 hr). DPO training. Build the eval harness. Launch eval overnight.

**Day 3 (24 Sep)** — Plots, README with the four-arm table, publish. 25–29 Sep stays clear.

---

## 6 · Technical specifics that will bite

**Never hardcode token ids.** The old code has Mistral's `[28792, 16289, 28793]` at `new_train_1.py:229`. Qwen uses ChatML. Pass the response template to `DataCollatorForCompletionOnlyLM` as the string `<|im_start|>assistant\n` and let it tokenize. Keep the sanity check from `new_train_1.py:254` that aborts if the collator masks everything — it is a genuinely good defensive touch and it will catch a template mismatch loudly.

**Use `tokenizer.apply_chat_template` everywhere** — SFT and inference alike. A silent template mismatch between training and rollout is the classic killer in this kind of pipeline.

**LoRA:** r=16, α=32, dropout 0.05, targeting attention *and* MLP projections (`q,k,v,o,gate,up,down`). Qwen benefits from the MLP targets.

**DPO config:** β=0.1, lr=5e-6 (much lower than SFT's 1e-4), 3 epochs, `max_length=1024`, `max_prompt_length=768`, `ref_model=None` with PEFT so TRL gets reference logprobs by disabling the adapter.

**Pair filter:** discard any scenario where the judge's top and bottom ranks aren't meaningfully separated. Unseparated pairs are noise and will flatten the loss curve. Expect ~60% yield — hence 700 scenarios for ~450 pairs.

**The dry run is not optional.** Twenty scenarios, look at the actual rank distribution. If all four samples rank the same, nothing downstream works. Raise temperature or sharpen the rubric before committing the full run.

**Reuse from the old repo, don't rewrite:**
- The judge rubric at `baseline/rlaif_pipeline/constitution.py:294-386` — adapt it from scoring one transcript to ranking four replies
- `_safe_parse_score` at `baseline/rlaif_pipeline/judge.py:355-420` — a four-layer parser with a `rfind` trick to grab the top-level score rather than a nested one. It is good work; keep it.
- `airdialogue_sampled_5k.jsonl` — the only data source needed
- The held-out split must be scenarios **never seen** in SFT or pair generation

**Eval metrics:** task success rate (BOOKING_CONFIRMED with the correct flight), hallucination rate, mean judge score, turns to completion. Report per-arm.

---

## 7 · What "done" looks like

- [ ] Four-arm results table on a shared held-out set of 50 scenarios
- [ ] DPO loss curve and reward-margin plot
- [ ] Pair-generation stats: scenarios attempted, pairs kept, yield, rank-gap distribution
- [ ] README that states plainly: the original was a three-person course project; this is a rebuild; here is what changed and why
- [ ] A short post-mortem section covering the greedy-decoding diagnosis, the KL that wasn't a KL, and the discarded soft labels
- [ ] Secret scan of the **real git repo** (not this snapshot) before anything goes public
- [ ] `.DS_Store` files removed; `airdialogue_sampled_5k.jsonl` (55MB) replaced with a download script

---

## 8 · The resume fix — independent of all the above, do it first

Three claims need correcting before a single application goes out. If someone opens the repo — which is the entire point of publishing it — a mismatch is what they will remember.

**Current:** *"Evaluating PPO versus DPO using PyTorch and Hugging Face, focusing on stabilizing model training with KL-divergence penalties to prevent semantic drift."*

**Accurate, and stronger:** *"Implemented RLAIF for a flight-booking dialogue agent — Gemini as both user simulator and reward judge, no human preference data. Compared hand-rolled PPO against REINFORCE-with-baseline (d-RLAIF), with full-vocabulary KL against a frozen reference adapter."*

Also: the project was three people. Present it as team work with the owner's specific contribution named.

Once this rebuild lands, the resume can honestly claim the PPO-vs-DPO comparison — because it will exist.

---

## 9 · Things to avoid

- **Do not try to repair the old pipeline.** Wholesale duplication between `rlaif_pipeline/` and `Reinfore_baseline/` means fixes applied to one silently miss the other — which is exactly how the `simulator.py:155` regression arose. Build clean beside it.
- **Do not publish before the real git history is scanned.** GCP project id, project number and bucket name appear in ~15 source locations and on ~2,800 Cloud Logging entries in `vertex_ai_logs/`. Scrubbing source alone will not remove them.
- **Do not scale up before the dry run.** The whole budget is generatable in an afternoon; so is the whole budget wasted.
- **Do not let this run past 24 September.** Rerouted is the portfolio piece that matters. This is a three-day repair of a resume liability, not a project in its own right.

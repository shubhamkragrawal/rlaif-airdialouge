# RLAIF final results

Project: `portfolioproject-505823`  
Region: `us-central1`  
Completed: 9 October 2026

## Scope completed

- SFT with PEFT LoRA
- Preference generation with Gemini 2.5 Pro judging
- DPO with a PEFT adapter
- Identical held-out evaluation of base, SFT, and DPO models
- Bounded PPO from the SFT adapter, including a stable checkpoint-19 policy
- Adaptive PPO continuation to 50 cumulative accepted updates
- Identical held-out evaluation of both PPO policies

## Training evidence

| Stage | Result | Evidence |
|---|---:|---|
| SFT | 625 steps, loss 0.86527, 4,659.58 s | `gs://portfolioproject-505823-rlaif/runs/20261008-sft` |
| Preference generation | 386 pairs from 700 scenarios; 55.14% yield; all gates passed | `gs://portfolioproject-505823-rlaif/runs/20261008-pairs-full-pro` |
| DPO pilot | 20 steps, loss 0.68082, 384.84 s | `gs://portfolioproject-505823-rlaif/runs/20261008-dpo-pilot` |
| Full DPO | 72 steps, loss 0.62709, 1,379.40 s | `gs://portfolioproject-505823-rlaif/runs/20261008-dpo` |
| PPO checkpoint 19 | 19 accepted updates; gate-early-stopped before update 20 | `gs://portfolioproject-505823-rlaif/runs/20261009-ppo-v3/checkpoint-0019` |
| Adaptive PPO | 31 additional accepted updates; 50 cumulative; 5,994.14 s | `gs://portfolioproject-505823-rlaif/runs/20261009-ppo-adaptive-v4` |

## Held-out evaluation

Each arm used the same 50 scenarios and protocol v3.

| Arm | Task success | Wilson 95% interval | Mean judge score | Booking success | Explicit unknown-flight ID rate |
|---|---:|---:|---:|---:|---:|
| Base | 8/50 (16%) | 8.3%–28.5% | 2.94 | 4% | 0% |
| SFT | 13/50 (26%) | 15.9%–39.6% | 2.54 | 0% | 0% |
| DPO | 17/50 (34%) | 22.4%–47.8% | 2.58 | 0% | 0% |
| PPO checkpoint 19 | 19/50 (38%) | 25.9%–51.8% | 1.92 | — | 0% |
| Adaptive PPO update 50 | 17/50 (34%) | 22.4%–47.8% | 2.46 | — | 0% |

DPO improved task success by 18 percentage points over base and 8 points over SFT.
The intervals overlap, so this 50-scenario evaluation supports a promising directional
improvement rather than a definitive statistical claim. Mean judge score and booking
success did not improve, which should be treated as a limitation and investigated by
scenario type before presenting a broad quality claim.

Checkpoint-19 PPO had the highest observed task-success rate, but its interval overlaps
every trained arm. Continuing PPO to update 50 did not improve task success in this
evaluation: it scored 34% versus 38% at checkpoint 19. These are paired runs on only 50
scenarios, so the result does not establish that checkpoint 19 is generally better.

## Evaluation artifacts

- Base: `gs://portfolioproject-505823-rlaif/runs/20261009-eval-base-v3`
- SFT: `gs://portfolioproject-505823-rlaif/runs/20261009-eval-sft-v3`
- DPO: `gs://portfolioproject-505823-rlaif/runs/20261009-eval-dpo-v3`
- PPO checkpoint 19: `gs://portfolioproject-505823-rlaif/runs/20261009-eval-ppo-v3-r3`
- Adaptive PPO update 50: `gs://portfolioproject-505823-rlaif/runs/20261009-eval-ppo-adaptive-v3`

Every final arm completed all 50 scenarios without Gemini API errors. Protocol v3 fixed
the known-flight offer parser, and all five final arms recorded zero explicit unknown-flight
IDs under the evaluator's detector. This metric is narrower than general factual
hallucination and should not be described as such. The first adaptive-evaluation submission failed before any scenario or
Gemini call because of a duplicate CLI flag; the corrected rerun produced the result above,
and the failed attempt remains recorded in `STATUS.md` and `budget.json`.

The adaptive run used 124 rollout attempts, dropped 10 semantic rewards (8.06%, below
the 10% gate), and atomically rolled back one unsafe optimizer attempt before accepting
the same batch at half learning rate. Maximum accepted frozen-reference KL was 0.03719,
maximum update KL was 0.01505, and maximum likelihood ratio was 3.357, all within the
configured hard gates.

The conservative incremental PPO forecast was $146.94 against the authorized $150 cap.
This is a budget-control forecast; consult Cloud Billing for invoiced spend.

Historical training jobs identified the base model as `Qwen/Qwen2.5-3B-Instruct`
without pinning a Hugging Face repository commit. Evaluation artifacts record their
resolved revision, but exact from-scratch training reproduction requires recovering
the original base revision from cloud evidence or cache.

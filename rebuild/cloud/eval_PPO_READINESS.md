# PPO evaluation readiness — protocol v3

## Exact prepared job

`eval_ppo_full_a100.json` is a local configuration for a future completed PPO adapter.
It uses the recovery image `qwen:20261009-v12`, one `a2-highgpu-1g` worker with one
A10040GB, one replica, a100GB SSD and a four-hour execution timeout. **The image must
be built with the CLI addition that accepts `--arm ppo`; v10 rejects that arm name.**

- Adapter: `gs://portfolioproject-505823-rlaif/runs/20261009-ppo-v3/checkpoint-0019/adapter`
- Outputs: `gs://portfolioproject-505823-rlaif/runs/20261009-eval-ppo-v3-r3`
- Local outputs: `/tmp/eval-ppo`
- Vertex project: `portfolioproject-505823`; region: `us-central1`.
- Command: `python -m rlaif.evaluation.eval_qwen`, with `--arm ppo` and the adapter above.

All evaluation settings match completed base/SFT/DPO protocol v3: seed20260922,
50 scenarios, greedy decoding,160 policy output tokens,16 maximum agent turns,
Gemini2.5Flash customer plus secondary judge,512 API output tokens,1,200 attempted
API calls and2million reported-plus-reserved tokens. Customer consent validation and
bounded semantic repair remain `customer-v3-known-flight-offers`; no metric or
customer prompt changes were made for PPO. The simulator never receives acceptable
flight labels or an oracle matching-flight set.

The fixed manifest hash is
`eae0dda2106cab5b293cc9c2f5096bef796507052c763c1b80bf106e7883089a`:
25 bookable and25 no-flight scenarios from105 structurally answerable booking
heldouts. Task success, booking-only success, unknown-flight-ID hallucination,
turns and secondary judge scores retain their existing definitions and limitations.
This is not evaluation over all AirDialogue goals.

## Gates before submission

1. Complete the PPO pilot and full training on the same SFT checkpoint used by DPO:
   `gs://portfolioproject-505823-rlaif/runs/20261008-sft/sft/adapter`.
   Confirm the training metadata records its fingerprint and source/split provenance.
   Held-out scenarios must not enter training, reward tuning, or checkpoint selection.
2. Verify the final PPO export contains adapter weights, adapter config and completed
   training metadata; its Qwen base model must be `Qwen/Qwen2.5-3B-Instruct`. Use the
   completed export rather than a pilot or an untrained copy of SFT. Inspect finite
   losses, KL and rewards plus proof that trainable adapter weights changed.
3. Build the planned v11 image with the CLI change. Smoke-import its evaluation runner
   and verify `SIMULATOR_VERSION` remains `customer-v3-known-flight-offers`.
4. Add the evaluation reservation to the central budget forecast and confirm it remains
   within the owner's applicable soft/hard ceilings, with cancellation monitoring active.
   This file does not alter budget policy or authorize a cloud launch.
5. Confirm one GPU slot is available; the approved A100 quota permits at most two
   simultaneous single-GPU jobs across training and evaluation.
6. For final comparison, confirm all four output trees have50 completed scenarios,
   the same manifest, same v3 simulator/settings and same resolved Qwen revision.
   The existing comparison command checks manifests/completeness; inspect configuration
   and revision files too before interpreting paired bootstrap differences.

Existing v3 result prefixes are `20261009-eval-base-v3`, `20261009-eval-sft-v3`,
and `20261009-eval-dpo-v3` beneath the same bucket's `runs/` directory. Preserve
earlier failed v1/v2 artifacts as diagnostic evidence. Do not combine them with v3
results. PPO execution is not assumed complete by these local files.

## Time and cost estimate for evaluation only

Plan for roughly **1–2 hours**, with the four-hour job cap covering slow API responses.
The previous completed arms used283–333 API attempts and194k–276k reported tokens,
which is workload evidence; it does not guarantee PPO's dialogue lengths or latency.

Using a conservative planning allowance of **$5 per A100 worker-hour**, expect about
**$6–12** for one evaluation, including approximately$1 of API/storage allowance.
Reserve **$25** before launch: four GPU-hours at the planning rate plus at most$5
for the2million-token API cap. This is a forecast; verify the actual regional SKU
and project spending in the parent monitor. Training and the image build are excluded.

The API bound uses the higher Gemini2.5Flash standard text rate ($2.50/million output
tokens; input is$0.30/million), conservatively treating every capped token as output.
Source checked9October2026: [Google's Gemini pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing).
GPU planning allowance is intentionally above the listed$4.408062/hour A2 price;
check the applicable training SKU before approval:
[Google's Agent Platform pricing](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing).

## Verification completed locally

Three new tests verify one bounded worker, identical configuration settings across
all four arms, planned artifact paths, CLI acceptance of PPO with the exact existing
50-scenario manifest, and refusal of PPO without an adapter. All15 evaluation tests
pass. The CLI dry run loads no model, makes no Gemini call and uploads nothing.
No image was built and no cloud job was launched by this readiness task.

For an interrupted future job, preserve all arguments and add
`--resume-from gs://portfolioproject-505823-rlaif/runs/20261009-eval-ppo-v3-r3`.
The runner restores per-turn state and usage, and skips completed scenarios. Run
four-arm paired comparison only after every planned scenario is complete.

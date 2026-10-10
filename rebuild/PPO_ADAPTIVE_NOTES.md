# Adaptive PPO continuation

`ppo_qwen_adaptive.py` duplicates the bounded trainer. The existing `ppo_qwen.py` remains unchanged.

## Initialization and outcome

- Policy: `gs://portfolioproject-505823-rlaif/runs/20261009-ppo-v3/checkpoint-0019/adapter`.
- Reference: original SFT adapter, `gs://portfolioproject-505823-rlaif/runs/20261008-sft/sft/adapter`, permanently frozen.
- Restore parent value head and RNG; reset AdamW moments for the changed optimization schedule. Validate checkpoint state/metadata equality, model, source hash, split seed, SFT fingerprint and scenario ordering seed.
- Start accepted update counter and scenario cursor at19. Collect four fresh on-policy replies from each next ordered prefix. Target31 additional accepted updates, ending at cumulative update50. No parent rollout or judge response is used for a new batch.
- New run journals/caps track incremental resources. Parent run costs remain separately accounted by the overall budget monitor.

## Stability and rollback

Policy LR1.5e-6; value LR2e-5; one PPO epoch; normalized token advantages clipped at±5. Original hard gates remain update KL0.05, sampled ratio bound4, and frozen-reference KL1. A stricter acceptance threshold rejects update KL above0.04 or ratio outside[1/3.4,3.4]. This provides headroom without increasing any hard limit.

Before each optimizer attempt, copy trainable policy weights, value head, optimizer moments, and RNG to a rollback snapshot. Rejections restore the whole snapshot and retry the same immutable old-policy rollout batch, halving both learning rates. At most four attempts per accepted-update index; policy LR floor1e-7. Nonfinite failures abort after restoration. Durable attempt reservations are uploaded before optimization, and survive checkpoint resume. Accepted lower rates persist for subsequent updates.

Fresh batches are collected after acceptance. A rejected optimization attempt makes no additional judge calls. Checkpoints retain policy, value head, optimizer, RNG and committed cursor. A new script/source/protocol/config identity prevents accidental resume of the original trainer. Exhausted caps or retries preserve the last accepted checkpoint and fail; a partial result is not marked completed.

## Configuration and verification

Generate the separate Vertex config with the built image tag:

```sh
python RLAIF/rebuild/cloud/prepare_ppo_adaptive.py \
  --tag YYYYMMDD-vN --date YYYYMMDD --suffix closeout-v1 \
  --output-dir RLAIF/rebuild/cloud/prepared-closeout
```

This writes `cloud/ppo_adaptive.json`; no cloud job is launched. It specifies one
A100 40GB, 50 cumulative updates, 50 total ordered prefixes, 124 incremental
rollout attempts, 300 judge attempts, 800,000 conservatively reserved/reported
tokens, a 6,600-second training cap, and a 7,200-second job timeout. These caps
fit the measured continuation workload and the remaining project budget.

Before building the image, include imports of `ppo_adaptive` and `ppo_qwen_adaptive`, and run `test_ppo_adaptive` alongside the existing PPO tests. The CPU suite checks measured update gates, atomic rollback including Adam moments/RNG, finite retry/resume caps, parent provenance, and the generated reference/continuation configuration. GPU throughput and full continuation outcome require the authorized cloud run.

For the post-restructure closeout image, use the immutable preparation command:

```sh
python RLAIF/rebuild/cloud/prepare_closeout.py \
  --tag YYYYMMDD-v14 --date YYYYMMDD --suffix closeout-v1 \
  --model-revision HUGGINGFACE_COMMIT \
  --output-dir RLAIF/rebuild/cloud/closeout
```

The output directory must be new. This writes a fresh source manifest and configs
without modifying `qwen:20261009-v13` or historical run prefixes.

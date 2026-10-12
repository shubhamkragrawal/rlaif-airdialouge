# RLAIF project overview

This repository contains a completed local flight-booking dialogue application
and a reproducible RLAIF training/evaluation package. The supported interactive
runtime is under `local_app/`; the historical/rebuild implementation is under
`rebuild/rlaif`. Older experiments remain under `baseline/` as audit material.

## Layout

```text
RLAIF/
├── local_app/            # local controller, API/UI, runtime, evaluation
├── artifacts/            # versioned matrix and release evidence
├── test_artifacts/       # recorded browser regression evidence
├── rebuild/
│   ├── rlaif/
│   │   ├── shared/       # corpus loading, immutable splits, prompts
│   │   ├── data/         # dataset preparation entry points
│   │   ├── preferences/  # Gemini preference generation and validation
│   │   ├── rewards/      # grounded PPO reward and bounded judge retries
│   │   ├── training/     # PEFT SFT, DPO, PPO, adaptive PPO
│   │   ├── evaluation/   # common metrics and held-out runner
│   │   └── tests/        # CPU tests and cloud-config tests
│   ├── cloud/            # Dockerfile, job configs, build and artifact helpers
│   └── artifacts/        # frozen examples and split manifest
├── baseline/             # historical implementation and audit evidence
├── vertex_ai_logs/       # historical exported logs
└── PROJECT_OVERVIEW.md
```

## Local release status

The local release is complete and does not require GCP or an external model API.
It provides a localhost-only browser/API service, CLI, deterministic simulator,
optional local Qwen/PEFT inference, four-adapter evaluation, strict output and
transaction validation, and a hash-pinned release gate.

The final integrated matrix evaluated 50 scenarios for each of `sft`, `dpo`,
`ppo_checkpoint_19`, and `ppo_adaptive_50` on Apple MPS. All four arms achieved
100% task completion, 100% validation of displayed responses, and zero displayed
repetition. Candidate acceptance and safe fallback rates are reported separately
in [RLAIF_2_RELEASE_REPORT.md](RLAIF_2_RELEASE_REPORT.md).

The final verification passed 340 local tests and 85 rebuild tests. The release
gate result is `artifacts/release/gate-v1/release-gate-v1.json`; operational
commands are in [OPERATIONS.md](OPERATIONS.md).

Cloud JSON files preserve the resource limits, arguments, historical image tags,
and artifact prefixes. They use one GPU worker and finite timeouts. Their command
fields were normalized to the reorganized `rlaif.*` modules, so the dated files
are archival parameter records rather than exact submitted manifests and are not
runnable with their historical flat-layout images. Generate a new config and
build `rebuild/cloud/Dockerfile` under a fresh immutable image tag and artifact
prefix. Do not overwrite an experiment image tag or artifact prefix.

`rebuild/artifacts/dataset_manifest.json` pins SHA256 values for the source
corpus, committed split file, and SFT examples. Runtime loaders validate these
before using the data. Historical training runs did not pin a Hugging Face
repository revision; evaluation recorded the resolved revision, but exact
from-scratch training reproduction remains limited by that missing historical
pin. New closeout preparation requires an immutable model revision.

## Local setup and verification

The cloud image installs `rebuild/cloud/requirements.txt`. For CPU checks from
the repository root:

```sh
PYTHONPATH=RLAIF/rebuild python3 -m unittest discover \
  -s RLAIF/rebuild/rlaif/tests -p 'test*.py'
python3 -m json.tool RLAIF/rebuild/cloud/ppo_adaptive.json >/dev/null
```

Historical GPU training and Vertex evaluation were defined through the bounded
configs in `rebuild/cloud`. Use `prepare.py` for an isolated Docker context and
`prepare_ppo_adaptive.py` for the adaptive PPO config; both write only to a new
output directory. Use `prepare_closeout.py` for a fresh post-restructure image
context/config with a pinned model revision. These scripts only generate files
and do not submit jobs. These cloud workflows are not required to operate or
verify the completed local release.

`rebuild/cloud/build_context.json` describes the most recently prepared local
context. Published-image provenance is identified by the build IDs and image
digests in `rebuild/cloud/STATUS.md` and `rebuild/cloud/budget.json`.

## Completed measurements

The required base/SFT/DPO held-out comparison used the same 50-scenario v3
manifest:

| Arm | Task success | Mean judge score | Explicit unknown-flight ID rate |
| --- | ---: | ---: | ---: |
| Base | 8/50 (16%) | 2.94 | 0% |
| SFT | 13/50 (26%) | 2.54 | 0% |
| DPO | 17/50 (34%) | 2.58 | 0% |

The result is directional evidence only: the sample is small, the confidence
intervals overlap, and judge score/booking success did not improve uniformly.

PPO was evaluated separately. The stable checkpoint-19 policy scored 19/50
(38%) with mean judge score 1.92. Adaptive PPO reached 50 cumulative accepted
updates from checkpoint 19 and scored 17/50 (34%) with mean judge score 2.46.
Both PPO evaluations recorded zero explicit unknown-flight IDs. The adaptive
run had one rolled-back update attempt and retained the original hard KL/ratio
gates; all evidence is preserved in GCS.

## Artifact provenance

The private artifact bucket is
`gs://portfolioproject-505823-rlaif`. Final prefixes are:

- SFT: `runs/20261008-sft`
- Preference pairs: `runs/20261008-pairs-full-pro`
- DPO: `runs/20261008-dpo`
- Base/SFT/DPO evaluation: `runs/20261009-eval-{base,sft,dpo}-v3`
- PPO checkpoint 19 and evaluation: `runs/20261009-ppo-v3`,
  `runs/20261009-eval-ppo-v3-r3`
- Adaptive PPO and evaluation: `runs/20261009-ppo-adaptive-v4`,
  `runs/20261009-eval-ppo-adaptive-v3`

Failed and superseded runs remain in their original prefixes and are recorded
in `rebuild/cloud/budget.json` and `rebuild/cloud/STATUS.md`. The PPO
incremental forecast was $146.94 against the authorized $150 cap; this is a
conservative forecast, not a replacement for the Cloud Billing report.

## Limitations

The held-out comparisons are not large enough for a definitive claim. Gemini
judge and simulator responses are part of the measurement protocol, so API
availability and judge behavior affect results. PPO was deliberately bounded
by KL, ratio, reward-drop, timeout, and budget gates. Historical `baseline/`
code is retained for provenance and is not the supported entry point.
The reported unknown-flight metric detects explicit flight-ID mentions and is
not a general factual-hallucination measurement.
The historical training configs name the Hugging Face base model but do not pin
its repository commit. Evaluation artifacts record the revision resolved at
evaluation time, but an exact from-scratch reconstruction of the training base
revision is not guaranteed unless that revision is recovered from the original
model cache or cloud evidence.

The imported policy-only evidence is summary-level because the original raw
per-scenario policy outputs are not present locally. The integrated local matrix
contains scenario- and turn-level evidence, but it remains a simulator evaluation
and does not establish performance against live airline systems.

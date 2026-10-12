# RLAIF flight-booking agent

This repository contains a completed local RLAIF flight-dialogue application,
its evidence-backed release artifacts, and preserved historical training
experiments. The supported interactive application is under `local_app/`; the
reproducible training/evaluation package is under `rebuild/rlaif/`.

Start here:

- [RLAIF_2_RELEASE_REPORT.md](RLAIF_2_RELEASE_REPORT.md) for final local results.
- [OPERATIONS.md](OPERATIONS.md) for startup, testing, and release verification.
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for architecture and historical results.
- [LOCAL_CONVERSATION_SYSTEM_SPEC.md](LOCAL_CONVERSATION_SYSTEM_SPEC.md) for the
  implemented local-system contract.

The historical/rebuild package contains:

- `shared/` loads immutable data splits and prompts.
- `data/` prepares training data.
- `preferences/` generates and validates AI preference pairs.
- `rewards/` contains grounded PPO rewards and bounded judge retries.
- `training/` contains PEFT SFT, DPO, PPO, and adaptive PPO.
- `evaluation/` contains the common held-out evaluation runner.
- `tests/` contains CPU and configuration tests.

Cloud job JSON files under `rebuild/cloud/` preserve historical experiment arguments,
resource limits, historical image tags, and GCS prefixes. Their command fields
were normalized to the reorganized `rlaif.*` module paths, so they are archival
parameter records rather than byte-for-byte submission manifests and cannot run
against the historical flat-layout images. No cloud account is required to run,
test, evaluate, or verify the completed local release. Historical code and logs
under `baseline/` and `vertex_ai_logs/` are retained for provenance and are not
supported entry points.

The checked-in `rebuild/artifacts/dataset_manifest.json` records SHA256 hashes for
the corpus, split file, and SFT examples. Training and evaluation refuse changed
corpus/split inputs. Historical runs did not pin a Hugging Face model commit;
new closeout preparation requires `--model-revision` and records the resolved
revision when the model loads. See [REPRODUCIBILITY.md](rebuild/cloud/REPRODUCIBILITY.md).

## Quick start

From `RLAIF/`, create the project-local environment and run the deterministic
application:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
.venv/bin/python -m local_app.server
```

Open `http://127.0.0.1:8000/`. The server is localhost-only and every booking is
simulated. Run all supported regression checks with:

```sh
./scripts/full-local-regression
```

The final closeout run passed 340 local tests and 85 rebuild tests. Browser
behavior tests require Node.js 18 or newer; no npm dependencies are needed.

## Historical cloud jobs

These tools are retained for historical reproducibility and are not needed by
the local release. Use `rebuild/cloud/prepare.py` and
`rebuild/cloud/prepare_ppo_adaptive.py` to generate isolated configs in a new
output directory. These scripts only generate files; they do not submit jobs.
For final post-restructure metadata and a fresh image context, use
`rebuild/cloud/prepare_closeout.py` with a new v14-or-later tag, suffix, and
pinned model revision. Submit only bounded JSON configs after checking the
budget ledger in `rebuild/cloud/budget.json`.

Historical measured results and GCS locations are documented in
`rebuild/cloud/FINAL_RESULTS.md` and `PROJECT_OVERVIEW.md`.

## Results at a glance

All five final policies used the same 50 held-out scenarios. The observed task
success rates are directional because their 95% intervals overlap. PPO 19 is
the last accepted checkpoint before the original trainer's update gate stopped
the run; PPO 50 is the completed adaptive continuation.

![Held-out task success with Wilson confidence intervals](figures/evaluation_task_success.svg)

The judge score provides a second view of the same evaluations; it does not
rank the policies in the same order as task success.

![Held-out judge scores](figures/evaluation_judge_score.svg)

The curves below come from the rebuilt runs' saved training logs. SFT and DPO
optimize different objectives, so their loss values should not be compared to
each other. PPO reward is an on-policy training signal, not held-out task success.

![SFT and DPO training loss curves](figures/training_losses.svg)

The preference quality gates retained 386 of 700 scenarios as training pairs.
DPO's in-training pair accuracy is shown separately from held-out task success.

![Preference generation outcomes](figures/preference_generation.svg)

![DPO training pair accuracy](figures/dpo_pair_accuracy.svg)

![Accepted PPO update diagnostics](figures/ppo_diagnostics.svg)

See [figure provenance and regeneration](figures/README.md) for source files,
checks, and limitations.

## Local conversation runtime

The local runtime is under `local_app/`. It uses the
checked-in AirDialogue scenarios as a deterministic flight knowledge base and
never contacts an airline, payment provider, model endpoint, or cloud service.
The controller owns all tool actions and state transitions. A selected Qwen/PEFT
adapter may phrase grounded responses, but guardrail failures fall back to the
deterministic controller response.

After the environment has been created, use:

```sh
.venv/bin/python -m unittest discover -s local_app/tests -p 'test*.py'
.venv/bin/python -m local_app.server             # deterministic, http://127.0.0.1:8000
.venv/bin/python -m local_app.server --model     # lazy MPS model inference
.venv/bin/python -m local_app.cli --adapter dpo --model
```

Model mode is strictly local-only: it does not download a base model or adapter.
It requires the pinned Qwen revision to already exist in the Hugging Face cache
and adapter files under `local_models/{sft,dpo,ppo_checkpoint_19,ppo_adaptive_50}`.
If those files are unavailable, deterministic mode remains fully usable. The
legacy `local_app.download_adapters` helper refers to the historical private GCS
bucket and is not part of the supported closeout workflow.

The browser regressions BR006 and BR007 also require Node.js 18 or newer on
`PATH`. Set `RLAIF_NODE` to the Node executable if it is not on `PATH`. They
execute the shipped browser script with Node's built-in `node:vm` and a
behavioral DOM harness; no npm package is required. These tests fail with a
prerequisite error if Node.js is unavailable.

The API exposes `GET /health`, `GET /adapters`, `POST /sessions`,
`GET /sessions/{id}`, `POST /sessions/{id}/messages`, and
`POST /sessions/{id}/reset`. `GET /` serves a small browser UI. It binds to
localhost by default. Booking is simulator-only and requires a flight returned
by the current local search. Redacted audit events go to `.local/audit.jsonl`
by default; set `RLAIF_AUDIT_LOG` to change the path.

The local ML profile pins the final evaluation's resolved Qwen revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Adapter files live under ignored
`local_models/{sft,dpo,ppo_checkpoint_19,ppo_adaptive_50}`. The base model cache
is `local_models/huggingface`; startup uses local files only. Only one policy is
kept in memory at a time for the validated 16 GB Apple Silicon profile.

Run a deterministic controller-only evaluation with:

```sh
.venv/bin/python -m local_app.evaluation --count 50 --adapter sft \
  --output outputs/local-regression-sft.json
```

This report is a controller/simulator regression, not a replacement for the
model-policy held-out evaluations in `rebuild/cloud/FINAL_RESULTS.md`.

## Final local release

The final integrated release ran the same 50 scenarios across all four local
adapters on Apple MPS. Every arm completed all scenarios with 100% task
completion, 100% validation of user-visible responses, and zero repetitive
user-visible output. Rejected model candidates were recorded and replaced by
the authoritative deterministic controller response; fallback rate therefore
remains a model-quality signal rather than a hidden safety failure.

Key artifacts:

- `artifacts/matrix/releases/v1/release-v1.json` — complete 50-by-4 matrix.
- `artifacts/evaluation/policy-only-v1.json` — historical summary-level policy
  evidence, explicitly incomplete because raw scenario records are absent.
- `artifacts/release/evidence-v1-20261012T005310274515Z/evidence-v1.json` — six
  passing local release checks.
- `artifacts/release/gate-v1/release-gate-v1.json` — final gate result (`pass`).

See [RLAIF_2_RELEASE_REPORT.md](RLAIF_2_RELEASE_REPORT.md) for per-adapter
metrics, provenance, and limitations.

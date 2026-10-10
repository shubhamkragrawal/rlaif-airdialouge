# RLAIF flight-booking agent

This repository contains the supported RLAIF implementation and preserved
historical experiments. Start with [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
for architecture, measured results, artifact provenance, and limitations.

The supported package is `rebuild/rlaif`:

- `shared/` loads immutable data splits and prompts.
- `data/` prepares training data.
- `preferences/` generates and validates AI preference pairs.
- `rewards/` contains grounded PPO rewards and bounded judge retries.
- `training/` contains PEFT SFT, DPO, PPO, and adaptive PPO.
- `evaluation/` contains the common held-out evaluation runner.
- `tests/` contains CPU and configuration tests.

Cloud job JSON files under `rebuild/cloud/` preserve the experiment arguments,
resource limits, historical image tags, and GCS prefixes. Their command fields
were normalized to the reorganized `rlaif.*` module paths, so they are archival
parameter records rather than byte-for-byte submission manifests and cannot run
against the historical flat-layout images. Generate a fresh config, image tag,
and artifact prefix with the preparation tools below. Historical code and logs
under `baseline/` and `vertex_ai_logs/` are retained for provenance and are not
supported entry points.

The checked-in `rebuild/artifacts/dataset_manifest.json` records SHA256 hashes for
the corpus, split file, and SFT examples. Training and evaluation refuse changed
corpus/split inputs. Historical runs did not pin a Hugging Face model commit;
new closeout preparation requires `--model-revision` and records the resolved
revision when the model loads. See [REPRODUCIBILITY.md](rebuild/cloud/REPRODUCIBILITY.md).

## Local checks

From the repository root:

```sh
PYTHONPATH=RLAIF/rebuild python3 -m unittest discover \
  -s RLAIF/rebuild/rlaif/tests -p 'test*.py'
```

The cloud image runs the same suite with its PyTorch environment. The local
environment must have PyTorch installed to run the PPO tensor tests.

## Cloud jobs

Use `rebuild/cloud/prepare.py` and
`rebuild/cloud/prepare_ppo_adaptive.py` to generate isolated configs in a new
output directory. These scripts only generate files; they do not submit jobs.
For final post-restructure metadata and a fresh image context, use
`rebuild/cloud/prepare_closeout.py` with a new v14-or-later tag, suffix, and
pinned model revision. Submit only bounded JSON configs after checking the
budget ledger in `rebuild/cloud/budget.json`.

Final measured results and GCS locations are documented in
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

From `RLAIF/`:

```sh
pyenv shell 3.12.12               # or use any Python 3.12 interpreter
python -m venv .venv
.venv/bin/pip install -r requirements-local.txt
.venv/bin/python -m local_app.download_adapters  # requires authenticated gsutil
.venv/bin/python -m unittest discover -s local_app/tests -p 'test*.py'
.venv/bin/python -m local_app.server             # deterministic, http://127.0.0.1:8000
.venv/bin/python -m local_app.server --model     # lazy MPS model inference
.venv/bin/python -m local_app.cli --adapter dpo --model
```

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
`local_models/{sft,dpo,ppo_checkpoint_19,ppo_adaptive_50}`. The base model is
cached under `local_models/huggingface` on first `--model` use. Only one policy
is kept in memory at a time for the 16 GB Apple Silicon profile.

Run the deterministic 50-scenario local regression suite with:

```sh
.venv/bin/python -m local_app.evaluation --count 50 --adapter sft \
  --output outputs/local-regression-sft.json
```

This report is a controller/simulator regression, not a replacement for the
model-policy held-out evaluations in `rebuild/cloud/FINAL_RESULTS.md`.

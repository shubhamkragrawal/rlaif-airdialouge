# Bounded PPO implementation

`ppo_qwen.py` starts from the same completed Qwen SFT adapter as DPO. It trains on
decision-turn prefixes in the committed `pair_gen` split; heldout and SFT dialogues
are excluded. Each sampled reply is a finite episode ending at EOS or the explicit
128-token horizon. Shared multi-turn evaluation is still required afterward.

## Objective and evidence

- Four fresh replies per prefix, sampled from the raw policy distribution:
  temperature 1, top-p 1, top-k 0. Sampling warpers would make the PPO likelihood
  ratio disagree with the behavior policy, so the DPO sampling settings are not used.
- Old action log probabilities and old values are detached and fixed through both
  PPO epochs. Only response tokens, including sampled EOS, enter the objective.
- An explicit frozen copy of the original SFT policy supplies reference likelihoods.
  Token reward is `-0.1 * (log pi_old - log pi_SFT)`, with the grounded judge reward
  added at the last response token. This is a sampled reference KL estimate, rather
  than entropy or a penalty that ignores the reference model. Signed KL and the
  nonnegative k3 estimator are both logged; individual signed estimates may be negative.
- GAE uses gamma 1 and lambda .95. Advantages are normalized across valid response
  tokens. Policy ratios and value predictions are clipped; gradients are bounded.
- The separate value head learns from policy hidden states. Exported `adapter/`
  is directly usable by the existing evaluation runner; `value_head.pt` is retained
  for training provenance and resume.
- Invalid judge results are dropped. Local deterministic grounding penalties and
  valid judge scores are labeled explicitly by the reward adapter; no guessed reward
  is supplied for a parsing failure.

After every optimizer step the runner measures old/new approximate KL, mean/min/max
ratio, policy/value clip fractions, policy/value losses and pre-clipping gradient norm.
Nonfinite tensors, an initial old-policy ratio mismatch, excessive update/reference
KL, extreme ratios or a reward drop rate reaching 10% abort the run. Failed updates
are not published as complete checkpoints or final adapters.

## Checkpoints and caps

Each accepted update atomically saves the adapter, tokenizer, value head, optimizer,
Python/torch/CUDA RNG and scenario cursor. A checkpoint at update zero preserves the
starting policy. `rollouts-NNNN.pt` preserves the sampled actions and immutable old
likelihood/value/reference snapshots. All saved identities contain corpus/SFT hashes,
split seed, source hashes and exact hyperparameters.
Local checkpoint retention is the initial SFT-start snapshot plus the latest two
accepted updates. Cloud resume fetches only the latest committed checkpoint and
cumulative reward/compute journals, avoiding downloads of historical optimizer files.

Resume with the same arguments and `--resume-from gs://.../RUN`. It restores the last
fully committed update. Uncommitted batches may replay; the reward adapter binds each
cached result to exact response/policy/protocol identity. Durable judge reservations,
rollout attempt counts and elapsed-time ledgers remain cumulative. The configured
compute/API caps cannot be raised during resume. Cloud timeouts provide an additional
external bound. A fresh full run must start from SFT, not from the pilot adapter.

## Preparation and execution

Generate finite pilot/full configurations:

```bash
python RLAIF/rebuild/cloud/prepare_ppo.py \
  --tag YYYYMMDD-v14 --date YYYYMMDD --full-run-suffix closeout-v1 \
  --output-dir /tmp/rlaif-ppo-prepared
```

The pilot allows 2 updates, 2 prefixes, 8 sampled replies, at most 24 Gemini calls,
300,000 reported/reserved tokens, and a one-hour job timeout. The full configuration
allows 50 updates, 50 prefixes, 200 sampled replies, 400 calls, 4 million tokens,
and a 9,000-second runner cap inside a 9,600-second job timeout. Both use one A100
40GB. These are resource caps, not
spending estimates; the project's existing budget gate must approve any job.

Build a new image first. `cloud/prepare.py --tag YYYYMMDD-v14 --date YYYYMMDD`
already includes
all rebuild Python files in its isolated context; the Dockerfile now imports the PPO
modules and runs their CPU numerical/integration tests. The generated image tag
does not exist until that build succeeds. Do not launch the full configuration until
the pilot passes its runtime, reward-quality, checkpoint and budget gates.

The standalone launch command for the approved pilot is:

```bash
gcloud ai custom-jobs create --project=portfolioproject-505823 \
  --region=us-central1 --display-name=rlaif-ppo-pilot-a100-20261009 \
  --config=RLAIF/rebuild/cloud/ppo_pilot.json
```

Run the numerical tests in an environment with the pinned dependencies:

```bash
PYTHONPATH=RLAIF/rebuild python -m unittest test_ppo_core
```

The tests include categorical reference-KL identities, token masking and EOS rewards,
GAE termination, policy/value clipping, detached old snapshots, nonfinite aborts,
padding gradient isolation, actual tiny randomly initialized Qwen + LoRA generation
and updates, unchanged frozen reference weights, and optimizer/value/RNG checkpoint
roundtrips. They require no pretrained download, Gemini call or GPU.

## Limits

This is an explicit audited PPO implementation rather than TRL's reward-model PPO
trainer. Its single-turn grounded reward protocol is separately fingerprinted from
the DPO ranking protocol. A PPO/DPO comparison therefore differs in reward-interface
and on-policy sampling as well as the optimization algorithm; report that plainly.
CPU tests establish the math and pinned architecture integration. A100 memory,
throughput and live reward consistency must still be measured in the pilot.

References: [PPO paper](https://arxiv.org/abs/1707.06347),
[pinned TRL PPO metrics and implementation references](https://huggingface.co/docs/trl/v0.12.2/ppo_trainer).

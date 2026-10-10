# Evaluation readiness — local configuration audit

The three `eval_{base,sft,dpo}_full_a100.json` configurations use image
`us-central1-docker.pkg.dev/portfolioproject-505823/rlaif/qwen:20261008-v7`.
The current eval runner, deterministic metric code and dependencies exactly match
the copied v7 build context. The prior two-scenario base GPU smoke succeeded; it
established runtime/upload functionality, not final performance.

Each configuration requests one A100 40GB (`a2-highgpu-1g`) and one replica with
a four-hour execution timeout. The project has two approved A100s, so at most two
of these jobs may run concurrently. Configurations set the intended Vertex project
and region explicitly. No API key is supplied, so the runner uses Vertex ADC billing.

| Arm | Adapter input | Result prefix |
| --- | --- | --- |
| Base | none | `gs://portfolioproject-505823-rlaif/runs/20261008-eval-base` |
| SFT | `gs://portfolioproject-505823-rlaif/runs/20261008-sft/sft/adapter` | `gs://portfolioproject-505823-rlaif/runs/20261008-eval-sft` |
| DPO | `gs://portfolioproject-505823-rlaif/runs/20261008-dpo/adapter` | `gs://portfolioproject-505823-rlaif/runs/20261008-eval-dpo` |

The SFT URI follows the completed SFT wrapper's nested run layout. The DPO URI follows
the DPO exporter's direct run-directory upload; **DPO evaluation must wait for that
completed export**, including adapter weights and config. A DPO pilot adapter is not
the full-training result. No remote adapter-existence check or cloud job launch was
performed during this local configuration audit.

All arms use the same 50 held-out scenarios, seed 20260922, 16-agent-turn limit,
160 policy output tokens, Gemini 2.5 Flash customer/secondary judge, greedy policy
decoding, 512-token Gemini output limit, 1,200 API-attempt cap and 2,000,000-token cap
per arm. The manifest contains 25 bookable and 25 no-flight scenarios, selected from
105 structurally answerable booking heldouts. This restricted scope and the separate
booking-success denominator must accompany the final table.

All three configuration argument lists passed local CLI dry runs without API calls,
GPU loads or GCS uploads. They produced the same manifest SHA256:

`eae0dda2106cab5b293cc9c2f5096bef796507052c763c1b80bf106e7883089a`.

Nine evaluation tests pass. Config JSONs include no resume source for a first run.
If a job is interrupted, add `--resume-from` with that arm's result prefix, preserving
all other arguments and the same model revision/adapter. The runner restores and
validates saved state, skips completed scenarios, and retains cumulative API usage.

Before submission, the parent monitor must reserve up to 12 A100 GPU-hours for all
three four-hour caps, plus bounded API usage, against the overall $150 soft/$200 hard
policy. The per-arm API caps are resource bounds, not a project-wide dollar guarantee.
Confirm the recorded resolved model revisions agree across arms before final comparison.
Once all three outputs contain all50 complete scenarios, download them and run
`eval_qwen.py --compare <base-dir> <sft-dir> <dpo-dir>` for paired differences.

# PPO extension plan

Authorized on 9 October 2026 with 195 credits available.

## Budget

- Working target: $100 incremental
- Absolute PPO cap: $150 incremental
- Minimum remaining credits: $45
- One GPU per job, finite timeouts, durable checkpoints

## Gates

1. Local PPO math, provenance, parser, and resume tests pass.
2. Container dependency check and image build pass.
3. Twenty-episode smoke run on one A100, maximum two hours.
4. Inspect finite losses, value predictions, true reference KL, update ratios,
   clip fraction, reward parsing, API usage, and uploaded artifacts.
5. Fifty-episode pilot only if the smoke gate passes, maximum five hours.
6. Full run only if the pilot is stable and the conservative cumulative forecast
   remains below $150.
7. Evaluate the final PPO adapter on the same held-out 50 scenarios and protocol v3
   used for base, SFT, and DPO.

Any failed gate stops expansion. No fabricated reward may replace a failed judge parse.

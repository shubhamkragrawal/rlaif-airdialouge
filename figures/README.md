# Figure provenance

These SVGs are generated from the completed rebuilt runs, not the historical
prototype logs under `vertex_ai_logs/` or `baseline/plots/`.

| Figure | Evidence used | Interpretation limit |
|---|---|---|
| `training_losses.svg` | SFT and DPO `trainer_state.json` log histories | Different objectives and vertical scales; losses are not comparable across stages. |
| `preference_generation.svg` | Full preference run `pair_stats.json` | The five outcome counts partition 700 scenarios; this is a data-quality view, not policy performance. |
| `dpo_pair_accuracy.svg` | Full DPO `trainer_state.json` log history | In-training chosen-vs-rejected batch accuracy is not held-out task success. |
| `ppo_diagnostics.svg` | Rebuilt PPO v3 and adaptive PPO `training_log.jsonl` accepted updates | On-policy rewards are noisy and are not held-out evaluation scores. The rejected v3 update 20 is not plotted as an accepted update. |
| `evaluation_task_success.svg` | Five final protocol-v3 `summary.json` files | Each arm has 50 scenarios; Wilson 95% intervals overlap. |
| `evaluation_judge_score.svg` | The same five final `summary.json` files | Mean score has no displayed uncertainty interval and ranks arms differently from task success. |

`data/metrics.json` stores only the numeric fields plotted and SHA256 hashes of
the source files. Raw prompts, scenario transcripts, cloud account identifiers,
and private artifact addresses stay out of the figure data. The original evidence
is preserved in the run prefixes listed in the internal cloud status record.

To regenerate the SVGs from the committed numeric extract:

```sh
python3 -m pip install reportlab
python3 RLAIF/figures/build_charts.py
```

To refresh the extract from the private run artifacts, download these files into
one local directory, then pass that directory to `--extract-source-dir`:

```text
sft/trainer_state.json             <- 20261008-sft/sft/trainer_state.json
dpo/trainer_state.json             <- 20261008-dpo/trainer_state.json
ppo19/training_log.jsonl           <- 20261009-ppo-v3/training_log.jsonl
ppo50/training_log.jsonl           <- 20261009-ppo-adaptive-v4/training_log.jsonl
prefs/pair_stats.json              <- 20261008-pairs-full-pro/pair_stats.json
eval/{base,sft,dpo,ppo19,ppo50}.json <- final protocol-v3 summary.json files
```

Then run `python3 RLAIF/figures/build_charts.py --extract-source-dir /path/to/local/files`.
The extractor checks the expected 19+31 PPO update sequence and complete
non-provisional 50-scenario evaluation summaries, reconciles the preference
outcome counts, and requires its quality gate to have passed before replacing
the public numeric extract.

The bounded SFT and DPO pilots are recorded in the experiment status, but these
figures show the completed full runs. Historical university prototype plots are
preserved separately under `baseline/rlaif_pipeline/plots/`; they use different
code and protocols and should not be overlaid with the rebuilt runs.

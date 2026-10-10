# RLAIF Git commit strategy

_Planning document. No files have been staged, committed, or pushed by this plan._

## Recommendation on the old files

**Keep a curated historical snapshot, not the entire old tree as an undifferentiated first commit.** The original university implementation is useful provenance: it shows the starting SFT, PPO, and REINFORCE designs and explains why the rebuild was necessary. Put its source, Dockerfiles, and small configuration files in one explicitly labeled `baseline/` commit. Mark it as historical and unsupported.

Keep large generated logs, JSONL outputs, offline datasets, plots, and the course PDF out of the first public code commit. They can be added later as a separately reviewed evidence release if redistribution rights, privacy, and value justify it. Preserve their local copies and hashes in the meantime. The rebuilt `rebuild/rlaif/` package is the supported implementation.

## What the inventory shows

- `RLAIF/` is currently **not** a Git repository. The neighboring `rerouted/` directory has its own repository; avoid accidentally initializing or staging the whole portfolio workspace.
- `RLAIF/` is about 124 MB. The sampled source corpus is about 55 MB, and the derived SFT examples are about 46 MB. Those two files account for most of the size. The historical offline JSONL data adds about 13 MB; exported Vertex logs add about 2.9 MB.
- There is no RLAIF `.gitignore` or license file in the current tree. Dataset redistribution terms have not been established from repository files.
- Historical dated cloud JSONs retain old image tags and artifact prefixes, but their command fields were normalized during the restructure. They are archival parameter records, not exact submitted manifests or runnable templates for those old images.
- Internal cloud project/bucket identifiers occur in code, tests, configs, status, and results documents. A lightweight filename-only scan did not establish that credentials are present; a full pre-publication secret and privacy review is still required.

## Repository boundary and publication decision

Use `RLAIF/` as the repository root if this remote is dedicated to the RLAIF project. If the new remote is for the whole portfolio, place RLAIF under its own top-level directory and keep commits scoped to `RLAIF/`; do not mix in `rerouted/` work. Decide whether the remote is public before selecting files. This strategy assumes **public** because the architecture report is intended to be public.

Before the first public commit:

1. Confirm the AirDialogue-derived corpus and course materials may be redistributed. No license file in this tree answers that question. If redistribution is not permitted or remains unclear, publish a documented acquisition step and small synthetic test fixtures instead of the source corpus or old offline datasets.
2. Decide how the 55 MB source corpus will be supplied. If redistribution is permitted, Git LFS or a versioned external download with SHA256 is preferable to ordinary Git history. Generate the 46 MB SFT examples from the source and committed split definition when possible; this requires a documented generation step and any necessary adjustment to the current manifest validation.
3. Parameterize or redact account-specific project IDs, bucket names, and local paths in public code, tests, example configs, and docs. Keep raw GCS URIs, job IDs, build IDs, and cost ledgers in a private evidence archive or publish a deliberate sanitized summary.
4. Keep the private interview guide out of the public repository. Review the historical PDF for third-party names, permission, and redistribution rights before any publication.
5. Add `.gitignore` before staging. At minimum ignore `.DS_Store`, `__pycache__/`, `*.py[cod]`, virtual environments, local settings, `.env*`, generated checkpoints/adapters, run outputs, and temporary build contexts. Do not rely on `.gitignore` to protect a secret after it has entered history.
6. Check every public README link. The current README and project overview link to internal evidence files; replace or sanitize those targets if the internal files are withheld.

## Proposed commit sequence

| Order | Example commit intent | Include | Keep out at this stage |
|---:|---|---|---|
| 1 | `chore: establish RLAIF repository boundary and hygiene` | Public README, `.gitignore`, dependency files, license only after owner chooses one | Private interview notes, local settings, data, logs |
| 2 | `archive: preserve original university prototype` | `baseline/` source, historical Dockerfiles/configuration, a short note that these paths are unsupported | Large offline JSONL, generated transcripts, exported logs, course PDF; review any selected plots separately |
| 3 | `feat(data): add deterministic splits and provenance checks` | `rebuild/rlaif/shared/`, `data/`, split metadata, manifest, data tests, documented corpus acquisition/generation path | 55 MB corpus and 46 MB derived file until rights/storage choice is settled |
| 4 | `feat(training): add SFT, preference generation, and DPO` | `training/sft_qwen.py`, `training/dpo_qwen.py`, `preferences/`, relevant tests and prompt contracts | Cloud run outputs, raw judge traces |
| 5 | `feat(ppo): add bounded PPO and adaptive continuation` | PPO trainers, `rewards/`, PPO math/rollback/retry tests, design notes | Checkpoints and rollout journals |
| 6 | `feat(eval): add matched held-out evaluation` | `evaluation/`, common metric and simulator tests, protocol description | Raw scenario transcripts unless separately cleared |
| 7 | `ops: add reproducible bounded cloud preparation` | Dockerfile, pinned requirements, fresh identity/preparation helpers, sanitized example configs and tests | `build_context.json`, old dated runnable-looking configs, internal budget/status ledgers |
| 8 | `docs: publish architecture and measured results` | Public `DETAILED_PROJECT_REPORT.md`, sanitized overview and results, generated SVG figures, numeric chart extract and its renderer, a clear limitations section | `INTERVIEW_PREP.md`, raw cloud identifiers, private cost/credit details |

The exact grouping may change if a commit would otherwise leave broken imports or tests. Keep each commit about one understandable concern, run its relevant checks, and make the final repository pass the full in-container test suite (85 tests in the validated v14 closeout image). Preserve immutable run identifiers and hashes in a private provenance index even when public summaries omit the private bucket address.

## File disposition guide

| Path/group | Public commit recommendation | Reason |
|---|---|---|
| `rebuild/rlaif/` source and tests | Yes, after identifier/fixture review | Supported implementation |
| `rebuild/cloud/Dockerfile`, requirements, preparation helpers | Yes, after account-specific defaults are parameterized | Reproducible runtime and bounded job setup |
| `DETAILED_PROJECT_REPORT.md` | Yes | Public, sanitized architecture and results narrative |
| `figures/` SVGs, renderer, and numeric metric extract | Yes, after final chart review | README visuals with reproducible, sanitized values; raw logs remain private |
| `INTERVIEW_PREP.md` | No; keep private | Private coaching and internal budget details |
| `baseline/` source and small configs | Yes, in a labeled archival commit | Honest before/after provenance |
| `baseline/Offline_rl_datasets/`, generated `output*.jsonl`, `vertex_ai_logs/` | Private archive or later reviewed evidence release | Size, possible transcript content, and historical context |
| `final_project_ryadav4_sa69_vshinde.pdf` | Hold pending rights/privacy review | Course report with named contributors |
| `airdialogue_sampled_5k.jsonl` | Conditional: rights check, then LFS or external versioned source | 55 MB source data; current runtime needs it |
| `rebuild/artifacts/sft_examples.jsonl` | Prefer regenerate and verify hash | 46 MB derived data |
| `rebuild/artifacts/splits.json`, `dataset_manifest.json` | Yes, after data-publication decision | Split identity and provenance |
| `rebuild/cloud/budget.json`, `STATUS.md`, `build_context.json`, dated job JSONs | Keep private or publish sanitized/archive-labeled versions | Internal identifiers; temporary path; historical commands are normalized |
| `.claude/settings.local.json`, `.DS_Store`, `__pycache__/`, `*.pyc` | No | Local/generated state |

## Review gate before each push

Inspect the staged file list and staged diff, run a secret and personal-data scan on the staged content and history, check large files and redistribution rights, verify all public links, and run the relevant tests. For the final commit, run the complete container suite, verify dataset/protocol hashes, and confirm the public result table matches the preserved internal evidence. Record the exact published image digest in the release notes if the public repo mentions a runnable cloud image.

## When the remote URL arrives

Confirm its intended root and visibility, then map these groups to the actual repository layout. Prepare the public-safe file set and reviewable diffs before any staging or commit. The remote URL is not needed to decide whether the historical source is worth keeping; it is needed to verify the repository boundary and publication posture.

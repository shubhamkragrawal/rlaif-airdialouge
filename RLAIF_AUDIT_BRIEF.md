# RLAIF Repo Audit — Read-Only Brief

**Goal:** understand what this repository actually contains and produce a single `SUMMARY.md` at the repo root that accurately describes it. You are reverse-engineering a project the owner built earlier and has not touched in a while.

---

## Hard constraints — read these first

1. **This is a read-only audit.** Do not modify, create, rename, move or delete any file, with exactly one exception: you will write `SUMMARY.md` at the repo root at the very end.
2. **No state-changing git.** `git log`, `git show`, `git diff`, `git status`, `git ls-files`, `git branch -a` are fine. Do **not** run `commit`, `add`, `checkout`, `switch`, `stash`, `clean`, `reset`, `rebase`, `merge`, `push`, `pull`, or `gc`.
3. **Do not install anything.** No `pip install`, `uv add`, `npm install`, `conda`. If a dependency is missing, note it — don't fix it.
4. **Do not execute the project.** No training runs, no inference, no API calls, no `make` targets that do work. Reading code is how you learn what it does.
5. **Do not print secrets.** If you encounter `.env`, `*.json` service-account keys, `credentials`, `*.pem`, or anything that looks like a token, **do not output its contents**. Record only: the filename, whether it is git-tracked, and whether it appears in `.gitignore`.
6. **Cite everything.** Every factual claim in `SUMMARY.md` gets a `path/to/file.py:LINE` reference. If you cannot point at a line, it goes under *Open Questions*, not in the body.
7. **Never guess a number.** Metrics, hyperparameters and results are copied from files, never inferred. If a result is referenced but the file is absent, say so explicitly.

---

## Investigation order

Work through these in sequence. Take notes as you go; write the file only at the end.

### 1 · Orientation
- Full directory tree, excluding `.git`, `__pycache__`, `node_modules`, virtualenvs, and checkpoint blobs.
- Read any existing `README`, `notes`, `TODO`, docs folder.
- Read `pyproject.toml` / `requirements.txt` / `environment.yml` — what's the dependency surface, and what does it imply about the stack?
- Read `Makefile`, `.gitignore`, `Dockerfile`, CI workflow files if present.

### 2 · The story in the commit history
- `git log --oneline --date=short --pretty=format:'%h %ad %s'` — the whole history.
- First commit date, last commit date, total commits, number of contributors.
- `git log --stat` on the ~15 most substantial commits: where did the effort actually go?
- Any branches other than the default? Any work that looks abandoned mid-stream?
- **What narrative does the history tell** — steady build, one big dump, a pivot partway through?

### 3 · Entry points
- What can a person actually run? Scripts with `if __name__ == "__main__"`, CLI definitions, Makefile targets, notebook entry cells.
- For each: what does it do, what does it need (env vars, data, GPU), what does it produce?

### 4 · Configuration
- Every config file, YAML, dataclass or constants module.
- Extract the real values: model names/checkpoints, learning rates, batch sizes, KL coefficient(s), epochs/steps, LoRA rank if present, temperature/sampling settings, seeds.
- Note anything hardcoded in-line that should have been config.

### 5 · The RLAIF loop — the core of this audit
Trace the data path end to end and describe it concretely:
- Where do the **initial prompts / scenarios** come from?
- How does the **User Simulator** work — is it a prompted LLM, a script, a fixed dataset? What prompt does it use?
- How does the **AI Judge** work — which model, what rubric, what output format, how are parse failures handled?
- How is a **preference pair** constructed from a judged interaction? What exactly is chosen vs rejected?
- Where does the generated preference dataset land on disk, and **does that file exist in the repo**? How many rows?

Produce a short ASCII or mermaid flow of this loop in the summary.

### 6 · Training
- **PPO path:** which library (TRL? custom?), the reward signal, how the KL penalty is applied, reference model handling, whether there's a value head.
- **DPO path:** beta, the pairs it consumes, reference model handling.
- What's genuinely shared between the two paths and what's duplicated?
- Are both complete, or is one a stub?

### 7 · Evaluation — extract every number you find
- What metrics are computed, and in what file?
- Find every results artifact: `results/`, `*.json`, `*.csv`, `*.png`, logs, wandb or MLflow directories, notebook output cells.
- **Copy the actual numbers into a table**, with the file each came from.
- Is there a PPO-vs-DPO comparison anywhere, and is it complete?
- Any plots on disk? List paths.

### 8 · Infrastructure
- Vertex AI usage: which services, which regions, what's parameterised by project id.
- Docker, cloud configs, any deploy scripts.
- Where was training meant to run — locally, Vertex Custom Training, a notebook?

### 9 · Completeness audit
- Every `TODO`, `FIXME`, `XXX`, `HACK`, `NotImplementedError`, `pass  # stub`.
- Functions defined but never called. Modules never imported.
- Tests: do any exist, do they cover anything meaningful?
- Anything that reads as half-finished or abandoned.

### 10 · Pre-publication safety check
The owner intends to make this repository public. Report, without printing any values:
- Any credential-shaped file, and whether it is git-tracked.
- Any hardcoded project id, bucket name, endpoint, email address or API key **in tracked source**. Give `path:line` and say what *kind* of value it is — never the value itself.
- Whether any secret appears in **git history** even if deleted from the working tree (`git log --all --full-history -- <path>` for suspicious paths).
- Anything in `data/` or `outputs/` that is tracked and shouldn't be.

---

## Output: `SUMMARY.md`

Write exactly one file, at the repo root, with these sections in this order. Keep it factual and plain — this is a technical record, not a pitch.

```markdown
# RLAIF — Repository Summary
_Audited <date>. Read-only pass; no files were modified._

## 1. What this project is
Three to five sentences, plain language. What problem, what approach, what state.

## 2. Current state
One of: complete and runnable · complete but unverified · partially built · prototype.
Justify the call in two sentences, with references.

## 3. Architecture
The pipeline as actually implemented. Flow diagram plus a paragraph.
Name the real modules, not idealised ones.

## 4. The RLAIF loop
Prompts → simulator → judge → preference pairs → trainer.
Include the actual judge rubric and the preference-pair schema.
State how many pairs were generated and whether the file is present.

## 5. Training
| | PPO | DPO |
|---|---|---|
| library / implementation | | |
| key hyperparameters | | |
| KL handling | | |
| status | | |

## 6. Results
Every number found, with its source file. Say plainly if a claimed
result has no artifact behind it.

## 7. Repo map
| Path | Purpose | Status |
Table covering every meaningful file.

## 8. How to run it
Reconstructed from the code. Prerequisites, env vars, commands, expected
outputs. Flag any step you could not verify.

## 9. Gaps and unfinished work
Bulleted, each with a path reference. Ordered by how much it would
matter to someone reading this repo cold.

## 10. Pre-publication checklist
Secrets, hardcoded identifiers, tracked data that shouldn't be.
Findings only — no values.

## 11. Open questions for the owner
Things the code cannot answer. Be specific enough that each one is a
yes/no or one-line answer, not "what was your intent here".
```

---

## Tone

Describe what is there, not what should be there. If something is incomplete, say so once, neutrally, and move on — the owner knows this is an old project and needs an accurate map, not encouragement or criticism. Where the code is good, saying so briefly is useful signal too.

When you finish, report in chat: the state call from §2, the three largest gaps, and any pre-publication finding — then stop.

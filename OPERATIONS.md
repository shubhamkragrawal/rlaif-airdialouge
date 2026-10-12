# Local operations

The conversation service is a workstation-only development server. It accepts
only `127.0.0.1`, `localhost`, and `::1` as bind addresses. It rejects wildcard,
LAN, and public addresses. The default is `127.0.0.1:8000`.

From the `RLAIF` directory, start the deterministic simulator:

```sh
python -m local_app.server
```

To enable lazy use of a locally present model adapter:

```sh
python -m local_app.server --model
```

Both modes serve the same HTTP API. `--model` does not download adapters or base
models. A request uses a model only when its files and runtime dependencies are
already available; otherwise the app uses its deterministic fallback. Model
adapter downloads, when intentionally needed, are a separate explicit operation:
`python -m local_app.download_adapters`.

Operational endpoints:

- `GET /health` returns `{"status":"ok","simulator":true}` when the server
  is responding.
- `GET /adapters` returns the complete four-adapter inventory: `sft`, `dpo`,
  `ppo_checkpoint_19`, and `ppo_adaptive_50`. This endpoint reports names only;
  it does not reveal local paths or credentials.
- `GET /runtime-status?adapter=sft` returns the selected adapter's lifecycle,
  availability, runtime mode, and provenance status. The response intentionally
  omits local paths and detailed errors.

From the `RLAIF` directory, run the complete local regression set with:

```sh
./scripts/full-local-regression
```

The script runs all `local_app` tests, including the regression catalog and the
browser client's behavioral DOM checks, then runs the rebuild tests from the
`rebuild` working directory. Node.js 18 or newer is required for the browser
checks. If Node is not on `PATH`, set `RLAIF_NODE` to its executable. Missing
Node is a hard, clearly reported failure; these checks are not silently skipped.
The script uses `.venv/bin/python` when present and otherwise `python3`; set
`PYTHON` to a Python 3.11 or newer executable when needed.

## Release evidence (local only)

After an integrated release artifact has already been produced, run the local
checks and bundle their outcomes with hash references to that artifact:

```sh
.venv/bin/python -m local_app.release_evidence \
  --matrix artifacts/matrix/releases/v1/release-v1.json
```

The recorder runs six checks: the full local suite, rebuild suite, browser
regressions, regression catalog, clean-package contract, and operations
contracts. It does not run the integrated model matrix, download models, or
contact cloud services. It writes an immutable, atomically published
`artifacts/release/evidence-v1-<UTC timestamp>/` directory containing the
evidence bundle and six JSON check records. Process output is hashed in memory;
stdout/stderr contents, machine paths, and environment values are not written
to the evidence files. A failed check is still recorded and produces a
nonzero exit status. `--checks` can run a subset for diagnosis, but a subset
bundle cannot pass the release gate.

The clean-package check copies only the package source and build metadata into
a temporary staging directory, builds one wheel and one source archive with
the already available `setuptools.build_meta` backend, then creates a fresh
temporary venv. It installs the wheel using pip with `--no-index --no-deps`,
verifies imports from that venv in isolated mode with the verification working
directory outside the source copy, starts the installed server on loopback and
checks `/health`, and runs a deterministic one-scenario evaluation. The sdist
is verified as a build output; installation is tested from the wheel. It does
not install build tools or dependencies globally or fetch build requirements.
If the selected Python does not already have `setuptools>=68`, the check
records `build_backend_unavailable` instead of attempting a download.

Review the recorded check statuses, then verify the bundle:

```sh
.venv/bin/python -m local_app.release_gate \
  artifacts/release/evidence-v1-<UTC timestamp>/evidence-v1.json \
  --root .
```

Use the exact generated directory name in place of `<UTC timestamp>`. Evidence
paths must remain within this project, and the supplied integrated artifact
must be the real release output for the candidate under review. The recorder
validates the artifact's envelope and hash but does not create or claim an
integrated release pass.

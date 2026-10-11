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

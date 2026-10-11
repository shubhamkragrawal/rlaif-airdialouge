# Read-only release gate

`local_app.release_gate` verifies evidence that has already been produced. It
does not run tests, load models, access cloud services, or change the checkout.
Provide a version 1 evidence bundle containing hash-pinned JSON result records
for `local`, `rebuild`, `browser`, `catalog`, `clean_package`, and `ops`, plus a
hash-pinned integrated matrix report or `matrix_release` artifact. Each check record must contain a human
readable `command`, `status: "passed"`, and `exit_code: 0`. The supplied command
and result are claims backed by the referenced file; the gate verifies their
integrity and declared result, not the historical process that produced them.

Example invocation:

```sh
python -m local_app.release_gate /path/to/evidence-v1.json --root /path/to/evidence-root
```

The bundle's paths are relative to `--root` (or the bundle directory by
default). The output is atomically replaced at
`artifacts/release/release-gate-v1.json`; exit status is zero only on pass.
The report records SHA-256 references to all accepted evidence. A failed gate
still writes a versioned failure report when the bundle can be read.

The integrated report must contain all four complete integrated arms on the
same nonempty scenario cohort, consistent controller/outcome-schema/runtime
fingerprints, complete local-model provenance and participation, hashed run
and turn manifests, and required release metrics. Unsupported transaction
claims and malformed output must be zero. Fallback rate must be present and in
the valid `[0, 1]` range; it is reported as a metric rather than treated as a
failure by itself. The gate also fails if protected data, local model artifacts,
or cloud-ledger/training records appear among staged paths.

This gate does not constitute an actual release pass until real supplied
evidence is evaluated. Synthetic fixtures in the unit tests exercise gate
acceptance and tamper rejection only.

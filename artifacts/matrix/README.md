# Adapter matrix artifacts

`schema-v1.json` defines the versioned matrix baseline envelope, including required
metrics, canonical run and turn manifests, and controller/schema/corpus/generation
fingerprints. Each turn manifest includes its redacted immutable observability
record, the metric evidence derived from that turn, the record hash, and a canonical
hash covering both. Run manifests contain the immutable run record and its canonical
record hash. Baselines are
written explicitly with `python -m local_app.matrix --mode controller_only
--freeze-baseline artifacts/matrix/baseline-v1.json --output /tmp/matrix.json`.
The repository intentionally has no frozen integrated baseline. An integrated arm
is complete only when the evaluator reports the requested `integrated` layer, at
least one turn used the model, and normalized runtime status and adapter provenance
carry the task 20 lifecycle envelope (`ready`, `local_model`, available, complete
provenance), an adapter digest, base model and revision, and matching
generation settings, device, precision/dtype, and non-negative peak-memory bytes.
Integrated run manifests copy device, precision, and peak memory into their hardware
record. Each adapter runtime is released before the next adapter is
loaded. Missing participation or provenance makes the arm incomplete and prevents
baseline freezing. Controller-only manifests explicitly carry no model provenance.
Matrix reports are atomically replaced and retain every scenario row and turn.

Turn evidence is captured from the evaluator's public controller results. It records
validation/fallback state, malformed-output indicators, repetition only when a
validator or explicit result field supplies evidence,
grounded entity candidates, no-results presentation, transaction claims and evidence,
latency, and reported tokens. Identical response text across turns is not repetition
evidence. Aggregate metrics include validation coverage and a repetition evidence
rate. Search intent includes “show flights,” and no-results metrics are populated
for those turns. Time-to-first-token remains null because this runner
does not stream tokens. These local metrics are not judge scores; judge-dependent
scores remain separately labeled and null unless separately supplied.

Booking support is derived from the current turn's typed hold and booking results.
Both first-time and idempotent repeated confirmations therefore retain their actual
simulator `hold_id` and `booking_id` in `transaction_evidence_ids`; digests are not
used as transaction evidence.

Use `--baseline PATH` to compare a run with a compatible frozen baseline. The
scenario IDs, corpus digest, and generation settings must match exactly.

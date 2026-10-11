# RLAIF Release Report

Report contract version: 1

## Release status

**PENDING — no complete integrated release artifact is available.**

## Evidence sources

- Held-out policy evidence (`policy-only-v1.json`): missing; version `policy report v1`; fields `report_version, evaluation_layer, evidence_artifacts, results_imported, summary`.
- Controller-only matrix (`controller-only-v1.json`): SHA-256 `381560c895f9171f83c34d4b631b53fc9d790e606bf57e7cff606542f13954ea`; version `matrix v1`; fields `schema, report_version, layer, cohort, adapters`.
- Integrated matrix release (`release-v1.json`): missing; version `release v1`; fields `schema, release_version, layer, identity, adapters`.
- Browser regressions (`browser_regressions_v1.json`): SHA-256 `19c94f1af3f53f06e696e4010aa04fa90454b9ce69306a474c305557b3f70fac`; version `browser regression v1`; fields `artifact_version, cases`.

## Evaluation layers

### Held-out policy

Evidence reference status: `unavailable`. Results imported: `false`.
Policy metrics: unavailable; the referenced evidence report does not import result metrics.

### Controller-only

Cohort: `matrix-20ed0c0dc64dad74580b`; scenarios: `50`.
This measures controller behavior without model participation.

#### `sft` — `complete`

- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.0
- `fallback_reason_distribution`: {}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 0.0, "output": 0.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 0.6721637081628775}
- `peak_memory_mb`: 32.09375
- `judge_score`: {"basis": "judge-dependent", "value": null}
Evidence: `adapters.sft.metrics`; manifest hash `40fbeadf91ac6c02ddf9595370658a825031059f46dae5b2a325886768fbd764`.

#### `dpo` — `complete`

- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.0
- `fallback_reason_distribution`: {}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 0.0, "output": 0.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 0.5979455658234656}
- `peak_memory_mb`: 33.796875
- `judge_score`: {"basis": "judge-dependent", "value": null}
Evidence: `adapters.dpo.metrics`; manifest hash `5361d91a9fa835af2856b64c671363bfc932e6b4e9ab1c6b821ccde6f0d99430`.

#### `ppo_checkpoint_19` — `complete`

- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.0
- `fallback_reason_distribution`: {}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 0.0, "output": 0.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 0.5882507128476658}
- `peak_memory_mb`: 35.359375
- `judge_score`: {"basis": "judge-dependent", "value": null}
Evidence: `adapters.ppo_checkpoint_19.metrics`; manifest hash `d25aa935a111f77d684e245fce1ba517c0effe46479b0a2fb73ffb08d8935d1a`.

#### `ppo_adaptive_50` — `complete`

- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.0
- `fallback_reason_distribution`: {}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 0.0, "output": 0.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 0.5848602713633548}
- `peak_memory_mb`: 36.921875
- `judge_score`: {"basis": "judge-dependent", "value": null}
Evidence: `adapters.ppo_adaptive_50.metrics`; manifest hash `5a72414be70e0bcafedbc3994d22ad2cd00576776149a93a0173e22daf4f4f5d`.

### Integrated

Integrated metrics: unavailable. No complete integrated release artifact exists; controller-only results do not substitute for integrated evaluation.

## Browser regressions

- `BR001`: Reject flight 9999 with explicit invalid-option wording; booking remains absent.
- `BR002`: An impossible IAD–LAS search under $1 yields friendly no-results text and structured raw results [].
- `BR003`: A model confirmation without simulator transaction evidence is rejected.
- `BR004`: Repeated model output is rejected and deterministic concise text is used.
- `BR005`: Positive control: model response is accepted when flight and fare match retrieved local data.
- `BR006`: Executes the shipped page script in a JavaScript engine with a behavioral DOM and submits injection/XSS markup; verifies no payload elements, event attributes, or execution marker appear, invalid selection is rejected, and the input clears on submit and after response.
- `BR007`: Executes the shipped page script and verifies submitted accepted text appears literally in the transcript with no parsed markup, and the input is empty both on submit and after the assistant response.

## Interpretation and limitations

Judge scores are judge-dependent labels and are reported only when a metric artifact supplies a value; no score is inferred.
Fallback rates and reason distributions, sample sizes, Wilson intervals, latency, token counts, and memory remain unavailable whenever their source field is absent.
The simulator is local and simulated; these artifacts do not establish performance against live airline inventory or booking systems.
Regression cases document browser behaviors; they are not population estimates and do not imply an integrated model run.

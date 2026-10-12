# RLAIF Release Report

Report contract version: 1

## Release status

**Integrated artifact present; per-adapter completeness is shown below.**

## Evidence sources

- Held-out policy evidence (`policy-only-v1.json`): SHA-256 `59bf2bb64932fb15eb2f2a908595ac2357603e193ae0714ff179f699f8b8427d`; version `policy report v1`; fields `report_version, evaluation_layer, evidence_artifacts, results_imported, summary`.
- Controller-only matrix (`controller-only-v1.json`): SHA-256 `381560c895f9171f83c34d4b631b53fc9d790e606bf57e7cff606542f13954ea`; version `matrix v1`; fields `schema, report_version, layer, cohort, adapters`.
- Integrated matrix release (`release-v1.json`): SHA-256 `ed859fe9f322f1d5f853d653e5ea5b93b06359629b68a00ec6d7256b651706a9`; version `release v1`; fields `schema, release_version, layer, identity, adapters`.
- Browser regressions (`browser_regressions_v1.json`): SHA-256 `19c94f1af3f53f06e696e4010aa04fa90454b9ce69306a474c305557b3f70fac`; version `browser regression v1`; fields `artifact_version, cases`.

## Evaluation layers

### Held-out policy

Evidence reference status: `incomplete`. Results imported: `true`.
Policy evidence level: `reported_summary`. Aggregate metrics below are imported from the cited summary; they are not scenario-level records.

#### `sft` — summary-level

- `task_completion`: {"basis": "reported_summary", "observed": 50, "rate": 0.26, "successes": 13, "wilson_95": [0.159, 0.396]}
- `valid_transitions`: unavailable
- `unsupported_transaction_claim_rate`: unavailable
- `grounded_entity_precision`: unavailable
- `no_results_presentation_accuracy`: unavailable
- `malformed_output_rate`: unavailable
- `repetition_rate`: unavailable
- `repetition_evidence`: unavailable
- `validation`: unavailable
- `fallback_rate`: unavailable
- `fallback_reason_distribution`: unavailable
- `turns`: unavailable
- `tokens`: unavailable
- `latency_ms`: unavailable
- `peak_memory_mb`: unavailable
- `judge_score`: {"basis": "reported_mean", "scored_scenarios": null, "value": 2.54}
- `booking_success`: {"basis": "reported_summary", "denominator": null, "rate": 0.0}
- `unknown_flight_id_hallucination_rate`: {"basis": "reported_summary", "observed": 50, "rate": 0.0}

#### `dpo` — summary-level

- `task_completion`: {"basis": "reported_summary", "observed": 50, "rate": 0.34, "successes": 17, "wilson_95": [0.224, 0.478]}
- `valid_transitions`: unavailable
- `unsupported_transaction_claim_rate`: unavailable
- `grounded_entity_precision`: unavailable
- `no_results_presentation_accuracy`: unavailable
- `malformed_output_rate`: unavailable
- `repetition_rate`: unavailable
- `repetition_evidence`: unavailable
- `validation`: unavailable
- `fallback_rate`: unavailable
- `fallback_reason_distribution`: unavailable
- `turns`: unavailable
- `tokens`: unavailable
- `latency_ms`: unavailable
- `peak_memory_mb`: unavailable
- `judge_score`: {"basis": "reported_mean", "scored_scenarios": null, "value": 2.58}
- `booking_success`: {"basis": "reported_summary", "denominator": null, "rate": 0.0}
- `unknown_flight_id_hallucination_rate`: {"basis": "reported_summary", "observed": 50, "rate": 0.0}

#### `ppo_checkpoint_19` — summary-level

- `task_completion`: {"basis": "reported_summary", "observed": 50, "rate": 0.38, "successes": 19, "wilson_95": [0.259, 0.518]}
- `valid_transitions`: unavailable
- `unsupported_transaction_claim_rate`: unavailable
- `grounded_entity_precision`: unavailable
- `no_results_presentation_accuracy`: unavailable
- `malformed_output_rate`: unavailable
- `repetition_rate`: unavailable
- `repetition_evidence`: unavailable
- `validation`: unavailable
- `fallback_rate`: unavailable
- `fallback_reason_distribution`: unavailable
- `turns`: unavailable
- `tokens`: unavailable
- `latency_ms`: unavailable
- `peak_memory_mb`: unavailable
- `judge_score`: {"basis": "reported_mean", "scored_scenarios": null, "value": 1.92}
- `booking_success`: {"basis": "reported_summary", "denominator": null, "rate": null}
- `unknown_flight_id_hallucination_rate`: {"basis": "reported_summary", "observed": 50, "rate": 0.0}

#### `ppo_adaptive_50` — summary-level

- `task_completion`: {"basis": "reported_summary", "observed": 50, "rate": 0.34, "successes": 17, "wilson_95": [0.224, 0.478]}
- `valid_transitions`: unavailable
- `unsupported_transaction_claim_rate`: unavailable
- `grounded_entity_precision`: unavailable
- `no_results_presentation_accuracy`: unavailable
- `malformed_output_rate`: unavailable
- `repetition_rate`: unavailable
- `repetition_evidence`: unavailable
- `validation`: unavailable
- `fallback_rate`: unavailable
- `fallback_reason_distribution`: unavailable
- `turns`: unavailable
- `tokens`: unavailable
- `latency_ms`: unavailable
- `peak_memory_mb`: unavailable
- `judge_score`: {"basis": "reported_mean", "scored_scenarios": null, "value": 2.46}
- `booking_success`: {"basis": "reported_summary", "denominator": null, "rate": null}
- `unknown_flight_id_hallucination_rate`: {"basis": "reported_summary", "observed": 50, "rate": 0.0}
Policy evidence limitation: Aggregate values were imported from the local final-results summary. Per-scenario outputs, resolved evaluation revisions, and verified links between local adapter bytes and evaluated adapters are not present locally.
Evidence artifact digests:
- Evidence SHA-256 `874b419da9b5a70a274e997affc7c77d27383c9938edc3983a384256317e2677`
- Evidence SHA-256 `3995b0860f65afb63069f116ca70f6ba42da29cd537baa25a9c8a42958c8de74`
- Evidence SHA-256 `8ecb7a30f8d6ea8dc4f9e42efdcddadb24ee39e3f2ee33053061d26eb7a42aeb`
- Evidence SHA-256 `4a98da30af88e580ee958eaf2ad3a30212719f1dd1b0506d8fd5796727ddd3e3`
- Evidence SHA-256 `5be5782dc2daf8f2fafaa593e9e047baad45ccb3bc3d3fe135f967ededf3668d`
- Evidence SHA-256 `f9cdc4b6c0e893f1e74a7da0426b674ebe2b648b61efc731c928b9150be0c761`
- Evidence SHA-256 `667a6953b9d69b7e30da2774bbd9aea5a683c7d5a0f6feaf3c3b4c61d76199c1`
- Evidence SHA-256 `1c10ed209d52314fef4d8cf118f744dbffad47603b823e7b01666a0c141de1cd`
- Evidence SHA-256 `af943cd0b0ceb8c88bd750fe5eda641e5adc234a42d7b918496d4e34d31fb2a3`

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
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
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
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
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
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
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
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
Evidence: `adapters.ppo_adaptive_50.metrics`; manifest hash `5a72414be70e0bcafedbc3994d22ad2cd00576776149a93a0173e22daf4f4f5d`.

### Integrated

Base model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`; settings: `{"[redacted-key]": "[redacted]", "base_model": "Qwen/Qwen2.5-3B-Instruct", "count": 50, "mode": "model", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1", "start": 0, "temperature": 0.0}`.

#### `sft` — `complete`

Model participation turns: `70`; device: `mps`; precision: `float16`; peak memory bytes: `6327788544`.
- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.21666666666666667
- `fallback_reason_distribution`: {"availability_contradiction": 21, "repeated_sentence": 3, "unsupported_airline": 2}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 25266.0, "output": 1405.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 3118.900779105367}
- `peak_memory_mb`: 6547.421875
- `judge_score`: {"basis": "judge-dependent", "value": null}
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
Evidence fields: `adapters.sft.metrics`, `runtime_provenance`, `run_manifest.record.hardware`; arm fingerprint `7663ed67c73f5463b52c4e6668f7a9be7e8b1750998176223aa6fc29cf0b6d69`.

#### `dpo` — `complete`

Model participation turns: `70`; device: `mps`; precision: `float16`; peak memory bytes: `6327788544`.
- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.075
- `fallback_reason_distribution`: {"availability_contradiction": 8, "unsupported_airline": 1}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 25152.0, "output": 606.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 2266.812146815252}
- `peak_memory_mb`: 6547.421875
- `judge_score`: {"basis": "judge-dependent", "value": null}
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
Evidence fields: `adapters.dpo.metrics`, `runtime_provenance`, `run_manifest.record.hardware`; arm fingerprint `ef31784d2383e0c26c25367dc5dde1a64551efa9a1182129ee1e6a0f7238f895`.

#### `ppo_checkpoint_19` — `complete`

Model participation turns: `70`; device: `mps`; precision: `float16`; peak memory bytes: `6327788544`.
- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.24166666666666667
- `fallback_reason_distribution`: {"availability_contradiction": 20, "repeated_sentence": 6, "unsupported_airline": 3}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 25238.0, "output": 1724.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 3936.6454734891545}
- `peak_memory_mb`: 6547.421875
- `judge_score`: {"basis": "judge-dependent", "value": null}
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
Evidence fields: `adapters.ppo_checkpoint_19.metrics`, `runtime_provenance`, `run_manifest.record.hardware`; arm fingerprint `0518af98c9fc2e40b376d5f03905f183e2b41c5438b03e9fce95103226da2a4a`.

#### `ppo_adaptive_50` — `complete`

Model participation turns: `70`; device: `mps`; precision: `float16`; peak memory bytes: `6327788544`.
- `task_completion`: {"rate": 1.0, "successes": 50, "wilson_95": [0.9286524008666414, 1.0]}
- `valid_transitions`: {"booking_rate": 1.0, "observed": 85, "rate": 1.0, "search_rate": 1.0, "valid": 85}
- `unsupported_transaction_claim_rate`: 0.0
- `grounded_entity_precision`: 1.0
- `no_results_presentation_accuracy`: 1.0
- `malformed_output_rate`: 0.0
- `repetition_rate`: 0.0
- `repetition_evidence`: {"basis": "duplicate_response_or_validator_repetition_reason", "detected_turns": 0, "observed_turns": 120, "rate": 0.0}
- `validation`: {"observed_turns": 120, "rate": 1.0, "valid_turns": 120}
- `fallback_rate`: 0.225
- `fallback_reason_distribution`: {"availability_contradiction": 18, "repeated_sentence": 6, "unsupported_airline": 3}
- `turns`: {"average": 2.4, "p95": 3.0}
- `tokens`: {"input": 25235.0, "output": 1837.0}
- `latency_ms`: {"[redacted-key]": "[redacted]", "total": 3930.0357046906893}
- `peak_memory_mb`: 6547.421875
- `judge_score`: {"basis": "judge-dependent", "value": null}
- `booking_success`: unavailable
- `unknown_flight_id_hallucination_rate`: unavailable
Evidence fields: `adapters.ppo_adaptive_50.metrics`, `runtime_provenance`, `run_manifest.record.hardware`; arm fingerprint `5e929d5885321c802d6f11bdbba09bdc4fa926e0cb35a923b2ab630a30514280`.

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

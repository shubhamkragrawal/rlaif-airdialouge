"""Render an artifact-backed release report without calculating or guessing metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = 1
POLICY_DEFAULT = ROOT / "artifacts" / "evaluation" / "policy-only-v1.json"
CONTROLLER_DEFAULT = ROOT / "artifacts" / "matrix" / "controller-only-v1.json"
INTEGRATED_DEFAULT = ROOT / "artifacts" / "matrix" / "releases" / "v1" / "release-v1.json"
BROWSER_DEFAULT = ROOT / "test_artifacts" / "browser_regressions_v1.json"
OUTPUT_DEFAULT = ROOT / "RLAIF_2_RELEASE_REPORT.md"
ADAPTERS = ("sft", "dpo", "ppo_checkpoint_19", "ppo_adaptive_50")
METRIC_FIELDS = (
    "task_completion", "valid_transitions", "unsupported_transaction_claim_rate",
    "grounded_entity_precision", "no_results_presentation_accuracy",
    "malformed_output_rate", "repetition_rate", "repetition_evidence", "validation",
    "fallback_rate", "fallback_reason_distribution", "turns", "tokens", "latency_ms",
    "peak_memory_mb", "judge_score", "booking_success",
    "unknown_flight_id_hallucination_rate",
)


def _read(path: Path) -> tuple[Any | None, str | None, str | None]:
    """Return parsed JSON and its digest, without leaking input paths into report text."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None, None
    digest = hashlib.sha256(raw).hexdigest()
    try:
        return json.loads(raw), digest, None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, digest, "invalid JSON"


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    lines = []
    for name in METRIC_FIELDS:
        value = metrics.get(name)
        rendered = "unavailable" if value is None else json.dumps(_safe(value), sort_keys=True, ensure_ascii=True)
        lines.append(f"- `{name}`: {rendered}")
    return lines


def _hardware_matches_provenance(record: Mapping[str, Any], provenance: Mapping[str, Any],
                                 summary: Any) -> bool:
    hardware = record.get("hardware")
    if not isinstance(hardware, Mapping):
        return False
    for field in ("device", "precision"):
        expected = provenance.get(field)
        if not isinstance(expected, str) or not expected or hardware.get(field) != expected:
            return False
    expected_peak = provenance.get("peak_memory_bytes")
    observed_peak = hardware.get("peak_memory_bytes")
    if (not isinstance(expected_peak, int) or isinstance(expected_peak, bool) or expected_peak < 0
            or not isinstance(observed_peak, int) or isinstance(observed_peak, bool)
            or observed_peak != expected_peak):
        return False
    if "peak_process_rss_mb" in hardware:
        return (isinstance(summary, Mapping)
                and hardware["peak_process_rss_mb"] == summary.get("peak_process_rss_mb"))
    return True


def _safe(value: Any) -> Any:
    """Redact paths and credential-like strings while retaining measured values."""
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            if re.search(r"credential|password|secret|api[_-]?key|token|passenger|email|name", str(key), re.I):
                redacted["[redacted-key]"] = "[redacted]"
            else:
                redacted[str(key)] = _safe(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        if re.search(r"(?i)(?:[A-Z]:\\|(?:^|\s)/(?:Users|home|private|tmp|opt|var|etc)/|(?:api[_-]?key|token|password|secret)\s*[:=])", value):
            return "[redacted]"
    return value


def _source(label: str, path: Path, digest: str | None, field: str, version: str) -> str:
    state = f"SHA-256 `{digest}`" if digest else "missing"
    return f"- {label} (`{path.name}`): {state}; version `{version}`; fields `{field}`."


def _integrated_complete(integrated: Any) -> bool:
    if (not isinstance(integrated, Mapping)
            or integrated.get("schema") != "rlaif.local_app.matrix_release"
            or integrated.get("release_version") != 1
            or integrated.get("mode") != "integrated"
            or integrated.get("layer") != "integrated"):
        return False
    supplied_hash = integrated.get("sha256")
    payload = {key: value for key, value in integrated.items() if key != "sha256"}
    if supplied_hash != hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest():
        return False
    cohort = integrated.get("cohort") or {}
    ids = cohort.get("scenario_ids")
    if (not isinstance(ids, list) or len(ids) != 50 or cohort.get("scenario_count") != 50
            or any(not isinstance(item, int) or isinstance(item, bool) for item in ids)
            or len(set(ids)) != len(ids)):
        return False
    arms = integrated.get("adapters")
    if not isinstance(arms, Mapping) or set(arms) != set(ADAPTERS):
        return False
    for name in ADAPTERS:
        arm = arms[name]
        if not isinstance(arm, Mapping):
            return False
        provenance = arm.get("runtime_provenance") or {}
        participation = arm.get("model_participation") or {}
        manifest = arm.get("run_manifest") or {}
        metrics = arm.get("metrics") or {}
        if not all(isinstance(item, Mapping) for item in (provenance, participation, manifest, metrics)):
            return False
        run_record = manifest.get("record")
        hardware = run_record.get("hardware") if isinstance(run_record, Mapping) else None
        provenance_payload = ({key: value for key, value in provenance.items() if key != "sha256"}
                              if isinstance(provenance, Mapping) else {})
        try:
            provenance_hash_valid = provenance.get("sha256") == hashlib.sha256(json.dumps(
                provenance_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False).encode()).hexdigest()
            run_hash_valid = (isinstance(run_record, Mapping)
                              and manifest.get("sha256") == hashlib.sha256(json.dumps(
                                  run_record, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode()).hexdigest())
        except (TypeError, ValueError):
            return False
        turns = arm.get("turn_manifests")
        turn_hashes_valid = isinstance(turns, list) and bool(turns)
        scenario_turn_counts = {scenario_id: 0 for scenario_id in ids}
        observed_model_counts = {key: 0 for key in (
            "attempted_turns", "generated_turns", "accepted_turns", "fallback_turns")}
        observed_token_totals = {"input": 0, "output": 0}
        observed_token_values = {"input": False, "output": False}
        if turn_hashes_valid:
            for turn in turns:
                if not isinstance(turn, Mapping) or not isinstance(turn.get("record"), Mapping):
                    turn_hashes_valid = False
                    break
                record = turn.get("record")
                evidence = turn.get("evidence")
                try:
                    record_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                    full_hash = hashlib.sha256(json.dumps({"record": record, "evidence": evidence},
                                                          sort_keys=True, separators=(",", ":"),
                                                          ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                except (TypeError, ValueError):
                    turn_hashes_valid = False
                    break
                if turn.get("record_sha256") != record_hash or turn.get("sha256") != full_hash:
                    turn_hashes_valid = False
                    break
                if (not isinstance(evidence, Mapping)
                        or evidence.get("unsupported_transaction_claim") is not False
                        or evidence.get("malformed_output") is not False
                        or evidence.get("repetitive_output") is not False
                        or evidence.get("validation_result") != "valid"):
                    turn_hashes_valid = False
                    break
                if evidence.get("transaction_claim") is True:
                    transaction = evidence.get("authoritative_transaction_evidence")
                    if (not isinstance(transaction, Mapping)
                            or transaction.get("current_turn") is not True
                            or transaction.get("booking_status") != "confirmed"
                            or transaction.get("simulated") is not True
                            or not re.fullmatch(r"hold_[0-9a-f]{24}", str(transaction.get("hold_id", "")))
                            or not re.fullmatch(r"booking_[0-9a-f]{24}", str(transaction.get("booking_id", "")))
                            or (record or {}).get("transaction_evidence_ids")
                            != [transaction.get("hold_id"), transaction.get("booking_id")]):
                        turn_hashes_valid = False
                        break
                try:
                    scenario_id = record.get("scenario_id")
                    if isinstance(scenario_id, bool):
                        raise ValueError
                    scenario_id = int(scenario_id)
                except (TypeError, ValueError):
                    turn_hashes_valid = False
                    break
                if scenario_id not in scenario_turn_counts or evidence.get("scenario_id") != scenario_id:
                    turn_hashes_valid = False
                    break
                scenario_turn_counts[scenario_id] += 1
                for counter, evidence_key in (
                        ("attempted_turns", "model_attempted"),
                        ("generated_turns", "model_generated"),
                        ("accepted_turns", "model_accepted"),
                        ("fallback_turns", "model_fallback")):
                    flag = evidence.get(evidence_key)
                    if not isinstance(flag, bool):
                        turn_hashes_valid = False
                        break
                    observed_model_counts[counter] += int(flag)
                if not turn_hashes_valid:
                    break
                for token_name, evidence_name in (("input", "input_tokens"),
                                                   ("output", "output_tokens")):
                    value = evidence.get(evidence_name)
                    if value is not None:
                        if (isinstance(value, bool) or not isinstance(value, (int, float))
                                or not math.isfinite(value) or value < 0):
                            turn_hashes_valid = False
                            break
                        observed_token_totals[token_name] += value
                        observed_token_values[token_name] = True
                if not turn_hashes_valid:
                    break
                if (evidence.get("valid_search_transition") is False
                        or evidence.get("valid_booking_transition") is False
                        or (evidence.get("expected_no_results") is True
                            and evidence.get("presented_no_results") is not True)):
                    turn_hashes_valid = False
                    break
        target_rates = {
            "task_completion": 1.0, "valid_transitions": 1.0,
            "grounded_entity_precision": 1.0, "no_results_presentation_accuracy": 1.0,
            "unsupported_transaction_claim_rate": 0.0, "malformed_output_rate": 0.0,
            "repetition_rate": 0.0, "repetition_evidence": 0.0, "validation": 1.0,
        }
        quality_complete = True
        for field, target in target_rates.items():
            value = metrics.get(field)
            value = value.get("rate") if isinstance(value, Mapping) else value
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != target:
                quality_complete = False
                break
        counts = tuple(participation.get(key) for key in (
            "attempted_turns", "generated_turns", "accepted_turns", "fallback_turns"))
        counts_valid = (all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                            for value in counts)
                        and participation.get("required") is True)
        if counts_valid:
            attempted, generated, accepted, fallbacks = counts
            counts_valid = (attempted > 0 and generated > 0 and generated <= attempted
                            and accepted <= generated and fallbacks <= attempted
                            and participation.get("turns") == generated
                            and observed_model_counts == {
                                "attempted_turns": attempted, "generated_turns": generated,
                                "accepted_turns": accepted, "fallback_turns": fallbacks})
        scenarios = arm.get("scenarios")
        scenario_coverage_valid = isinstance(scenarios, list) and [
            row.get("scenario_id") for row in scenarios if isinstance(row, Mapping)
        ] == ids
        if scenario_coverage_valid:
            for row in scenarios:
                row_metrics = row.get("metrics")
                turn_count = row_metrics.get("turn_count") if isinstance(row_metrics, Mapping) else None
                if (not isinstance(turn_count, int) or isinstance(turn_count, bool)
                        or turn_count < 0 or scenario_turn_counts[row["scenario_id"]] != turn_count):
                    scenario_coverage_valid = False
                    break
                normalized = row.get("normalized_outcome")
                success = row_metrics.get("task_success") if isinstance(row_metrics, Mapping) else None
                if (not isinstance(success, bool) or not isinstance(normalized, Mapping)
                        or normalized.get("task_success") is not success):
                    scenario_coverage_valid = False
                    break
        completion = metrics.get("task_completion")
        successes = (sum(row["metrics"]["task_success"] is True for row in scenarios)
                     if scenario_coverage_valid else None)
        completion_valid = (isinstance(completion, Mapping)
                            and isinstance(scenarios, list)
                            and isinstance(completion.get("successes"), int)
                            and not isinstance(completion.get("successes"), bool)
                            and completion.get("successes") == successes
                            and completion.get("rate") == successes / len(ids)
                            and metrics.get("scenario_count") == len(ids))
        token_metrics = metrics.get("tokens")
        tokens_valid = isinstance(token_metrics, Mapping)
        if tokens_valid:
            for token_name in ("input", "output"):
                retained_total = (observed_token_totals[token_name]
                                  if observed_token_values[token_name] else None)
                if token_metrics.get(token_name) != retained_total:
                    tokens_valid = False
                    break
        inference = metrics.get("model_inference")
        inference_valid = (isinstance(inference, Mapping)
                           and all(inference.get(key) == value for key, value in zip(
                               ("attempted_turns", "generated_turns", "accepted_turns", "fallback_turns"),
                               counts))
                           and isinstance(inference.get("fallback_reasons"), list))
        if (arm.get("layer") != "integrated" or arm.get("status") != "complete"
                or arm.get("scenario_ids") != ids or provenance.get("complete") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("sha256", "")))
                or not counts_valid
                or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256", "")))
                or not _hardware_matches_provenance(run_record, provenance, arm.get("summary"))
                or not quality_complete
                or metrics.get("scenario_count") != len(ids)
                or not provenance_hash_valid or not run_hash_valid or not turn_hashes_valid
                or not scenario_coverage_valid or not completion_valid or not tokens_valid
                or not inference_valid):
            return False
    return True


def render_release_report(*, policy_path: str | Path = POLICY_DEFAULT,
                          controller_path: str | Path = CONTROLLER_DEFAULT,
                          integrated_path: str | Path = INTEGRATED_DEFAULT,
                          browser_path: str | Path = BROWSER_DEFAULT) -> str:
    """Build Markdown using only allowlisted values copied from versioned artifacts."""
    paths = {"policy": Path(policy_path), "controller": Path(controller_path),
             "integrated": Path(integrated_path), "browser": Path(browser_path)}
    loaded = {key: _read(path) for key, path in paths.items()}
    policy, policy_hash, _ = loaded["policy"]
    controller, controller_hash, _ = loaded["controller"]
    integrated, integrated_hash, _ = loaded["integrated"]
    browser, browser_hash, _ = loaded["browser"]

    policy_layer_raw = policy.get("evaluation_layer", {}) if isinstance(policy, Mapping) else {}
    policy_valid = (isinstance(policy, Mapping) and policy.get("report_version") == 1
                    and isinstance(policy_layer_raw, Mapping) and policy_layer_raw.get("layer") == "policy_only")
    controller_valid = (isinstance(controller, Mapping) and controller.get("schema") == "rlaif.local_app.adapter_matrix"
                        and controller.get("report_version") == 1 and controller.get("layer") == "controller_only")
    integrated_valid = _integrated_complete(integrated)

    out = ["# RLAIF Release Report", "", "Report contract version: 1", ""]
    out += ["## Release status", ""]
    out.append("**PENDING — no complete integrated release artifact is available.**" if not integrated_valid
               else "**Integrated artifact present; per-adapter completeness is shown below.**")
    out += ["", "## Evidence sources", "",
            _source("Held-out policy evidence", paths["policy"], policy_hash,
                    "report_version, evaluation_layer, evidence_artifacts, results_imported, summary", "policy report v1"),
            _source("Controller-only matrix", paths["controller"], controller_hash,
                    "schema, report_version, layer, cohort, adapters", "matrix v1"),
            _source("Integrated matrix release", paths["integrated"], integrated_hash,
                    "schema, release_version, layer, identity, adapters", "release v1"),
            _source("Browser regressions", paths["browser"], browser_hash,
                    "artifact_version, cases", "browser regression v1"), ""]

    out += ["## Evaluation layers", "", "### Held-out policy", ""]
    policy_layer = policy_layer_raw if policy_valid else {}
    imported = policy.get("results_imported") is True if policy_valid else False
    out.append(f"Evidence reference status: `{_safe(policy_layer.get('status', 'unavailable'))}`. "
               f"Results imported: `{str(imported).lower()}`.")
    if not imported:
        out.append("Policy metrics: unavailable; the referenced evidence report does not import result metrics.")
    else:
        summary = policy.get("summary") if isinstance(policy.get("summary"), Mapping) else {}
        policy_adapters = summary.get("adapters")
        if isinstance(policy_adapters, Mapping):
            evidence_level = _safe(policy_layer.get("evidence_level", summary.get("evidence_level", "unspecified")))
            out.append(f"Policy evidence level: `{evidence_level}`. Aggregate metrics below are imported from the cited summary; they are not scenario-level records.")
            for adapter in ADAPTERS:
                arm = policy_adapters.get(adapter, {})
                arm = arm if isinstance(arm, Mapping) else {}
                out += ["", f"#### `{adapter}` — summary-level", ""]
                out.extend(_metric_lines(arm))
        else:
            out.append("Policy metrics (allowlisted fields only):")
            out.extend(_metric_lines(summary))
        reason = policy_layer.get("status_reason")
        if reason:
            out.append(f"Policy evidence limitation: {_safe(reason)}")
    artifacts = policy.get("evidence_artifacts", []) if policy_valid else []
    if artifacts:
        out.append("Evidence artifact digests:")
        for item in artifacts:
            if isinstance(item, Mapping):
                out.append(f"- Evidence SHA-256 `{item.get('sha256', 'unavailable')}`")
    out += ["", "### Controller-only", ""]
    if controller_valid:
        cohort = controller.get("cohort", {})
        cohort = cohort if isinstance(cohort, Mapping) else {}
        out.append(f"Cohort: `{_safe(controller.get('cohort_id', 'unavailable'))}`; scenarios: `{cohort.get('scenario_count', 'unavailable')}`.")
        out.append("This measures controller behavior without model participation.")
        for adapter in ADAPTERS:
            arm = controller.get("adapters", {}).get(adapter, {})
            metrics = arm.get("metrics", {}) if isinstance(arm, Mapping) else {}
            out += ["", f"#### `{adapter}` — `{_safe(arm.get('status', 'unavailable'))}`", ""]
            out.extend(_metric_lines(metrics))
            out.append(f"Evidence: `adapters.{adapter}.metrics`; manifest hash `{arm.get('run_manifest', {}).get('sha256', 'unavailable')}`.")
    else:
        out.append("Controller-only metrics: unavailable; a compatible version 1 controller-only matrix artifact was not found.")

    out += ["", "### Integrated", ""]
    if integrated_valid:
        identity = integrated.get("identity", {})
        settings = integrated.get("generation_settings", {})
        out.append(f"Base model revision: `{_safe(identity.get('base_model_revision', 'unavailable'))}`; "
                   f"settings: `{json.dumps(_safe(settings), sort_keys=True)}`.")
        for adapter in ADAPTERS:
            arm = integrated.get("adapters", {}).get(adapter, {})
            metrics = arm.get("metrics", {}) if isinstance(arm, Mapping) else {}
            provenance = arm.get("runtime_provenance") or {}
            hardware = (arm.get("run_manifest", {}).get("record", {}).get("hardware", {})
                       if isinstance(arm, Mapping) else {})
            complete = arm.get("layer") == "integrated" and arm.get("status") == "complete" and provenance.get("complete") is True
            out += ["", f"#### `{adapter}` — `{arm.get('status', 'unavailable')}`", "",
                    f"Model participation turns: `{arm.get('model_participation', {}).get('turns', 'unavailable')}`; "
                    f"device: `{hardware.get('device', 'unavailable')}`; precision: `{hardware.get('precision', 'unavailable')}`; "
                    f"peak memory bytes: `{hardware.get('peak_memory_bytes', 'unavailable')}`."]
            out.extend(_metric_lines(metrics if complete else {}))
            out.append(f"Evidence fields: `adapters.{adapter}.metrics`, `runtime_provenance`, `run_manifest.record.hardware`; "
                       f"arm fingerprint `{arm.get('fingerprint', 'unavailable')}`.")
    else:
        out.append("Integrated metrics: unavailable. No complete integrated release artifact exists; controller-only results do not substitute for integrated evaluation.")

    out += ["", "## Browser regressions", ""]
    cases = browser.get("cases", []) if isinstance(browser, Mapping) and browser.get("artifact_version") == 1 else []
    if cases:
        for case in cases:
            if isinstance(case, Mapping):
                out.append(f"- `{_safe(case.get('case_id', 'unknown'))}`: {_safe(case.get('coverage', 'coverage unavailable'))}")
    else:
        out.append("Browser regression evidence unavailable.")

    out += ["", "## Interpretation and limitations", "",
            "Judge scores are judge-dependent labels and are reported only when a metric artifact supplies a value; no score is inferred.",
            "Fallback rates and reason distributions, sample sizes, Wilson intervals, latency, token counts, and memory remain unavailable whenever their source field is absent.",
            "The simulator is local and simulated; these artifacts do not establish performance against live airline inventory or booking systems.",
            "Regression cases document browser behaviors; they are not population estimates and do not imply an integrated model run.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY_DEFAULT)
    parser.add_argument("--controller", type=Path, default=CONTROLLER_DEFAULT)
    parser.add_argument("--integrated", type=Path, default=INTEGRATED_DEFAULT)
    parser.add_argument("--browser", type=Path, default=BROWSER_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(argv)
    args.output.write_text(render_release_report(policy_path=args.policy, controller_path=args.controller,
                         integrated_path=args.integrated, browser_path=args.browser), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

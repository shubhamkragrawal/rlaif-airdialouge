"""Render an artifact-backed release report without calculating or guessing metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    "peak_memory_mb", "judge_score",
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
    if not isinstance(ids, list) or len(ids) != 50 or cohort.get("scenario_count") != 50:
        return False
    arms = integrated.get("adapters")
    if not isinstance(arms, Mapping) or set(arms) != set(ADAPTERS):
        return False
    for name in ADAPTERS:
        arm = arms[name]
        provenance = arm.get("runtime_provenance") or {}
        participation = arm.get("model_participation") or {}
        manifest = arm.get("run_manifest") or {}
        metrics = arm.get("metrics") or {}
        hardware = manifest.get("record", {}).get("hardware", {}) if isinstance(manifest, Mapping) else {}
        provenance_payload = ({key: value for key, value in provenance.items() if key != "sha256"}
                              if isinstance(provenance, Mapping) else {})
        provenance_hash_valid = provenance.get("sha256") == hashlib.sha256(json.dumps(
            provenance_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()
        run_hash_valid = (isinstance(manifest.get("record"), Mapping)
                          and manifest.get("sha256") == hashlib.sha256(json.dumps(
                              manifest["record"], sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode()).hexdigest())
        turns = arm.get("turn_manifests")
        turn_hashes_valid = isinstance(turns, list) and bool(turns)
        if turn_hashes_valid:
            for turn in turns:
                record = turn.get("record")
                evidence = turn.get("evidence")
                record_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                full_hash = hashlib.sha256(json.dumps({"record": record, "evidence": evidence},
                                                      sort_keys=True, separators=(",", ":"),
                                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
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
        if (arm.get("layer") != "integrated" or arm.get("status") != "complete"
                or arm.get("scenario_ids") != ids or provenance.get("complete") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("sha256", "")))
                or participation.get("required") is not True or participation.get("turns", 0) <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256", "")))
                or not hardware.get("device") or not hardware.get("precision")
                or hardware.get("peak_memory_bytes") is None or not quality_complete
                or not provenance_hash_valid or not run_hash_valid or not turn_hashes_valid):
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
        out.append("Policy metrics (allowlisted fields only):")
        out.extend(_metric_lines(policy.get("summary") if isinstance(policy.get("summary"), Mapping) else {}))
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

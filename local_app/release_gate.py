"""Read-only verification of supplied local release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .controller import ADAPTERS

SCHEMA = "rlaif.local_app.release_gate"
VERSION = 1
REQUIRED_CHECKS = ("local", "rebuild", "browser", "catalog", "clean_package", "ops")
RELEASE_METRICS = {
    "unsupported_transaction_claim_rate": 0,
    "malformed_output_rate": 0,
    "fallback_rate": 1,
}
REQUIRED_RATE_TARGETS = {
    "task_completion": 1.0,
    "valid_transitions": 1.0,
    "grounded_entity_precision": 1.0,
    "no_results_presentation_accuracy": 1.0,
    "repetition_rate": 0.0,
    "repetition_evidence": 0.0,
    "validation": 1.0,
}
REQUIRED_METRIC_FIELDS = (
    "scenario_count", "task_completion", "valid_transitions",
    "unsupported_transaction_claim_rate", "grounded_entity_precision",
    "no_results_presentation_accuracy", "malformed_output_rate",
    "repetition_rate", "repetition_evidence", "validation", "fallback_rate",
    "fallback_reason_distribution", "turns", "tokens", "latency_ms",
    "peak_memory_mb", "judge_score", "model_inference",
)
PROTECTED_STAGING = re.compile(
    r"(^|/)(?:\.env(?:\.[^/]+)?|secrets?)(?:/|$)|"
    r"(?:^|/)(?:local_models|rebuild/cloud|cloud-ledger|ledger|training-logs?|"
    r"checkpoints?|adapters?)(?:/|$)|(?:^|/)\.aws(?:/|$)|"
    r"(?:^|/)(?:airdialogue_sampled_5k\.jsonl|updated_spec\.md|dai_1_brief\.md)$|"
    r"\.(?:safetensors|pt|pth|ckpt|bin)$", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON evidence {path}: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite numeric rate")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _hardware_matches_provenance(record: Mapping[str, Any], provenance: Mapping[str, Any],
                                 summary: Any) -> bool:
    hardware = record.get("hardware")
    if not isinstance(hardware, Mapping):
        return False
    for field in ("device", "precision"):
        expected = provenance.get(field)
        if not isinstance(expected, str) or not expected or hardware.get(field) != expected:
            return False
    peak = provenance.get("peak_memory_bytes")
    observed_peak = hardware.get("peak_memory_bytes")
    if not (isinstance(peak, int) and not isinstance(peak, bool) and peak >= 0
            and isinstance(observed_peak, int) and not isinstance(observed_peak, bool)
            and observed_peak == peak):
        return False
    if "peak_process_rss_mb" in hardware:
        return (isinstance(summary, Mapping)
                and hardware["peak_process_rss_mb"] == summary.get("peak_process_rss_mb"))
    return True


def _verify_ref(root: Path, ref: Mapping[str, Any], label: str) -> tuple[Path, str]:
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise ValueError(f"{label}: evidence reference needs a path")
    path = (root / ref["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"{label}: evidence file is missing: {ref['path']}")
    digest = _sha256(path)
    if ref.get("sha256") != digest:
        raise ValueError(f"{label}: SHA-256 mismatch for {ref['path']}")
    return path, digest


def _verify_matrix(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    adapter_matrix = (isinstance(report, dict)
                      and report.get("schema") == "rlaif.local_app.adapter_matrix"
                      and report.get("report_version") == 1)
    release_matrix = (isinstance(report, dict)
                      and report.get("schema") == "rlaif.local_app.matrix_release"
                      and report.get("release_version") == 1)
    if (not (adapter_matrix or release_matrix) or report.get("mode") != "integrated"
            or report.get("layer") != "integrated"):
        raise ValueError("matrix report must be version 1 and integrated layer")
    if release_matrix:
        supplied_hash = report.get("sha256")
        payload = {key: value for key, value in report.items() if key != "sha256"}
        if supplied_hash != _canonical_sha256(payload):
            raise ValueError("matrix release hash does not match its contents")
        settings = report.get("generation_settings") or {}
        if (settings.get("start"), settings.get("count"), settings.get("temperature"),
                settings.get("max_new_tokens")) != (0, 50, 0.0, 160):
            raise ValueError("matrix release does not use the fixed release settings")
    arms = report.get("adapters")
    if not isinstance(arms, dict) or set(arms) != set(ADAPTERS):
        raise ValueError("matrix report must contain exactly the four supported adapter arms")
    cohort = report.get("cohort") or {}
    ids = cohort.get("scenario_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("matrix report has an invalid scenario cohort")
    if cohort.get("scenario_count") != len(ids):
        raise ValueError("matrix cohort count does not match its scenario IDs")
    if len(ids) != 50:
        raise ValueError("integrated release evidence must contain the fixed 50-scenario cohort")
    cohort_settings = cohort.get("settings") or {}
    common_controller = common_schema = common_model = common_revision = common_generation = None
    for name in ADAPTERS:
        arm = arms[name]
        if arm.get("status") != "complete" or arm.get("layer") != "integrated":
            raise ValueError(f"{name}: integrated arm is not complete")
        if arm.get("scenario_ids") != ids:
            raise ValueError(f"{name}: scenario cohort differs from matrix")
        if arm.get("cohort_id") != report.get("cohort_id"):
            raise ValueError(f"{name}: cohort ID mismatch")
        fingerprints = arm.get("fingerprints") or {}
        if not re.fullmatch(r"[0-9a-f]{64}", str(fingerprints.get("controller_sha256", ""))):
            raise ValueError(f"{name}: missing controller digest")
        schema = fingerprints.get("outcome_schema") or {}
        if (schema.get("name") != "rlaif.local_app.turn_outcome" or schema.get("version") != 1
                or not re.fullmatch(r"[0-9a-f]{64}", str(schema.get("sha256", "")))):
            raise ValueError(f"{name}: missing outcome schema fingerprint")
        schema_identity = (schema["name"], schema["version"], schema["sha256"])
        if common_controller is None:
            common_controller, common_schema = fingerprints["controller_sha256"], schema_identity
        elif fingerprints["controller_sha256"] != common_controller or schema_identity != common_schema:
            raise ValueError(f"{name}: controller or outcome schema fingerprint differs across arms")
        if not re.fullmatch(r"[0-9a-f]{64}", str(fingerprints.get("runtime_provenance_sha256", ""))):
            raise ValueError(f"{name}: missing runtime provenance fingerprint")
        prov = arm.get("runtime_provenance") or {}
        status = arm.get("runtime_status") or {}
        if (prov.get("complete") is not True or prov.get("sha256") != fingerprints["runtime_provenance_sha256"]
                or not prov.get("adapter_sha256") or not prov.get("base_model")
                or not prov.get("base_model_revision") or not prov.get("generation_settings")
                or not prov.get("device") or not prov.get("precision")
                or not isinstance(prov.get("peak_memory_bytes"), int)
                or isinstance(prov.get("peak_memory_bytes"), bool)
                or prov.get("peak_memory_bytes", -1) < 0):
            raise ValueError(f"{name}: runtime provenance is incomplete or mismatched")
        provenance_payload = {key: value for key, value in prov.items() if key != "sha256"}
        if _canonical_sha256(provenance_payload) != prov["sha256"]:
            raise ValueError(f"{name}: runtime provenance hash does not match its contents")
        generation = prov["generation_settings"]
        if (prov["base_model"] != cohort_settings.get("base_model")
                or prov["base_model_revision"] != cohort_settings.get("revision")
                or generation.get("temperature") != cohort_settings.get("temperature")
                or generation.get("max_new_tokens") != cohort_settings.get("max_new_tokens")):
            raise ValueError(f"{name}: runtime provenance does not match integrated cohort settings")
        identity = (prov["base_model"], prov["base_model_revision"], json.dumps(generation, sort_keys=True))
        if common_model is None:
            common_model, common_revision, common_generation = identity
        elif identity != (common_model, common_revision, common_generation):
            raise ValueError(f"{name}: model revision or generation settings differ across arms")
        if status.get("lifecycle_status") != "ready" or status.get("mode") != "local_model" or status.get("available") is not True:
            raise ValueError(f"{name}: runtime status is not ready local_model")
        participation = arm.get("model_participation") or {}
        attempted = participation.get("attempted_turns", 0)
        generated = participation.get("generated_turns", participation.get("turns", 0))
        accepted = participation.get("accepted_turns", 0)
        fallback_count = participation.get("fallback_turns", 0)
        counts = (attempted, generated, accepted, fallback_count)
        if (participation.get("required") is not True
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for value in counts)
                or attempted <= 0 or generated <= 0 or generated > attempted
                or accepted > generated or fallback_count > attempted
                or participation.get("turns") != generated):
            raise ValueError(f"{name}: no real model participation")
        if not isinstance(arm.get("run_manifest"), dict) or not arm["run_manifest"].get("sha256"):
            raise ValueError(f"{name}: missing run manifest")
        run_manifest = arm["run_manifest"]
        run_record = run_manifest.get("record")
        if not isinstance(run_record, dict) or _canonical_sha256(run_record) != run_manifest["sha256"]:
            raise ValueError(f"{name}: run manifest hash does not match its record")
        if not _hardware_matches_provenance(run_record, prov, arm.get("summary")):
            raise ValueError(f"{name}: run manifest hardware does not match runtime provenance")
        turns = arm.get("turn_manifests")
        if not isinstance(turns, list) or not turns or any(not item.get("sha256") for item in turns):
            raise ValueError(f"{name}: missing turn manifest hashes")
        scenario_turn_counts = {scenario_id: 0 for scenario_id in ids}
        observed_model_counts = {"attempted_turns": 0, "generated_turns": 0,
                                 "accepted_turns": 0, "fallback_turns": 0}
        observed_token_totals = {"input": 0, "output": 0}
        observed_token_values = {"input": False, "output": False}
        for turn in turns:
            if not isinstance(turn, Mapping) or not isinstance(turn.get("record"), Mapping):
                raise ValueError(f"{name}: malformed turn manifest")
            record_hash = _canonical_sha256(turn.get("record"))
            hashed = _canonical_sha256({"record": turn.get("record"), "evidence": turn.get("evidence")})
            if (hashed != turn["sha256"] or record_hash != turn.get("record_sha256")):
                raise ValueError(f"{name}: turn manifest hash does not match its record and evidence")
            evidence = turn.get("evidence") or {}
            if (not isinstance(evidence, Mapping)
                    or evidence.get("unsupported_transaction_claim") is not False
                    or evidence.get("malformed_output") is not False
                    or evidence.get("repetitive_output") is not False
                    or evidence.get("validation_result") != "valid"):
                raise ValueError(f"{name}: unsafe or invalid per-turn evidence")
            try:
                scenario_id = int(turn["record"].get("scenario_id"))
            except (TypeError, ValueError):
                raise ValueError(f"{name}: turn manifest has an invalid scenario ID") from None
            if (scenario_id not in scenario_turn_counts
                    or evidence.get("scenario_id") != scenario_id):
                raise ValueError(f"{name}: turn manifest scenario ID is outside the cohort or mismatched")
            scenario_turn_counts[scenario_id] += 1
            for count_name, evidence_name in (
                    ("attempted_turns", "model_attempted"),
                    ("generated_turns", "model_generated"),
                    ("accepted_turns", "model_accepted"),
                    ("fallback_turns", "model_fallback")):
                value = evidence.get(evidence_name)
                if not isinstance(value, bool):
                    raise ValueError(f"{name}: missing per-turn {evidence_name} evidence")
                observed_model_counts[count_name] += int(value)
            for token_name, evidence_name in (("input", "input_tokens"),
                                               ("output", "output_tokens")):
                value = evidence.get(evidence_name)
                if value is not None:
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0):
                        raise ValueError(f"{name}: invalid per-turn {evidence_name} evidence")
                    observed_token_totals[token_name] += value
                    observed_token_values[token_name] = True
            if evidence.get("transaction_claim") is True:
                transaction = evidence.get("authoritative_transaction_evidence")
                if (not isinstance(transaction, Mapping) or transaction.get("current_turn") is not True
                        or transaction.get("booking_status") != "confirmed"
                        or transaction.get("simulated") is not True
                        or not re.fullmatch(r"hold_[0-9a-f]{24}", str(transaction.get("hold_id", "")))
                        or not re.fullmatch(r"booking_[0-9a-f]{24}", str(transaction.get("booking_id", "")))
                        or (turn.get("record") or {}).get("transaction_evidence_ids")
                        != [transaction.get("hold_id"), transaction.get("booking_id")]):
                    raise ValueError(f"{name}: transaction claim lacks current-turn simulator evidence")
            for transition in ("valid_search_transition", "valid_booking_transition"):
                if evidence.get(transition) is False:
                    raise ValueError(f"{name}: invalid per-turn transition evidence")
            if evidence.get("expected_no_results") is True and evidence.get("presented_no_results") is not True:
                raise ValueError(f"{name}: no-results turn was not presented correctly")
        rows = arm.get("scenarios")
        if not isinstance(rows, list) or [row.get("scenario_id") for row in rows] != ids:
            raise ValueError(f"{name}: incomplete scenario results")
        for row in rows:
            expected_turns = row.get("metrics", {}).get("turn_count")
            scenario_id = row["scenario_id"]
            if (isinstance(expected_turns, bool) or not isinstance(expected_turns, int)
                    or expected_turns < 0 or scenario_turn_counts[scenario_id] != expected_turns):
                raise ValueError(f"{name}: retained turn manifests do not cover scenario {scenario_id}")
            scenario_metrics = row.get("metrics")
            normalized = row.get("normalized_outcome")
            success = scenario_metrics.get("task_success") if isinstance(scenario_metrics, Mapping) else None
            if (not isinstance(success, bool) or not isinstance(normalized, Mapping)
                    or normalized.get("task_success") is not success):
                raise ValueError(f"{name}: scenario task_success disagrees with normalized outcome")
        task_completion_metrics = arm.get("metrics") or {}
        completion = (task_completion_metrics.get("task_completion")
                      if isinstance(task_completion_metrics, Mapping) else None)
        successes = sum(row["metrics"]["task_success"] is True for row in rows)
        if (not isinstance(completion, Mapping)
                or isinstance(completion.get("successes"), bool)
                or not isinstance(completion.get("successes"), int)
                or completion.get("successes") != successes
                or completion.get("rate") != successes / len(ids)
                or task_completion_metrics.get("scenario_count") != len(ids)):
            raise ValueError(f"{name}: task completion aggregate does not match scenario outcomes")
        if observed_model_counts != {key: participation_value for key, participation_value in (
                ("attempted_turns", attempted), ("generated_turns", generated),
                ("accepted_turns", accepted), ("fallback_turns", fallback_count))}:
            raise ValueError(f"{name}: model inference totals do not match retained turn evidence")
        metrics = arm.get("metrics") or {}
        if metrics.get("scenario_count") != len(ids):
            raise ValueError(f"{name}: metric scenario count mismatch")
        missing_metrics = [field for field in REQUIRED_METRIC_FIELDS if field not in metrics]
        if missing_metrics:
            raise ValueError(f"{name}: missing release metrics: {', '.join(missing_metrics)}")
        for metric, maximum in RELEASE_METRICS.items():
            value = _finite_rate(metrics.get(metric), f"{name}: {metric}")
            if value > maximum:
                raise ValueError(f"{name}: release metric {metric} exceeds {maximum}")
        for field in ("grounded_entity_precision", "no_results_presentation_accuracy", "repetition_rate"):
            _finite_rate(metrics[field], f"{name}: {field}")
        for field in ("task_completion", "valid_transitions", "repetition_evidence", "validation"):
            value = metrics[field]
            if not isinstance(value, Mapping) or "rate" not in value:
                raise ValueError(f"{name}: {field} must include a rate")
            _finite_rate(value["rate"], f"{name}: {field}.rate")
        for field, target in REQUIRED_RATE_TARGETS.items():
            value = metrics[field]
            rate = value["rate"] if isinstance(value, Mapping) else value
            if _finite_rate(rate, f"{name}: {field}.rate") != target:
                raise ValueError(f"{name}: {field} must equal release target {target}")
        if not isinstance(metrics["fallback_reason_distribution"], Mapping):
            raise ValueError(f"{name}: fallback_reason_distribution must be an object")
        inference = metrics["model_inference"]
        if (not isinstance(inference, Mapping)
                or inference.get("attempted_turns") != attempted
                or inference.get("generated_turns") != generated
                or inference.get("accepted_turns") != accepted
                or inference.get("fallback_turns") != fallback_count
                or not isinstance(inference.get("fallback_reasons"), list)):
            raise ValueError(f"{name}: model inference telemetry is missing or inconsistent")
        token_metrics = metrics["tokens"]
        for token_name in ("input", "output"):
            retained_total = (observed_token_totals[token_name]
                              if observed_token_values[token_name] else None)
            if token_metrics.get(token_name) != retained_total:
                raise ValueError(f"{name}: {token_name} token total does not match retained turn usage")
        for field in ("turns", "tokens", "latency_ms", "judge_score"):
            if not isinstance(metrics[field], Mapping):
                raise ValueError(f"{name}: {field} must be an object")
        peak = metrics["peak_memory_mb"]
        if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak < 0:
            raise ValueError(f"{name}: peak_memory_mb must be finite and non-negative")
    if release_matrix:
        identity = report.get("identity") or {}
        if (identity.get("base_model") != cohort_settings.get("base_model")
                or identity.get("base_model_revision") != cohort_settings.get("revision")
                or identity.get("corpus_sha256") is None
                or identity.get("controller_sha256") != common_controller):
            raise ValueError("matrix release identity does not match the verified arms")
        adapter_hashes = identity.get("adapter_sha256") or {}
        if any(adapter_hashes.get(name) != arms[name]["runtime_provenance"]["adapter_sha256"]
               for name in ADAPTERS):
            raise ValueError("matrix release adapter identity does not match the verified arms")
    return report


def _staged_exclusions(root: Path) -> list[str]:
    # Evidence may live outside the checkout; inspect the owning checkout.
    checkout = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(["git", "-C", str(checkout), "diff", "--cached", "--name-only",
                                 "--diff-filter=ACMR"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect staged paths: {exc}") from exc
    return sorted(line for line in result.stdout.splitlines() if PROTECTED_STAGING.search(line))


def evaluate(evidence_path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    evidence_path = Path(evidence_path).resolve()
    root_path = Path(root).resolve() if root else evidence_path.parent.resolve()
    bundle = _read_json(evidence_path)
    if not isinstance(bundle, dict) or bundle.get("schema") != "rlaif.local_app.release_evidence" or bundle.get("version") != 1:
        raise ValueError("unsupported release evidence bundle")
    refs: dict[str, dict[str, str]] = {}
    checks = bundle.get("checks") or {}
    failures = []
    for name in REQUIRED_CHECKS:
        item = checks.get(name)
        try:
            path, digest = _verify_ref(root_path, item or {}, name)
            record = _read_json(path)
            if (not isinstance(record, dict) or record.get("status") != "passed"
                    or record.get("exit_code") != 0 or not record.get("command")):
                raise ValueError(f"{name}: evidence must record passed, exit_code 0, and command")
            refs[name] = {"path": str(path.relative_to(root_path)), "sha256": digest}
        except (ValueError, OSError) as exc:
            failures.append(str(exc))
    matrix = None
    try:
        matrix_path, matrix_digest = _verify_ref(root_path, bundle.get("integrated_matrix") or {}, "integrated_matrix")
        matrix = _verify_matrix(matrix_path)
        refs["integrated_matrix"] = {"path": str(matrix_path.relative_to(root_path)), "sha256": matrix_digest}
    except (ValueError, OSError) as exc:
        failures.append(str(exc))
    excluded = _staged_exclusions(root_path)
    if excluded:
        failures.append("protected/local-model/cloud-ledger paths are staged: " + ", ".join(excluded))
    return {"schema": SCHEMA, "release_version": VERSION,
            "status": "pass" if not failures else "fail", "created_at": datetime.now(timezone.utc).isoformat(),
            "evidence_refs": refs, "integrated_matrix_sha256": refs.get("integrated_matrix", {}).get("sha256"),
            "adapters": list(ADAPTERS) if matrix else [],
            "scenario_count": matrix.get("cohort", {}).get("scenario_count") if matrix else None,
            "staging_exclusions": {"status": "pass" if not excluded else "fail", "excluded_paths": excluded},
            "failures": failures}


def write_versioned(report: Mapping[str, Any], output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"release-gate-v{VERSION}.json"
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="JSON bundle with checks and hash-pinned matrix reference")
    parser.add_argument("--root", type=Path, help="evidence root (default: bundle directory)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/release"))
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.evidence, root=args.root)
        output = write_versioned(report, args.output_dir)
    except ValueError as exc:
        print(f"release gate input error: {exc}")
        return 2
    print(f"release gate {report['status']}: {output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

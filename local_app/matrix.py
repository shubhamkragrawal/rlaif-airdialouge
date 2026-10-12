"""Reproducible local matrix evaluation for the four supported adapters.

This module composes :mod:`local_app.evaluation`; it deliberately leaves that
single-adapter evaluator unchanged. No cloud runtime or job submission exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from . import evaluation
from .controller import ADAPTERS, ConversationController
from .evaluation import DEFAULT_CORPUS, load_scenarios
from .evaluation_metrics import aggregate_evaluation_metrics
from .guardrails import detect_transaction_claim
from .observability import (RunRecord, TurnRecord, canonical_json, record_json,
                            record_sha256)
from .outcomes import Intent, OUTCOME_SCHEMA, OUTCOME_SCHEMA_VERSION
from .output_validation import validate_output

MATRIX_REPORT_VERSION = 1
BASELINE_SCHEMA = "rlaif.local_app.matrix_baseline"
BASELINE_VERSION = 1
RELEASE_SCHEMA = "rlaif.local_app.matrix_release"
RELEASE_VERSION = 1
RELEASE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "matrix" / "releases" / "v1"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalized(row: Mapping[str, Any]) -> dict[str, Any]:
    """Stable comparable outcomes; excludes timings, memory, and model telemetry."""
    return {
        "task_success": bool(row["task_success"]),
        "completed": bool(row["completed"]),
        "expected_status": row["expected_status"],
        "booked_flight_id": row.get("booked_flight_id"),
        "search_calls": int(row["search_calls"]),
        "search_valid_calls": int(row["search_valid_calls"]),
        "booking_calls": int(row["booking_calls"]),
        "booking_valid_calls": int(row["booking_valid_calls"]),
        "tool_calls": int(row["tool_calls"]),
        "tool_valid_calls": int(row["tool_valid_calls"]),
        "rejections": int(row["rejections"]),
        "fallback_reasons": sorted(row.get("fallback_reasons", [])),
    }


def _controller_version() -> str:
    return evaluation._sha256_path(Path(__file__).with_name("controller.py")) or "unknown"


def _turn_evidence(observed: Sequence[Mapping[str, Any]], scenarios: Sequence[Any],
                   *, run_id: str, adapter: str, controller_version: str,
                   generation_settings: Mapping[str, Any], adapter_metadata: Mapping[str, Any],
                   scenario_rows: Mapping[int, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build immutable turn records and metric evidence from public handle results."""
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    evidence_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for captured in observed:
        sid = captured["scenario_id"]
        scenario = by_id[sid]
        result = captured["result"]
        response = str(result.get("text", ""))
        results = result.get("results") if isinstance(result.get("results"), list) else []
        message = scenario.messages[min(captured["turn_index"], len(scenario.messages) - 1)]
        is_search = bool(re.search(
            r"\b(?:find|search|look for|show)\b.*\bflights?\b", message, re.I))
        outcome = captured.get("outcome")
        is_booking_request = (bool(re.search(
            r"\b(?:book|reserve|confirm|proceed\s+with|continue\s+with|go\s+ahead\s+with)\b",
            message, re.I)) or getattr(outcome, "intent", None) is Intent.BOOK)
        booking_after = captured.get("booking_after")
        booking_before = captured.get("booking_before")
        booking_transition = booking_after is not None and booking_after != booking_before
        booking_evidence = (outcome.booking_result
                            if outcome is not None and isinstance(outcome.booking_result, Mapping)
                            else None)
        hold_evidence = (outcome.hold_result
                         if outcome is not None and isinstance(outcome.hold_result, Mapping)
                         else None)
        authoritative_transaction_evidence = None
        if booking_evidence is not None and hold_evidence is not None:
            authoritative_transaction_evidence = {
                "hold_id": hold_evidence.get("hold_id"),
                "booking_id": booking_evidence.get("booking_id"),
                "booking_status": booking_evidence.get("status"),
                "flight_id": (booking_evidence.get("flight") or {}).get("flight_number"),
                "simulated": booking_evidence.get("simulated") is True,
                "current_turn": True,
            }
        claimed_entities = set(re.findall(r"\b\d{4,5}\b", response))
        claimed_entities.update(re.findall(r"\b[A-Z]{3}\b", response))
        claimed_entities.update(re.findall(r"\$\s?\d+(?:\.\d{2})?", response))
        grounded_entities: set[str] = set()
        for flight in results:
            if not isinstance(flight, Mapping):
                continue
            for key in ("flight_number", "origin", "departure_airport", "destination",
                        "return_airport", "price", "airline"):
                value = flight.get(key)
                if value is not None:
                    text = str(value)
                    grounded_entities.add(text)
                    if key == "flight_number":
                        grounded_entities.add(f"flight {text}")
                    if key == "price":
                        grounded_entities.add(f"${text}")
        grounded_entities.update(str(item) for item in claimed_entities
                                 if str(item).casefold() in {x.casefold() for x in grounded_entities})
        transaction_claim = detect_transaction_claim(response)
        unsupported_claim = transaction_claim and authoritative_transaction_evidence is None
        fallback_reason = result.get("fallback_reason") or result.get("validation_reason")
        model_fallback_reason = result.get("model_fallback_reason")
        model_attempted = bool(result.get("model_attempted"))
        model_generated = bool(result.get("model_generated", result.get("model_used")))
        model_accepted = bool(result.get("model_accepted", result.get("model_used")))
        model_fallback = bool(result.get("model_fallback", model_attempted and not model_accepted))
        validation_reason = result.get("final_validation_reason")
        if "final_validation_reason" not in result and outcome is not None:
            final_errors = validate_output(response, outcome)
            validation_reason = final_errors[0].code if final_errors else None
        malformed = bool(re.search(r"^\s*(?:\{|\[|None\b|\(\))", response)) or (
            isinstance(validation_reason, str) and "malformed" in validation_reason.casefold())
        expected_no_results = scenario.expected_status == "no_flight" if is_search else None
        presented_no_results = bool(re.search(r"\bno (?:matching )?flights?\b|couldn't find a matching flight", response, re.I))
        normalized_entities = sorted(claimed_entities)
        grounded = sorted(grounded_entities)
        tokens = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
        fallback = model_fallback or bool(fallback_reason)
        latency = float(captured["latency_ms"])
        evidence = {
            "scenario_id": sid,
            "valid_search_transition": (bool(results or scenario.expected_status == "no_flight")
                                        if is_search else None),
            "valid_booking_transition": (authoritative_transaction_evidence is not None
                                         if is_booking_request else None),
            "transaction_claim": transaction_claim,
            "unsupported_transaction_claim": bool(unsupported_claim),
            "authoritative_transaction_evidence": authoritative_transaction_evidence,
            "claimed_entities": normalized_entities, "grounded_entities": grounded,
            "expected_no_results": expected_no_results,
            "presented_no_results": presented_no_results if is_search else None,
            "malformed_output": malformed, "response": response,
            "validation_result": "failed" if validation_reason else "valid",
            "validation_reason": validation_reason,
            "model_attempted": model_attempted,
            "model_generated": model_generated,
            "model_accepted": model_accepted,
            "model_fallback": model_fallback,
            "model_fallback_reason": model_fallback_reason,
            "repetitive_output": validation_reason == "repeated_sentence",
            "fallback": fallback, "fallback_reason": fallback_reason,
            "input_tokens": tokens.get("input_tokens"),
            "output_tokens": tokens.get("output_tokens"),
            # The non-streaming evaluator has no token arrival event.
            "time_to_first_token_ms": None, "total_latency_ms": latency,
            "peak_memory_mb": scenario_rows[sid].get("peak_process_rss_mb"),
        }
        safe_result = {"model_used": bool(result.get("model_used")),
                       "model_attempted": model_attempted,
                       "model_generated": model_generated,
                       "model_accepted": model_accepted,
                       "model_fallback": model_fallback,
                       "result_count": len(results), "simulated": result.get("simulated") is True}
        record = TurnRecord(
            run_id=run_id, scenario_id=str(sid), controller_version=controller_version,
            adapter={"name": adapter, **dict(adapter_metadata)},
            tool_outputs=(safe_result, {"turn_evidence": evidence}),
            transaction_evidence_ids=(
                (str(authoritative_transaction_evidence["hold_id"]),
                 str(authoritative_transaction_evidence["booking_id"]))
                if authoritative_transaction_evidence is not None else ()),
            validation_result=("failed" if validation_reason else "valid"),
            fallback_reason=(str(model_fallback_reason or fallback_reason)
                             if model_fallback_reason or fallback_reason else None),
            input_tokens=tokens.get("input_tokens"), output_tokens=tokens.get("output_tokens"),
            latency_ms=latency,
            peak_memory_bytes=(int(float(evidence["peak_memory_mb"]) * 1024 * 1024)
                               if evidence["peak_memory_mb"] is not None else None),
            final_outcome=("booking" if authoritative_transaction_evidence is not None else
                           "no_results" if is_search and not results else "response"),
            generation_settings=generation_settings,
        )
        record_payload = json.loads(record_json(record))
        safe_evidence = json.loads(canonical_json(record.tool_outputs[1]["turn_evidence"]))
        evidence_rows.append(safe_evidence)
        manifests.append({"record": record_payload, "evidence": safe_evidence,
                          "record_sha256": record_sha256(record),
                          "sha256": _canonical_hash({"record": record_payload,
                                                     "evidence": safe_evidence})})
    return evidence_rows, manifests


def _runtime_snapshot(manager: Any, adapter: str,
                      expected_settings: Mapping[str, Any],
                      expected_adapter_sha256: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    status_fn = getattr(manager, "status", None) or getattr(manager, "adapter_status", None)
    provenance_fn = getattr(manager, "provenance", None)
    raw_status = status_fn(adapter) if callable(status_fn) else {}
    raw_provenance = provenance_fn(adapter) if callable(provenance_fn) else {}
    status = dict(raw_status) if isinstance(raw_status, Mapping) else {}
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
    lifecycle_status = status.get("lifecycle_status", status.get("status", "unavailable"))
    normalized_status = {
        "adapter": status.get("adapter", adapter),
        "lifecycle_status": lifecycle_status,
        "mode": status.get("mode", "local_model" if lifecycle_status in {
            "available", "loading", "ready"} else "deterministic_fallback"),
        "available": status.get("available", lifecycle_status in {"available", "ready"}),
        "provenance_status": status.get("provenance_status", {
            "ready": "complete", "available": "partial", "loading": "pending",
            "incompatible": "error", "configured": "unavailable",
            "unavailable": "unavailable",
        }.get(lifecycle_status, "unavailable")),
    }
    provenance_lifecycle = provenance.get(
        "lifecycle_status", provenance.get("status", normalized_status["lifecycle_status"]))
    normalized_provenance = {
        "adapter": provenance.get("adapter", adapter),
        "lifecycle_status": provenance_lifecycle,
        "mode": provenance.get("mode", normalized_status["mode"]),
        "available": provenance.get("available", normalized_status["available"]),
        "provenance_status": provenance.get(
            "provenance_status", "complete" if provenance_lifecycle == "ready" else "partial"),
        "adapter_sha256": provenance.get("adapter_sha256"),
        "base_model": provenance.get("base_model"),
        "base_model_revision": provenance.get("base_model_revision", provenance.get("revision")),
        "generation_settings": provenance.get("generation_settings"),
        "device": provenance.get("device"),
        "precision": provenance.get("precision", provenance.get("dtype")),
        "peak_memory_bytes": provenance.get("peak_memory_bytes"),
    }
    valid_hash = isinstance(normalized_provenance["adapter_sha256"], str) and bool(
        re.fullmatch(r"[0-9a-f]{64}", normalized_provenance["adapter_sha256"]))
    runtime_settings = normalized_provenance["generation_settings"]
    settings_match = isinstance(runtime_settings, Mapping) and all(
        runtime_settings.get(key) == expected_settings.get(key)
        for key in ("max_new_tokens", "temperature"))
    incomplete_fields = []
    checks = (
        (normalized_status["adapter"] == adapter, "runtime_status.adapter"),
        (normalized_provenance["adapter"] == adapter, "runtime_provenance.adapter"),
        (normalized_status["lifecycle_status"] == "ready", "runtime_status.lifecycle_status"),
        (normalized_status["mode"] == "local_model", "runtime_status.mode"),
        (normalized_status["available"] is True, "runtime_status.available"),
        (normalized_status["provenance_status"] == "complete", "runtime_status.provenance_status"),
        (normalized_provenance["lifecycle_status"] == "ready", "runtime_provenance.lifecycle_status"),
        (normalized_provenance["mode"] == "local_model", "runtime_provenance.mode"),
        (normalized_provenance["available"] is True, "runtime_provenance.available"),
        (normalized_provenance["provenance_status"] == "complete", "runtime_provenance.provenance_status"),
        (valid_hash, "runtime_provenance.adapter_sha256"),
        (normalized_provenance["adapter_sha256"] == expected_adapter_sha256,
         "runtime_provenance.adapter_sha256_match"),
        (bool(normalized_provenance["base_model"]), "runtime_provenance.base_model"),
        (bool(normalized_provenance["base_model_revision"]), "runtime_provenance.base_model_revision"),
        (isinstance(runtime_settings, Mapping), "runtime_provenance.generation_settings"),
        (settings_match, "runtime_provenance.generation_settings_match"),
        (bool(normalized_provenance["device"]), "runtime_provenance.device"),
        (bool(normalized_provenance["precision"]), "runtime_provenance.precision"),
        (isinstance(normalized_provenance["peak_memory_bytes"], int)
         and not isinstance(normalized_provenance["peak_memory_bytes"], bool)
         and normalized_provenance["peak_memory_bytes"] >= 0,
         "runtime_provenance.peak_memory_bytes"),
    )
    incomplete_fields.extend(name for valid, name in checks if not valid)
    normalized_provenance["incomplete_fields"] = incomplete_fields
    normalized_provenance["complete"] = not incomplete_fields
    normalized_provenance["sha256"] = _canonical_hash(normalized_provenance)
    return normalized_status, normalized_provenance


def _adapter_artifact(manager: Any, adapter: str) -> dict[str, Any]:
    config = manager.config
    path_value = (config.adapter_paths or {}).get(adapter)
    path = Path(path_value) if path_value else None
    provenance_fn = getattr(manager, "provenance", None)
    raw = provenance_fn(adapter) if callable(provenance_fn) else {}
    runtime_hash = raw.get("adapter_sha256") if isinstance(raw, Mapping) else None
    if not isinstance(runtime_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", runtime_hash):
        runtime_hash = evaluation._sha256_path(path) if path else None
    return {"artifact": str(path) if path else None, "sha256": runtime_hash}


def run_matrix(
    *, mode: str = "controller_only", corpus_path: str | Path = DEFAULT_CORPUS,
    count: int = 50, start: int = 0, scenarios: Sequence[Any] | None = None,
    runtime_manager: Any = None, temperature: float | None = None,
    max_new_tokens: int | None = None, adapters: Sequence[str] = ADAPTERS,
) -> dict[str, Any]:
    """Run all four arms over one immutable scenario list and shared settings."""
    if mode not in {"controller_only", "integrated"}:
        raise ValueError("mode must be controller_only or integrated")
    if mode == "integrated" and runtime_manager is None:
        raise ValueError("integrated mode requires a local runtime manager")
    if mode == "controller_only" and runtime_manager is not None:
        raise ValueError("controller_only forbids model loading")
    selected_adapters = tuple(adapters)
    if not selected_adapters or any(adapter not in ADAPTERS for adapter in selected_adapters):
        raise ValueError("adapters must be a non-empty subset of supported adapters")
    cohort = list(scenarios) if scenarios is not None else load_scenarios(
        corpus_path, count=count, start=start)
    cohort.sort(key=lambda item: item.scenario_id)
    ids = [item.scenario_id for item in cohort]
    if not cohort or len(ids) != len(set(ids)):
        raise ValueError("scenario cohort must be non-empty and contain unique IDs")
    cohort_payload = [{"scenario_id": s.scenario_id, "messages": list(s.messages),
                       "expected_status": s.expected_status,
                       "expected_flight_ids": list(s.expected_flight_ids)} for s in cohort]
    corpus_hash = evaluation._sha256_path(Path(corpus_path))
    if mode == "integrated":
        temperature = (runtime_manager.config.temperature if temperature is None else temperature)
        max_new_tokens = (runtime_manager.config.max_new_tokens if max_new_tokens is None
                          else max_new_tokens)
    settings = ({"mode": "deterministic_controller"} if mode == "controller_only" else {
        "mode": "model", "base_model": runtime_manager.config.base_model,
        "revision": runtime_manager.config.revision,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    })
    cohort_id = "matrix-" + _canonical_hash({"scenarios": cohort_payload,
                                               "corpus_sha256": corpus_hash,
                                               "settings": settings})[:20]
    controller_version = _controller_version()
    outcome_schema_sha256 = evaluation._sha256_path(Path(__file__).with_name("outcomes.py"))
    arms = {}
    for adapter in selected_adapters:
        artifact = (_adapter_artifact(runtime_manager, adapter) if mode == "integrated"
                    else {})
        observed: list[dict[str, Any]] = []
        outcomes: dict[tuple[int, int], Any] = {}
        original_handle = ConversationController.handle
        original_respond = ConversationController._respond

        def capture_respond(controller, state, message, parsed=None):
            outcome = original_respond(controller, state, message, parsed)
            outcomes[(controller.kb.scenario_id, state.turn_count)] = outcome
            return outcome

        def capture_handle(controller, state, message):
            before = _canonical_hash(state.booking) if state.booking is not None else None
            started_at = time.perf_counter()
            try:
                result = original_handle(controller, state, message)
            except (TypeError, ValueError) as exc:
                observed.append({
                    "scenario_id": controller.kb.scenario_id,
                    "turn_index": max(0, state.turn_count - 1),
                    "result": {"text": "", "results": [], "simulated": True,
                               "model_used": False, "usage": None,
                               "fallback_reason": f"exception:{type(exc).__name__}",
                               "validation_reason": f"exception:{type(exc).__name__}"},
                    "booking_before": before,
                    "booking_after": _canonical_hash(state.booking) if state.booking is not None else None,
                    "outcome": outcomes.pop((controller.kb.scenario_id, state.turn_count), None),
                    "latency_ms": max(0.0, (time.perf_counter() - started_at) * 1000),
                })
                raise
            observed.append({"scenario_id": controller.kb.scenario_id,
                             "turn_index": max(0, int(result.get("turn", state.turn_count)) - 1),
                             "result": result, "booking_before": before,
                             "booking_after": _canonical_hash(state.booking) if state.booking is not None else None,
                             "outcome": outcomes.pop((controller.kb.scenario_id, state.turn_count), None),
                             "latency_ms": max(0.0, (time.perf_counter() - started_at) * 1000)})
            return result

        try:
            with (patch.object(ConversationController, "_respond", capture_respond),
                  patch.object(ConversationController, "handle", capture_handle)):
                report = evaluation.build_report(
                    cohort, adapter=adapter, corpus_path=corpus_path,
                    model_loader=runtime_manager.get if mode == "integrated" else None,
                    layer=mode, adapter_artifact=artifact, generation_settings=settings,
                )
            if report["scenario_ids"] != ids:
                raise ValueError(f"adapter {adapter} evaluated a different scenario cohort")
            contract = report["evaluation_layer"]
            if contract.get("layer") != mode:
                raise ValueError(f"adapter {adapter} evaluation layer does not match requested mode")
            rows_by_id = {row["scenario_id"]: row for row in report["scenarios"]}
            runtime_status = runtime_provenance = None
            if mode == "integrated":
                runtime_status, runtime_provenance = _runtime_snapshot(
                    runtime_manager, adapter, settings, artifact.get("sha256"))
            model_attempted_turns = sum(bool(item["result"].get("model_attempted"))
                                        for item in observed)
            model_generated_turns = sum(bool(item["result"].get(
                "model_generated", item["result"].get("model_used"))) for item in observed)
            model_accepted_turns = sum(bool(item["result"].get(
                "model_accepted", item["result"].get("model_used"))) for item in observed)
            model_fallback_turns = sum(bool(item["result"].get(
                "model_fallback", item["result"].get("model_attempted")
                and not item["result"].get("model_used"))) for item in observed)
            participation = model_generated_turns
            adapter_metadata = ({"mode": "deterministic_controller"}
                                if mode == "controller_only" else {
                                    "artifact_sha256": artifact.get("sha256"),
                                    "runtime_provenance_sha256": runtime_provenance["sha256"],
                                })
            adapter_metadata.update({"cohort_id": cohort_id, "corpus_sha256": corpus_hash,
                                     "outcome_schema_sha256": outcome_schema_sha256})
            turn_rows, turn_manifests = _turn_evidence(
                observed, cohort, run_id=f"{cohort_id}-{adapter}", adapter=adapter,
                controller_version=controller_version, generation_settings=settings,
                adapter_metadata=adapter_metadata, scenario_rows=rows_by_id)
            metrics = aggregate_evaluation_metrics(report["scenarios"], turn_rows)
            valid_turns = sum(row["validation_result"] == "valid" for row in turn_rows)
            repeated_turns = sum(row["repetitive_output"] is True for row in turn_rows)
            metrics["validation"] = {
                "valid_turns": valid_turns, "observed_turns": len(turn_rows),
                "rate": valid_turns / len(turn_rows) if turn_rows else None,
            }
            metrics["repetition_evidence"] = {
                "detected_turns": repeated_turns, "observed_turns": len(turn_rows),
                "rate": repeated_turns / len(turn_rows) if turn_rows else None,
                "basis": "duplicate_response_or_validator_repetition_reason",
            }
            model_fallback_reasons = sorted({
                str(item["result"]["model_fallback_reason"])
                for item in observed if item["result"].get("model_fallback_reason")
            })
            metrics["model_inference"] = {
                "attempted_turns": model_attempted_turns,
                "generated_turns": model_generated_turns,
                "accepted_turns": model_accepted_turns,
                "fallback_turns": model_fallback_turns,
                "acceptance_rate": (model_accepted_turns / model_generated_turns
                                    if model_generated_turns else None),
                "fallback_reasons": model_fallback_reasons,
            }
            status, reason = contract["status"], contract["status_reason"]
            if mode == "integrated":
                if participation <= 0 or not runtime_provenance["complete"]:
                    status = "incomplete"
                    problems = [reason] if reason else []
                    if participation <= 0:
                        problems.append("model_participation.turns")
                    problems.extend(runtime_provenance["incomplete_fields"])
                    reason = "; ".join(problems)
            run_id = f"{cohort_id}-{adapter}"
            run_record = RunRecord(
                run_id=run_id, controller_version=controller_version,
                adapter={"name": adapter, "layer": mode,
                         "cohort_id": cohort_id, "corpus_sha256": corpus_hash,
                         "outcome_schema": {"name": OUTCOME_SCHEMA,
                                            "version": OUTCOME_SCHEMA_VERSION,
                                            "sha256": outcome_schema_sha256},
                         "runtime_status": runtime_status,
                         "runtime_provenance": runtime_provenance},
                hardware=({"device": runtime_provenance["device"],
                           "precision": runtime_provenance["precision"],
                           "peak_memory_bytes": runtime_provenance["peak_memory_bytes"],
                           "peak_process_rss_mb": report["summary"].get("peak_process_rss_mb")}
                          if runtime_provenance is not None else {}),
                generation_settings=settings,
            )
            run_manifest = {"record": json.loads(record_json(run_record)),
                            "sha256": record_sha256(run_record)}
            fingerprint_parts = {
                "controller_sha256": controller_version,
                "outcome_schema": {"name": OUTCOME_SCHEMA,
                                    "version": OUTCOME_SCHEMA_VERSION,
                                    "sha256": outcome_schema_sha256},
                "corpus_sha256": corpus_hash,
                "generation_settings_sha256": _canonical_hash(settings),
                "runtime_provenance_sha256": (runtime_provenance["sha256"]
                                               if runtime_provenance else None),
            }
            arms[adapter] = {
                "layer": contract["layer"], "status": status, "status_reason": reason,
                "fingerprint": contract["fingerprint"]["sha256"],
                "fingerprints": fingerprint_parts,
                "cohort_id": cohort_id, "scenario_ids": list(ids),
                "runtime_status": runtime_status,
                "runtime_provenance": runtime_provenance,
                "model_participation": {"turns": participation,
                                        "attempted_turns": model_attempted_turns,
                                        "generated_turns": model_generated_turns,
                                        "accepted_turns": model_accepted_turns,
                                        "fallback_turns": model_fallback_turns,
                                        "required": mode == "integrated"},
                "run_manifest": run_manifest, "turn_manifests": turn_manifests,
                "summary": report["summary"], "metrics": metrics,
                "scenarios": [{"scenario_id": row["scenario_id"],
                               "normalized_outcome": _normalized(row),
                               "metrics": row} for row in report["scenarios"]],
            }
        finally:
            if mode == "integrated":
                runtime_manager.release()
    if {tuple(arm["scenario_ids"]) for arm in arms.values()} != {tuple(ids)}:
        raise ValueError("matrix arms do not share an identical scenario cohort")
    reference_adapter = ADAPTERS[0] if ADAPTERS[0] in arms else selected_adapters[0]
    reference = {row["scenario_id"]: row["normalized_outcome"]
                 for row in arms[reference_adapter]["scenarios"]}
    comparisons = {}
    for adapter, arm in arms.items():
        outcomes = {row["scenario_id"]: row["normalized_outcome"] for row in arm["scenarios"]}
        if set(outcomes) != set(reference):
            raise ValueError(f"adapter {adapter} has non-identical scenario IDs")
        comparisons[adapter] = {
            "matches_reference": outcomes == reference,
            "scenario_deltas": {str(sid): {
                key: {"reference": reference[sid][key], "adapter": outcomes[sid][key]}
                for key in reference[sid] if outcomes[sid][key] != reference[sid][key]
            } for sid in ids if outcomes[sid] != reference[sid]},
        }
    return {
        "schema": "rlaif.local_app.adapter_matrix", "report_version": MATRIX_REPORT_VERSION,
        "mode": mode, "layer": mode, "cohort_id": cohort_id,
        "cohort": {"scenario_ids": ids, "scenario_count": len(ids),
                   "scenario_sha256": _canonical_hash(cohort_payload),
                   "corpus_sha256": corpus_hash, "settings": settings},
        "adapters": arms, "scenario_comparison": comparisons,
        "baseline_freeze_allowed": mode == "controller_only" and all(
            arm["status"] == "complete" for arm in arms.values()),
    }


def write_json_atomic(value: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def freeze_baseline(report: Mapping[str, Any], output_path: str | Path) -> Path:
    requested_mode = report.get("mode")
    if requested_mode != "controller_only" or report.get("layer") != requested_mode:
        raise ValueError("only controller_only reports can be frozen as baselines")
    adapters = report.get("adapters")
    if not isinstance(adapters, Mapping) or set(adapters) != set(ADAPTERS):
        raise ValueError("a baseline requires complete results for all four adapters")
    for adapter in ADAPTERS:
        arm = adapters[adapter]
        if not isinstance(arm, Mapping) or arm.get("layer") != requested_mode:
            raise ValueError(f"adapter {adapter} layer does not match requested mode")
        fingerprints = arm.get("fingerprints")
        participation = arm.get("model_participation")
        run_record = (arm.get("run_manifest") or {}).get("record", {})
        run_adapter = run_record.get("adapter", {}) if isinstance(run_record, Mapping) else {}
        if (arm.get("runtime_status") is not None
                or arm.get("runtime_provenance") is not None
                or not isinstance(fingerprints, Mapping)
                or fingerprints.get("runtime_provenance_sha256") is not None
                or not isinstance(participation, Mapping)
                or participation.get("turns") != 0
                or participation.get("required") is not False
                or not isinstance(run_adapter, Mapping)
                or run_adapter.get("layer") != requested_mode
                or run_adapter.get("runtime_status") is not None
                or run_adapter.get("runtime_provenance") is not None):
            raise ValueError(f"adapter {adapter} violates controller_only provenance contract")
        if arm.get("status") != "complete":
            raise ValueError("a baseline requires complete results for all four adapters")
    baseline = {"schema": BASELINE_SCHEMA, "baseline_version": BASELINE_VERSION,
                "matrix_report_version": report["report_version"],
                "mode": "controller_only", "cohort_id": report["cohort_id"],
                "cohort": report["cohort"], "adapters": report["adapters"],
                "scenario_comparison": report["scenario_comparison"]}
    return write_json_atomic(baseline, output_path)


def compare_baseline(report: Mapping[str, Any], baseline_path: str | Path) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    if baseline.get("schema") != BASELINE_SCHEMA or baseline.get("baseline_version") != BASELINE_VERSION:
        raise ValueError("unsupported baseline schema or version")
    if report.get("layer") != baseline.get("mode"):
        raise ValueError("baseline and report evaluation layers differ")
    if report.get("cohort") != baseline.get("cohort"):
        raise ValueError("baseline and report cohorts/settings are not identical")
    result = {}
    for adapter in ADAPTERS:
        old_rows = baseline["adapters"][adapter]["scenarios"]
        new_rows = report["adapters"][adapter]["scenarios"]
        old = {row["scenario_id"]: row["normalized_outcome"] for row in old_rows}
        new = {row["scenario_id"]: row["normalized_outcome"] for row in new_rows}
        if old.keys() != new.keys():
            raise ValueError(f"adapter {adapter} baseline scenario cohort differs")
        result[adapter] = {"matches": old == new,
                           "changed_scenario_ids": sorted(sid for sid in old if old[sid] != new[sid])}
    return result


def _release_identity(*, adapters: Mapping[str, Mapping[str, Any]], corpus_sha256: str,
                      controller_sha256: str, base_model: str, base_model_revision: str,
                      generation_settings: Mapping[str, Any]) -> dict[str, Any]:
    adapter_hashes = {name: (arm.get("runtime_provenance") or {}).get("adapter_sha256")
                      for name, arm in adapters.items()}
    return {"adapter_sha256": adapter_hashes, "base_model": base_model,
            "base_model_revision": base_model_revision, "corpus_sha256": corpus_sha256,
            "controller_sha256": controller_sha256,
            "generation_settings": dict(generation_settings)}


def run_release(*, runtime_manager: Any, corpus_path: str | Path = DEFAULT_CORPUS,
                output_dir: str | Path = RELEASE_ROOT, start: int = 0, count: int = 50,
                temperature: float = 0.0, max_new_tokens: int = 160,
                arm_runner: Any = run_matrix) -> dict[str, Any]:
    """Run and checkpoint an integrated deterministic release one adapter at a time."""
    if (start, count, temperature, max_new_tokens) != (0, 50, 0.0, 160):
        raise ValueError("deterministic release requires temperature=0, max_new_tokens=160, start=0, count=50")
    if runtime_manager is None:
        raise ValueError("integrated release requires a local runtime manager")
    out = Path(output_dir)
    checkpoints = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    expected_scenario_ids = [scenario.scenario_id for scenario in
                             load_scenarios(corpus_path, count=count, start=start)]
    corpus_hash = evaluation._sha256_path(Path(corpus_path))
    controller_hash = _controller_version()
    base_model = runtime_manager.config.base_model
    base_revision = runtime_manager.config.revision
    settings = {"mode": "model", "base_model": base_model,
                "revision": base_revision, "temperature": 0.0,
                "max_new_tokens": 160, "start": 0, "count": 50}
    arms: dict[str, dict[str, Any]] = {}
    for adapter in ADAPTERS:
        checkpoint = checkpoints / f"{adapter}.json"
        if checkpoint.exists():
            arm = json.loads(checkpoint.read_text(encoding="utf-8"))
            supplied_checkpoint_hash = arm.pop("checkpoint_sha256", None)
            if supplied_checkpoint_hash != _canonical_hash(arm):
                raise ValueError(f"release checkpoint hash mismatch for {adapter}")
            if arm.get("adapter") != adapter:
                raise ValueError(f"checkpoint adapter mismatch for {adapter}")
            expected_adapter_hash = _adapter_artifact(runtime_manager, adapter).get("sha256")
            cached_arm = arm.get("arm")
            provenance = cached_arm.get("runtime_provenance") or {} if isinstance(cached_arm, Mapping) else {}
            participation = cached_arm.get("model_participation") if isinstance(cached_arm, Mapping) else None
            if (arm.get("corpus_sha256") != corpus_hash
                    or arm.get("controller_sha256") != controller_hash
                    or arm.get("base_model") != base_model
                    or arm.get("base_model_revision") != base_revision
                    or arm.get("generation_settings") != settings
                    or provenance.get("adapter_sha256") != expected_adapter_hash
                    or provenance.get("base_model") != base_model
                    or provenance.get("base_model_revision") != base_revision
                    or provenance.get("complete") is not True
                    or not isinstance(cached_arm, Mapping)
                    or cached_arm.get("layer") != "integrated"
                    or cached_arm.get("status") != "complete"
                    or cached_arm.get("scenario_ids") != expected_scenario_ids
                    or (cached_arm.get("metrics") or {}).get("scenario_count") != count
                    or not isinstance(participation, Mapping)
                    or participation.get("generated_turns", participation.get("turns", 0)) <= 0):
                raise ValueError(f"release checkpoint identity mismatch for {adapter}")
            arms[adapter] = arm["arm"]
            continue
        report = arm_runner(mode="integrated", corpus_path=corpus_path, count=count,
                            start=start, runtime_manager=runtime_manager,
                            temperature=temperature, max_new_tokens=max_new_tokens,
                            adapters=(adapter,))
        arm = report["adapters"][adapter]
        provenance = arm.get("runtime_provenance") or {}
        if arm.get("layer") != "integrated" or arm.get("status") != "complete":
            raise ValueError(f"adapter {adapter} did not pass integrated release gates")
        participation = arm.get("model_participation") or {}
        if (not provenance.get("complete")
                or participation.get("generated_turns", participation.get("turns", 0)) <= 0
                or provenance.get("base_model") != base_model
                or provenance.get("base_model_revision") != base_revision
                or arm.get("scenario_ids") != expected_scenario_ids
                or (arm.get("metrics") or {}).get("scenario_count") != count):
            raise ValueError(f"adapter {adapter} has incomplete release provenance")
        identity = {"adapter": adapter, "arm": arm, "corpus_sha256": corpus_hash,
                    "controller_sha256": controller_hash, "base_model": base_model,
                    "base_model_revision": base_revision, "generation_settings": settings}
        identity["checkpoint_sha256"] = _canonical_hash(identity)
        write_json_atomic(identity, checkpoint)
        arms[adapter] = arm
    if set(arms) != set(ADAPTERS):
        raise ValueError("release requires complete results for all four adapters")
    cohort_ids = {arm.get("cohort_id") for arm in arms.values()}
    scenario_sets = {tuple(arm.get("scenario_ids") or ()) for arm in arms.values()}
    if (len(cohort_ids) != 1 or None in cohort_ids or len(scenario_sets) != 1
            or list(next(iter(scenario_sets))) != expected_scenario_ids):
        raise ValueError("release arms do not share one non-empty cohort")
    scenario_ids = list(next(iter(scenario_sets)))
    release = {"schema": RELEASE_SCHEMA, "release_version": RELEASE_VERSION,
               "mode": "integrated", "layer": "integrated",
               "cohort_id": next(iter(cohort_ids)),
               "cohort": {"scenario_ids": scenario_ids,
                           "scenario_count": len(scenario_ids), "settings": settings},
               "generation_settings": settings,
               "identity": _release_identity(adapters=arms, corpus_sha256=corpus_hash,
                   controller_sha256=controller_hash, base_model=base_model,
                   base_model_revision=base_revision, generation_settings=settings),
               "adapters": arms}
    release["sha256"] = _canonical_hash(release)
    write_json_atomic(release, out / "release-v1.json")
    write_json_atomic({"schema": RELEASE_SCHEMA, "release_version": RELEASE_VERSION,
                       "sha256": release["sha256"], "identity": release["identity"]},
                      out / "manifest-v1.json")
    return release


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("controller_only", "integrated"), default="controller_only")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--release", action="store_true",
                        help="run resumable deterministic integrated release")
    parser.add_argument("--release-dir", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--freeze-baseline", type=Path)
    args = parser.parse_args(argv)
    manager = None
    if args.mode == "integrated":
        from .runtime import RuntimeConfig, RuntimeManager
        config = RuntimeConfig.local_defaults()
        from dataclasses import replace
        config = replace(config, temperature=args.temperature,
                         max_new_tokens=args.max_new_tokens)
        manager = RuntimeManager(config)
    try:
        if args.release:
            if args.mode != "integrated":
                parser.error("--release requires --mode integrated")
            report = run_release(runtime_manager=manager, corpus_path=args.corpus,
                                 output_dir=args.release_dir, start=args.start,
                                 count=args.count, temperature=args.temperature,
                                 max_new_tokens=args.max_new_tokens)
            print(json.dumps({"status": "written", "mode": "integrated",
                              "output": str(args.release_dir / "release-v1.json")}, sort_keys=True))
            return 0
        report = run_matrix(mode=args.mode, corpus_path=args.corpus, count=args.count,
                            start=args.start, runtime_manager=manager,
                            temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        if args.freeze_baseline:
            if args.mode != "controller_only":
                parser.error("integrated results cannot be frozen until runtime metadata/contracts are integrated")
            freeze_baseline(report, args.freeze_baseline)
        if args.baseline:
            report["baseline_comparison"] = compare_baseline(report, args.baseline)
        write_json_atomic(report, args.output)
    finally:
        if manager:
            manager.release()
    print(json.dumps({"status": "written", "mode": args.mode,
                      "cohort_id": report["cohort_id"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

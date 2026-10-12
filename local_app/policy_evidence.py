"""Import published policy result summaries without running models or cloud jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROWS = {
    "SFT": "sft",
    "DPO": "dpo",
    "PPO checkpoint 19": "ppo_checkpoint_19",
    "Adaptive PPO update 50": "ppo_adaptive_50",
}
EVIDENCE_SOURCES = (
    ("rebuild/cloud/FINAL_RESULTS.md", "reported_results"),
    ("rebuild/eval_NOTES.md", "evaluation_protocol"),
    ("rebuild/cloud/eval_READINESS.md", "historical_cohort_manifest_reference"),
    ("rebuild/cloud/STATUS.md", "run_status_and_output_references"),
    ("rebuild/rlaif/evaluation/eval_qwen.py", "evaluation_implementation"),
    ("rebuild/cloud/eval_sft_full_a100.json", "sft_job_configuration"),
    ("rebuild/cloud/eval_dpo_full_a100.json", "dpo_job_configuration"),
    ("rebuild/cloud/eval_ppo_full_a100.json", "ppo_checkpoint_19_job_configuration"),
    ("rebuild/cloud/eval_ppo_adaptive_full_a100.json", "ppo_adaptive_50_job_configuration"),
)

_ROW = re.compile(
    r"^\|\s*(SFT|DPO|PPO checkpoint 19|Adaptive PPO update 50)\s*\|\s*"
    r"(\d+)/(\d+)\s*\(([\d.]+)%\)\s*\|\s*"
    r"([\d.]+)%–([\d.]+)%\s*\|\s*([\d.]+)\s*\|\s*"
    r"(—|[\d.]+%)\s*\|\s*([\d.]+)%\s*\|\s*$"
)
_MANIFEST_HASH = re.compile(r"manifest SHA256:\s*\n\s*`([0-9a-f]{64})`", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_results(markdown: str) -> dict[str, dict[str, Any]]:
    """Parse only the fixed numeric summary table; never copy arbitrary prose."""
    in_results = False
    rows: dict[str, dict[str, Any]] = {}
    for line in markdown.splitlines():
        if line.strip() == "## Held-out evaluation":
            in_results = True
            continue
        if in_results and line.startswith("## "):
            break
        if not in_results:
            continue
        match = _ROW.fullmatch(line)
        if not match:
            continue
        label, successes, observed, rate, low, high, judge, booking, unknown = match.groups()
        adapter = ADAPTER_ROWS[label]
        if adapter in rows:
            raise ValueError(f"duplicate policy result row for {adapter}")
        successes, observed = int(successes), int(observed)
        if observed != 50 or successes > observed:
            raise ValueError(f"unexpected held-out count for {adapter}")
        rows[adapter] = {
            "task_completion": {
                "successes": successes,
                "observed": observed,
                "rate": round(float(rate) / 100, 6),
                "wilson_95": [round(float(low) / 100, 6), round(float(high) / 100, 6)],
                "basis": "reported_summary",
            },
            "judge_score": {"value": float(judge), "basis": "reported_mean",
                            "scored_scenarios": None},
            "booking_success": {"rate": round(float(booking[:-1]) / 100, 6) if booking != "—" else None,
                                "denominator": None, "basis": "reported_summary"},
            "unknown_flight_id_hallucination_rate": {
                "rate": round(float(unknown) / 100, 6),
                "observed": observed,
                "basis": "reported_summary",
            },
            "evidence_level": "reported_summary",
            "scenario_records_available": False,
            "evaluated_adapter_sha256": None,
            "adapter_identity_verified": False,
        }
    if set(rows) != set(ADAPTER_ROWS.values()):
        missing = sorted(set(ADAPTER_ROWS.values()) - set(rows))
        raise ValueError("held-out results table is missing adapter rows: " + ", ".join(missing))
    return rows


def build_policy_evidence(*, root: str | Path = ROOT) -> dict[str, Any]:
    """Create a hash-pinned, summary-only policy report from local evidence."""
    project = Path(root).resolve()
    result_path = project / "rebuild/cloud/FINAL_RESULTS.md"
    if not result_path.is_file():
        raise FileNotFoundError(f"required local results summary is missing: {result_path.name}")
    rows = _parse_results(result_path.read_text(encoding="utf-8"))

    evidence = []
    for relative, role in EVIDENCE_SOURCES:
        source = project / relative
        if not source.is_file():
            raise FileNotFoundError(f"required local evidence source is missing: {relative}")
        evidence.append({"artifact": relative, "sha256": _sha256(source), "role": role})

    result_digest = next(item["sha256"] for item in evidence
                         if item["artifact"] == "rebuild/cloud/FINAL_RESULTS.md")
    protocol_digest = next(item["sha256"] for item in evidence
                           if item["artifact"] == "rebuild/eval_NOTES.md")
    readiness_path = project / "rebuild/cloud/eval_READINESS.md"
    manifest_match = _MANIFEST_HASH.search(readiness_path.read_text(encoding="utf-8"))
    if not manifest_match:
        raise ValueError("historical readiness source is missing its reported manifest SHA-256")
    return {
        "report_version": 1,
        "evaluation_layer": {
            "layer": "policy_only",
            "status": "incomplete",
            "evidence_level": "reported_summary",
            "cohort_id": "held-out-50-seed20260922",
            "protocol_id": "held-out-policy-protocol-v3",
            "controller_usage": {"executed": False, "rescue": False},
            "model_usage": {"runtime_invoked": False, "results_source": "published_summary"},
            "status_reason": (
                "Aggregate values were imported from the local final-results summary. "
                "Per-scenario outputs, resolved evaluation revisions, and verified links "
                "between local adapter bytes and evaluated adapters are not present locally."
            ),
        },
        "cohort": {
            "scenario_count": 50,
            "seed": 20260922,
            "manifest_sha256": manifest_match.group(1),
            "manifest_hash_status": "reported_in_protocol_readiness; manifest file not present locally",
        },
        "evidence_artifacts": evidence,
        "results_imported": True,
        "summary": {
            "evidence_level": "reported_summary",
            "source_artifact": "rebuild/cloud/FINAL_RESULTS.md",
            "source_sha256": result_digest,
            "scenario_count": 50,
            "adapters": rows,
        },
        "scenarios": [],
        "scenario_records_imported": False,
        "provenance": {
            "protocol_source": "rebuild/eval_NOTES.md",
            "protocol_source_sha256": protocol_digest,
            "gcp_accessed": False,
            "network_accessed": False,
        },
        "limitations": [
            "Only published aggregate values are imported; no per-scenario records are claimed.",
            "Local adapter files are not asserted to be byte-identical to the evaluated artifacts.",
            "Per-arm resolved model revisions and raw evaluation manifests are unavailable locally.",
            "Judge-score sample counts and booking-success denominators are absent from the summary table.",
            "Protocol notes contain stale pre-evaluation status text and are used only as hash-pinned protocol context.",
        ],
    }


def write_policy_evidence(*, root: str | Path = ROOT,
                          output: str | Path | None = None) -> dict[str, Any]:
    """Write the generated report atomically after all sources validate."""
    report = build_policy_evidence(root=root)
    destination = (Path(output) if output is not None else
                   Path(root) / "artifacts" / "evaluation" / "policy-only-v1.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="local project root containing the recorded evidence")
    parser.add_argument("--output", type=Path,
                        help="output JSON (defaults to artifacts/evaluation/policy-only-v1.json under --root)")
    args = parser.parse_args(argv)
    destination = args.output or args.root / "artifacts" / "evaluation" / "policy-only-v1.json"
    report = write_policy_evidence(root=args.root, output=destination)
    print(json.dumps({"status": report["evaluation_layer"]["status"],
                      "evidence_level": report["evaluation_layer"]["evidence_level"],
                      "output": destination.name,
                      "adapters": sorted(report["summary"]["adapters"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

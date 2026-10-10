"""Build public SVG charts from verified, sanitized RLAIF metric extracts.

Install ReportLab, then run this file from any directory. To refresh the public
metric extract from private downloaded GCS logs, pass --extract-source-dir with
the directory layout described in figures/README.md. Raw logs stay outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib.colors import HexColor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "metrics.json"
INK = HexColor("#192438")
MUTED = HexColor("#5b6b82")
GRID = HexColor("#dbe2eb")
BLUE = HexColor("#2364aa")
TEAL = HexColor("#008b83")
ORANGE = HexColor("#d47b18")
PURPLE = HexColor("#7858aa")
RED = HexColor("#bb4751")
PAPER = HexColor("#ffffff")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_metrics(source: Path) -> None:
    """Extract numeric fields only; keep raw prompts and cloud IDs out of Git."""
    sft_path = source / "sft" / "trainer_state.json"
    dpo_path = source / "dpo" / "trainer_state.json"
    original_path = source / "ppo19" / "training_log.jsonl"
    adaptive_path = source / "ppo50" / "training_log.jsonl"
    preference_path = source / "prefs" / "pair_stats.json"
    eval_paths = {name: source / "eval" / f"{name}.json" for name in
                  ("base", "sft", "dpo", "ppo19", "ppo50")}

    def trainer_rows(path: Path, include_accuracy: bool = False) -> list[dict]:
        history = json.loads(path.read_text())["log_history"]
        rows = []
        for item in history:
            if "loss" not in item:
                continue
            row = {"step": int(item["step"]), "loss": float(item["loss"])}
            if include_accuracy:
                row["training_pair_accuracy"] = float(item["rewards/accuracies"])
            rows.append(row)
        return rows

    def ppo_rows(path: Path) -> list[dict]:
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({
                "update": int(item["update"]),
                "reward_mean": float(item["reward_mean"]),
                "reference_kl_k3": float(item["reference_kl_k3"]),
                "accepted_ratio_max": max(float(e["ratio_max"]) for e in item["epochs"]),
                "dropped_rewards": int(item["dropped_rewards"]),
            })
        return rows

    evaluations = {}
    for name, path in eval_paths.items():
        item = json.loads(path.read_text())
        if item["completed_scenarios"] != 50 or item["provisional"]:
            raise ValueError(f"Incomplete evaluation: {name}")
        evaluations[name] = {
            "completed_scenarios": int(item["completed_scenarios"]),
            "task_success_count": int(item["task_success_count"]),
            "task_success_rate": float(item["task_success_rate"]),
            "task_success_wilson95": [float(v) for v in item["task_success_wilson95"]],
            "mean_judge_score": float(item["mean_judge_score"]),
        }

    preference = json.loads(preference_path.read_text())
    preference_fields = ("attempted", "ineligible", "parse_failed", "judge_failed",
                         "gap_filtered", "kept", "stability_tested",
                         "stability_top1_agreements")
    preference_summary = {key: int(preference[key]) for key in preference_fields}
    preference_summary["yield_rate"] = float(preference["yield_rate"])
    preference_summary["passed"] = bool(preference["passed"])
    if sum(preference_summary[key] for key in
           ("ineligible", "parse_failed", "judge_failed", "gap_filtered", "kept")) != preference_summary["attempted"]:
        raise ValueError("Preference outcome counts do not reconcile")
    if not preference_summary["passed"]:
        raise ValueError("Preference quality gate did not pass")

    result = {
        "note": "Numeric extracts from completed run logs; raw evidence remains in private GCS.",
        "source_sha256": {
            "sft_trainer_state": _hash(sft_path),
            "dpo_trainer_state": _hash(dpo_path),
            "ppo_original_training_log": _hash(original_path),
            "ppo_adaptive_training_log": _hash(adaptive_path),
            "preference_pair_stats": _hash(preference_path),
            **{f"eval_{name}_summary": _hash(path) for name, path in eval_paths.items()},
        },
        "sft": trainer_rows(sft_path),
        "dpo": trainer_rows(dpo_path, include_accuracy=True),
        "ppo_original": ppo_rows(original_path),
        "ppo_adaptive": ppo_rows(adaptive_path),
        "preference": preference_summary,
        "evaluations": evaluations,
    }
    if len(result["sft"]) != 62 or len(result["dpo"]) != 72:
        raise ValueError("Unexpected trainer log length")
    if [r["update"] for r in result["ppo_original"]] != list(range(1, 20)):
        raise ValueError("Original PPO update sequence is incomplete")
    if [r["update"] for r in result["ppo_adaptive"]] != list(range(20, 51)):
        raise ValueError("Adaptive PPO update sequence is incomplete")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def label(d: Drawing, x: float, y: float, value: str, size: int = 13,
          color=INK, anchor: str = "start", bold: bool = False) -> None:
    d.add(String(x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size, fillColor=color, textAnchor=anchor))


def base(width: int, height: int, title: str, subtitle: str) -> Drawing:
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PAPER, strokeColor=None))
    label(d, 48, height - 49, title, 24, bold=True)
    label(d, 48, height - 75, subtitle, 12, color=MUTED)
    return d


def axes(d: Drawing, box: tuple[float, float, float, float],
         xlim: tuple[float, float], ylim: tuple[float, float],
         xticks: list[float], yticks: list[float], xname: str,
         yfmt=lambda v: f"{v:g}"):
    x, y, w, h = box
    sx = lambda v: x + (v - xlim[0]) / (xlim[1] - xlim[0]) * w
    sy = lambda v: y + (v - ylim[0]) / (ylim[1] - ylim[0]) * h
    for v in yticks:
        yy = sy(v)
        d.add(Line(x, yy, x + w, yy, strokeColor=GRID, strokeWidth=0.8))
        label(d, x - 10, yy - 4, yfmt(v), 10, MUTED, "end")
    d.add(Line(x, y, x + w, y, strokeColor=INK, strokeWidth=1))
    d.add(Line(x, y, x, y + h, strokeColor=INK, strokeWidth=1))
    for v in xticks:
        xx = sx(v)
        d.add(Line(xx, y, xx, y - 5, strokeColor=INK, strokeWidth=0.8))
        label(d, xx, y - 21, f"{v:g}", 10, MUTED, "middle")
    label(d, x + w / 2, y - 48, xname, 12, MUTED, "middle")
    return sx, sy


def line_series(d: Drawing, rows: list[dict], xkey: str, ykey: str,
                sx, sy, color, dots: bool = False) -> None:
    points = [(sx(float(r[xkey])), sy(float(r[ykey]))) for r in rows]
    d.add(PolyLine([v for point in points for v in point], strokeColor=color,
                   strokeWidth=2.7, fillColor=None))
    if dots:
        for x, y in points:
            d.add(Circle(x, y, 2.6, fillColor=color, strokeColor=None))


def training_losses(data: dict) -> None:
    d = base(1120, 610, "Training objectives", "Measured logs from rebuilt SFT and DPO runs")
    label(d, 95, 490, "SFT token loss", 17, bold=True)
    label(d, 640, 490, "DPO preference loss", 17, bold=True)
    sx, sy = axes(d, (95, 132, 425, 320), (0, 625), (0, 4.2),
                  [0, 125, 250, 375, 500, 625], [0, 1, 2, 3, 4], "Optimizer step")
    line_series(d, data["sft"], "step", "loss", sx, sy, BLUE)
    sx, sy = axes(d, (640, 132, 425, 320), (0, 72), (0.5, 0.72),
                  [0, 12, 24, 36, 48, 60, 72], [0.5, 0.55, 0.6, 0.65, 0.7],
                  "Optimizer step", lambda v: f"{v:.2f}")
    line_series(d, data["dpo"], "step", "loss", sx, sy, TEAL)
    label(d, 95, 54, "Run-average SFT loss: 0.865 (625 steps)", 12, MUTED)
    label(d, 640, 54, "Run-average DPO loss: 0.627 (72 steps)", 12, MUTED)
    label(d, 95, 30, "Objectives and scales differ; do not compare their loss values directly.", 11, MUTED)
    renderSVG.drawToFile(d, str(ROOT / "training_losses.svg"))


def ppo_diagnostics(data: dict) -> None:
    d = base(1280, 640, "PPO training diagnostics", "Accepted updates 1–19: rebuilt PPO v3  |  20–50: adaptive continuation")
    original, adaptive = data["ppo_original"], data["ppo_adaptive"]
    panels = [
        (70, "Mean grounded reward", "reward_mean", (-1.1, 0.25), [-1, -0.75, -0.5, -0.25, 0, 0.25], lambda v: f"{v:.2f}"),
        (470, "Frozen-reference KL (k3)", "reference_kl_k3", (0, 0.06), [0, 0.015, 0.03, 0.045, 0.06], lambda v: f"{v:.3f}"),
        (870, "Max accepted ratio", "accepted_ratio_max", (0, 4.2), [0, 1, 2, 3, 4], lambda v: f"{v:g}"),
    ]
    for x, title, field, ylim, ticks, fmt in panels:
        label(d, x + 38, 515, title, 15, bold=True)
        sx, sy = axes(d, (x + 38, 145, 325, 330), (1, 50), ylim,
                      [1, 10, 19, 30, 40, 50], ticks, "Accepted update", fmt)
        boundary = sx(19.5)
        d.add(Line(boundary, 145, boundary, 475, strokeColor=ORANGE,
                   strokeWidth=1.5, strokeDashArray=[5, 4]))
        line_series(d, original, "update", field, sx, sy, PURPLE, True)
        line_series(d, adaptive, "update", field, sx, sy, TEAL, True)
        if field == "accepted_ratio_max":
            yy = sy(4)
            d.add(Line(x + 38, yy, x + 363, yy, strokeColor=RED,
                       strokeWidth=1.4, strokeDashArray=[4, 4]))
            label(d, x + 355, yy + 8, "hard gate 4.0", 10, RED, "end")
    d.add(Line(70, 85, 103, 85, strokeColor=PURPLE, strokeWidth=3))
    label(d, 112, 80, "Rebuilt PPO v3", 11, MUTED)
    d.add(Line(245, 85, 278, 85, strokeColor=TEAL, strokeWidth=3))
    label(d, 287, 80, "Adaptive PPO", 11, MUTED)
    label(d, 70, 47, "Only accepted updates are plotted. PPO v3 update 20 was rejected; adaptive update 20 passed after rollback and lower-rate retry.", 11, MUTED)
    label(d, 70, 28, "Per-update reward is noisy and is not held-out task success.", 11, MUTED)
    renderSVG.drawToFile(d, str(ROOT / "ppo_diagnostics.svg"))


def evaluation_success(data: dict) -> None:
    d = base(1100, 590, "Held-out task success", "Same 50 evaluation scenarios per arm  |  bars show observed rate  |  whiskers show Wilson 95% intervals")
    names = ["Base", "SFT", "DPO", "PPO 19", "PPO 50"]
    keys = ["base", "sft", "dpo", "ppo19", "ppo50"]
    colors = [MUTED, BLUE, TEAL, PURPLE, ORANGE]
    x, y, w, h = 105, 155, 915, 320
    sy = lambda v: y + v / 0.65 * h
    for tick in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        yy = sy(tick)
        d.add(Line(x, yy, x + w, yy, strokeColor=GRID, strokeWidth=0.8))
        label(d, x - 12, yy - 4, f"{tick:.0%}", 11, MUTED, "end")
    d.add(Line(x, y, x + w, y, strokeColor=INK, strokeWidth=1))
    for i, (name, key, color) in enumerate(zip(names, keys, colors)):
        item = data["evaluations"][key]
        cx = x + 95 + i * 182
        value = item["task_success_rate"]
        lo, hi = item["task_success_wilson95"]
        d.add(Rect(cx - 51, y, 102, sy(value) - y,
                   fillColor=color, strokeColor=None))
        d.add(Line(cx, sy(lo), cx, sy(hi), strokeColor=INK, strokeWidth=2))
        d.add(Line(cx - 11, sy(lo), cx + 11, sy(lo), strokeColor=INK, strokeWidth=2))
        d.add(Line(cx - 11, sy(hi), cx + 11, sy(hi), strokeColor=INK, strokeWidth=2))
        label(d, cx, sy(hi) + 13, f"{value:.0%}", 14, INK, "middle", True)
        label(d, cx, 127, name, 13, INK, "middle", True)
        label(d, cx, 106, f"{item['task_success_count']}/50", 11, MUTED, "middle")
    label(d, 105, 55, "PPO 19 = gate-early-stopped checkpoint; PPO 50 = adaptive continuation. Intervals overlap; differences are directional.", 11, MUTED)
    label(d, 105, 34, "Source: final non-provisional evaluation summaries; all arms use protocol v3.", 11, MUTED)
    renderSVG.drawToFile(d, str(ROOT / "evaluation_task_success.svg"))


def preference_generation(data: dict) -> None:
    d = base(1100, 490, "Preference generation quality", "Full validated run: 700 decision scenarios judged with the grounded rubric")
    pref = data["preference"]
    categories = [
        ("Retained pairs", "kept", TEAL),
        ("Score gap below threshold", "gap_filtered", ORANGE),
        ("Ineligible", "ineligible", MUTED),
        ("Parse failure", "parse_failed", PURPLE),
        ("Judge failure", "judge_failed", RED),
    ]
    x, y, width, height = 80, 285, 940, 76
    at = x
    for title, key, color in categories:
        segment = width * pref[key] / pref["attempted"]
        d.add(Rect(at, y, segment, height, fillColor=color, strokeColor=PAPER, strokeWidth=1))
        if segment > 70:
            label(d, at + segment / 2, y + 30, str(pref[key]), 20, PAPER, "middle", True)
        at += segment
    for idx, (title, key, color) in enumerate(categories):
        col = idx % 3
        row = idx // 3
        xx, yy = 80 + col * 345, 218 - row * 58
        d.add(Rect(xx, yy - 3, 17, 17, fillColor=color, strokeColor=None))
        label(d, xx + 27, yy, f"{title}: {pref[key]}", 12, INK)
    label(d, 80, 92, f"Retained: {pref['kept']}/{pref['attempted']} ({pref['yield_rate']:.1%})", 17, TEAL, bold=True)
    label(d, 80, 65, f"Quality gates passed; corrected top-choice stability: {pref['stability_top1_agreements']}/{pref['stability_tested']}.", 12, MUTED)
    label(d, 80, 40, "Outcome counts partition 700 scenarios. Parse-failure rate reported elsewhere uses judged responses as denominator.", 11, MUTED)
    renderSVG.drawToFile(d, str(ROOT / "preference_generation.svg"))


def dpo_accuracy(data: dict) -> None:
    d = base(940, 520, "DPO training pair accuracy", "Logged fraction where the chosen response scored above the rejected response in each training batch")
    sx, sy = axes(d, (100, 132, 775, 300), (0, 72), (0, 1),
                  [0, 12, 24, 36, 48, 60, 72], [0, 0.25, 0.5, 0.75, 1],
                  "Optimizer step", lambda v: f"{v:.0%}")
    line_series(d, data["dpo"], "step", "training_pair_accuracy", sx, sy, TEAL, True)
    label(d, 100, 53, "This is an in-training preference metric, not held-out task success or a model quality estimate.", 11, MUTED)
    renderSVG.drawToFile(d, str(ROOT / "dpo_pair_accuracy.svg"))


def evaluation_judge_score(data: dict) -> None:
    d = base(1050, 500, "Held-out judge score", "Same 50-scenario protocol; mean grounded-rubric score among judged scenarios")
    names = ["Base", "SFT", "DPO", "PPO 19", "PPO 50"]
    keys = ["base", "sft", "dpo", "ppo19", "ppo50"]
    colors = [MUTED, BLUE, TEAL, PURPLE, ORANGE]
    x, y, w, h = 105, 130, 890, 280
    sy = lambda v: y + v / 10 * h
    for tick in (0, 2, 4, 6, 8, 10):
        yy = sy(tick)
        d.add(Line(x, yy, x + w, yy, strokeColor=GRID, strokeWidth=0.8))
        label(d, x - 12, yy - 4, str(tick), 11, MUTED, "end")
    d.add(Line(x, y, x + w, y, strokeColor=INK, strokeWidth=1))
    for i, (name, key, color) in enumerate(zip(names, keys, colors)):
        value = data["evaluations"][key]["mean_judge_score"]
        cx = x + 90 + i * 178
        d.add(Rect(cx - 48, y, 96, sy(value) - y, fillColor=color, strokeColor=None))
        label(d, cx, sy(value) + 12, f"{value:.2f}", 14, INK, "middle", True)
        label(d, cx, 104, name, 12, INK, "middle", True)
    label(d, 105, 46, "Score and task success rank policies differently; this chart shows no uncertainty interval for the mean score.", 11, MUTED)
    renderSVG.drawToFile(d, str(ROOT / "evaluation_judge_score.svg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-source-dir", type=Path,
                        help="Private downloaded run metrics; export numeric data first")
    args = parser.parse_args()
    if args.extract_source_dir:
        export_metrics(args.extract_source_dir)
    data = json.loads(DATA.read_text())
    training_losses(data)
    ppo_diagnostics(data)
    evaluation_success(data)
    preference_generation(data)
    dpo_accuracy(data)
    evaluation_judge_score(data)


if __name__ == "__main__":
    main()

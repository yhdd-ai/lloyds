"""M5: assemble publication-ready baseline-versus-LinUCB comparisons.

Reads the completed static baseline and three LinUCB budget runs. It writes a
model-comparison table, a scenario-coverage table, and two PNG figures. The
script performs no feature engineering or model replay.

Usage:
    python 06_results_analysis.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


BUDGETS = (0.001, 0.005, 0.01)
LINUCB_PREFIXES = {0.001: "linucb_0p1", 0.005: "linucb_0p5", 0.01: "linucb_1p0"}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return json.loads(path.read_text())


def static_baseline_rows(outputs: Path, scenarios: pd.DataFrame) -> tuple[list[dict], dict[float, dict[int, int]]]:
    """Reconstruct baseline top-k scenario hits from its per-email score file."""
    daily_path = outputs / "daily_baseline_evaluation.json"
    if daily_path.exists():
        daily_summary = load_json(daily_path)["budget_metrics"]
        rows, scenario_hits = [], {}
        for budget in BUDGETS:
            metric = daily_summary[f"{budget:.1%}"]
            scenario_hits[budget] = {int(key): int(value) for key, value in metric["scenario_true_positives"].items()}
            rows.append({
                "method": "Static anomaly baseline", "budget": budget,
                "n_reviewed": metric["n_reviewed"], "true_positives": metric["true_positives"],
                "precision": metric["precision_at_budget"], "recall": metric["recall_at_budget"],
                "false_positive_rate": metric["false_positive_rate_at_budget"],
                "cumulative_regret": np.nan,
            })
        return rows, scenario_hits
    summary = load_json(outputs / "baseline_evaluation.json")
    score_path = Path(summary["score_output"])
    if not score_path.exists():
        raise FileNotFoundError(f"Baseline score file not found: {score_path}")
    all_scores, positives = [], []
    usecols = ["id", "is_malicious", "risk_score"]
    for chunk in pd.read_csv(score_path, usecols=usecols, chunksize=200_000, dtype={"id": str}):
        all_scores.append(chunk["risk_score"].to_numpy(float))
        positive = chunk.loc[chunk["is_malicious"].eq(1)]
        if not positive.empty:
            positives.append(positive.merge(scenarios, on="id", how="left"))
    scores = np.concatenate(all_scores)
    positive_scores = pd.concat(positives, ignore_index=True)
    rows, scenario_hits = [], {}
    for budget in BUDGETS:
        metric = summary["budget_metrics"][f"{budget:.1%}"]
        k = metric["n_reviewed"]
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        hits = positive_scores.loc[positive_scores["risk_score"] >= threshold, "scenario"].value_counts().to_dict()
        scenario_hits[budget] = {int(key): int(value) for key, value in hits.items()}
        rows.append({
            "method": "Static anomaly baseline", "budget": budget,
            "n_reviewed": metric["n_reviewed"], "true_positives": metric["true_positives"],
            "precision": metric["precision_at_budget"], "recall": metric["recall_at_budget"],
            "false_positive_rate": metric["false_positive_rate_at_budget"],
            "cumulative_regret": np.nan,
        })
    return rows, scenario_hits


def linucb_rows(outputs: Path) -> tuple[list[dict], dict[float, dict[int, int]]]:
    rows, scenario_hits = [], {}
    for budget, prefix in LINUCB_PREFIXES.items():
        summary = load_json(outputs / f"{prefix}_summary.json")
        scenario_hits[budget] = {int(key): int(value) for key, value in summary.get("scenario_true_positives", {}).items()}
        rows.append({
            "method": "LinUCB", "budget": budget,
            "n_reviewed": summary["n_reviewed"], "true_positives": summary["true_positives"],
            "precision": summary["precision_at_budget"], "recall": summary["recall_at_budget"],
            "false_positive_rate": summary["false_positive_rate_at_budget"],
            "cumulative_regret": summary["cumulative_regret"],
        })
    return rows, scenario_hits


def hybrid_row(outputs: Path) -> tuple[list[dict], dict[float, dict[int, int]]]:
    """Load the optional 1% coverage-balanced hybrid result."""
    path = outputs / "hybrid_0p5_static_0p5_linucb_summary.json"
    if not path.exists():
        return [], {}
    summary = load_json(path)
    budget = float(summary["target_budget"])
    row = {
        "method": "Coverage-balanced hybrid", "budget": budget,
        "n_reviewed": summary["n_reviewed"], "true_positives": summary["true_positives"],
        "precision": summary["precision_at_budget"], "recall": summary["recall_at_budget"],
        "false_positive_rate": summary["false_positive_rate_at_budget"], "cumulative_regret": np.nan,
    }
    hits = {budget: {int(key): int(value) for key, value in summary["scenario_true_positives"].items()}}
    return [row], hits


def scenario_table(scenarios: pd.DataFrame, method_hits: dict[str, dict]) -> pd.DataFrame:
    totals = scenarios["scenario"].value_counts().sort_index().to_dict()
    rows = []
    for budget in BUDGETS:
        for method, by_budget in method_hits.items():
            if budget not in by_budget:
                continue
            hits = by_budget[budget]
            for scenario, total in totals.items():
                hit = hits.get(int(scenario), 0)
                rows.append({
                    "method": method, "budget": budget, "scenario": int(scenario),
                    "malicious_emails_in_scenario": int(total), "true_positives": hit,
                    "scenario_recall": hit / total,
                })
    return pd.DataFrame(rows)


def _svg(width: int, height: int, contents: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:18px;font-weight:700}.axis{font-size:12px}.legend{font-size:12px}</style>',
        *contents,
        '</svg>',
    ])


def make_figures(comparison: pd.DataFrame, scenario_coverage: pd.DataFrame, figure_dir: Path) -> None:
    """Write portable SVG figures without a plotting-library dependency."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Static anomaly baseline": "#64748b", "LinUCB": "#2563eb", "Coverage-balanced hybrid": "#059669"}
    # Figure 1: review-budget versus recall.
    width, height, left, right, top, bottom = 760, 440, 85, 40, 70, 70
    plot_w, plot_h, ymax = width - left - right, height - top - bottom, 30.0
    contents = ['<text x="85" y="32" class="title">Threat detection recall under fixed review budgets</text>']
    for tick in range(0, 31, 10):
        y = top + plot_h - tick / ymax * plot_h
        contents += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                     f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}%</text>']
    xs = {0.001: left, 0.005: left + plot_w * 0.44, 0.01: left + plot_w}
    contents += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#334155"/>',
                 f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#334155"/>',
                 f'<text x="18" y="{top+plot_h/2}" class="axis" transform="rotate(-90 18 {top+plot_h/2})">Recall (%)</text>']
    for budget, x in xs.items():
        contents.append(f'<text x="{x}" y="{height-42}" text-anchor="middle" class="axis">{budget:.1%}</text>')
    contents.append(f'<text x="{left+plot_w/2}" y="{height-16}" text-anchor="middle" class="axis">Analyst review budget</text>')
    for method, group in comparison.groupby("method", sort=False):
        group = group.set_index("budget")
        points = []
        for budget in group.index:
            x, y = xs[budget], top + plot_h - group.loc[budget, "recall"] * 100 / ymax * plot_h
            points.append(f"{x:.1f},{y:.1f}")
            contents.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[method]}"/>')
        contents.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
    for index, method in enumerate(colors):
        y = 52 + index * 20
        contents += [f'<line x1="{width-225}" y1="{y}" x2="{width-202}" y2="{y}" stroke="{colors[method]}" stroke-width="3"/>',
                     f'<text x="{width-195}" y="{y+4}" class="legend">{method}</text>']
    (figure_dir / "review_budget_vs_recall.svg").write_text(_svg(width, height, contents))

    # Figure 2: three panels, each comparing model hits by scenario.
    width, height, top, bottom, ymax = 1120, 440, 70, 75, 40
    contents = ['<text x="45" y="32" class="title">Scenario coverage by review budget</text>']
    scenarios = sorted(scenario_coverage["scenario"].unique())
    panel_w, panel_gap, plot_h = 320, 35, height - top - bottom
    for panel, budget in enumerate(BUDGETS):
        x0 = 55 + panel * (panel_w + panel_gap)
        contents.append(f'<text x="{x0+panel_w/2}" y="{top-18}" text-anchor="middle" class="legend">Review budget: {budget:.1%}</text>')
        for tick in range(0, ymax + 1, 10):
            y = top + plot_h - tick / ymax * plot_h
            contents.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            if panel == 0:
                contents.append(f'<text x="{x0-9}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}</text>')
        subset = scenario_coverage.loc[scenario_coverage["budget"].eq(budget)]
        for index, scenario in enumerate(scenarios):
            centre = x0 + 55 + index * 100
            for offset, method in [(-20, "Static anomaly baseline"), (4, "LinUCB")]:
                value = subset.loc[(subset["method"].eq(method)) & (subset["scenario"].eq(scenario)), "true_positives"].iloc[0]
                bar_h = value / ymax * plot_h
                contents.append(f'<rect x="{centre+offset}" y="{top+plot_h-bar_h:.1f}" width="18" height="{bar_h:.1f}" fill="{colors[method]}"/>')
            contents.append(f'<text x="{centre}" y="{height-45}" text-anchor="middle" class="axis">S{scenario}</text>')
        contents += [f'<line x1="{x0}" y1="{top+plot_h}" x2="{x0+panel_w}" y2="{top+plot_h}" stroke="#334155"/>',
                     f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top+plot_h}" stroke="#334155"/>']
    contents += ['<rect x="55" y="405" width="14" height="14" fill="#64748b"/><text x="75" y="417" class="legend">Static baseline</text>',
                 '<rect x="205" y="405" width="14" height="14" fill="#2563eb"/><text x="225" y="417" class="legend">LinUCB</text>']
    (figure_dir / "scenario_coverage_by_budget.svg").write_text(_svg(width, height, contents))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.OUTPUT_DIR / "figures")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    comparison_path = args.outputs_dir / "model_comparison.csv"
    coverage_path = args.outputs_dir / "scenario_coverage.csv"
    figures = [args.figure_dir / "review_budget_vs_recall.svg", args.figure_dir / "scenario_coverage_by_budget.svg"]
    existing = [str(path) for path in [comparison_path, coverage_path, *figures] if path.exists()]
    if existing and not args.overwrite:
        parser.error("Output already exists; use --overwrite: " + ", ".join(existing))
    scenarios = pd.read_csv(config.GROUND_TRUTH_LOOKUP, dtype={"id": str})[["id", "scenario"]]
    baseline_rows, baseline_hits = static_baseline_rows(args.outputs_dir, scenarios)
    lin_rows, lin_hits = linucb_rows(args.outputs_dir)
    hybrid_rows, hybrid_hits = hybrid_row(args.outputs_dir)
    comparison = pd.DataFrame([*baseline_rows, *lin_rows, *hybrid_rows]).sort_values(["budget", "method"])
    coverage = scenario_table(scenarios, {
        "Static anomaly baseline": baseline_hits,
        "LinUCB": lin_hits,
        "Coverage-balanced hybrid": hybrid_hits,
    })
    comparison.to_csv(comparison_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    make_figures(comparison, coverage, args.figure_dir)
    print("[info] wrote", comparison_path)
    print("[info] wrote", coverage_path)
    print("[info] wrote figures to", args.figure_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

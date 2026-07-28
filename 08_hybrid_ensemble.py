"""M5.2: coverage-balanced static-baseline + LinUCB review ensemble.

The static model and LinUCB have complementary scenario coverage: the former
detects Scenario 4 while the online agent largely detects Scenario 3. This
script constructs a fixed-capacity hybrid queue without retraining either
model: it unions a static-score allocation with an existing LinUCB review
queue, then fills any overlap-created spare capacity with the next-best static
scores. Labels are never used to select emails; they are joined only for the
offline report.

Usage:
    python 08_hybrid_ensemble.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config


def load_baseline_scores(path: Path) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=["id", "date", "user", "is_malicious", "risk_score"], chunksize=200_000, dtype={"id": str}):
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-budget", type=float, default=0.005, help="Share allocated to the static baseline.")
    parser.add_argument("--target-budget", type=float, default=0.01, help="Total ensemble review capacity.")
    parser.add_argument("--linucb-prefix", default="linucb_0p5", help="Existing LinUCB run whose budget matches 1 - static budget.")
    parser.add_argument("--output-prefix", default="hybrid_0p5_static_0p5_linucb")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.static_budget < args.target_budget <= 1:
        parser.error("Require 0 < --static-budget < --target-budget <= 1.")
    outputs = config.OUTPUT_DIR
    score_summary = json.loads((outputs / "baseline_evaluation.json").read_text())
    n_test = score_summary["n_test"]
    static_k = round(n_test * args.static_budget)
    target_k = round(n_test * args.target_budget)
    output_csv = outputs / f"{args.output_prefix}_reviewed_emails.csv"
    output_json = outputs / f"{args.output_prefix}_summary.json"
    if (output_csv.exists() or output_json.exists()) and not args.overwrite:
        parser.error(f"Output exists; use --overwrite: {output_csv}, {output_json}")

    baseline = load_baseline_scores(Path(score_summary["score_output"]))
    static_selection = baseline.nlargest(static_k, "risk_score")
    linucb_path = outputs / f"{args.linucb_prefix}_reviewed_emails.csv"
    if not linucb_path.exists():
        raise FileNotFoundError(f"LinUCB review log not found: {linucb_path}")
    linucb = pd.read_csv(linucb_path, usecols=["id", "date", "user", "is_malicious", "ucb_score"], dtype={"id": str})
    linucb["selection_source"] = "LinUCB"
    static_selection = static_selection.assign(selection_source="Static baseline")
    ensemble = pd.concat([static_selection, linucb], ignore_index=True, sort=False)
    ensemble = ensemble.drop_duplicates("id", keep="first")
    spare = target_k - len(ensemble)
    if spare > 0:
        filler = baseline.loc[~baseline["id"].isin(ensemble["id"])].nlargest(spare, "risk_score")
        filler = filler.assign(selection_source="Static baseline fill")
        ensemble = pd.concat([ensemble, filler], ignore_index=True, sort=False)
    ensemble = ensemble.head(target_k)
    scenario_lookup = pd.read_csv(config.GROUND_TRUTH_LOOKUP, dtype={"id": str})[["id", "scenario"]]
    ensemble = ensemble.merge(scenario_lookup, on="id", how="left")
    ensemble["scenario"] = ensemble["scenario"].fillna(0).astype(int)
    hits = ensemble.loc[ensemble["is_malicious"].eq(1), "scenario"].value_counts().sort_index()
    true_positives = int(ensemble["is_malicious"].sum())
    summary = {
        "method": "coverage-balanced static + LinUCB hybrid",
        "n_test": n_test, "target_budget": args.target_budget, "n_reviewed": len(ensemble),
        "static_budget": args.static_budget, "linucb_source": args.linucb_prefix,
        "true_positives": true_positives, "available_malicious": int(score_summary["n_test_malicious"]),
        "precision_at_budget": true_positives / len(ensemble),
        "recall_at_budget": true_positives / score_summary["n_test_malicious"],
        "false_positive_rate_at_budget": (len(ensemble) - true_positives) / (n_test - score_summary["n_test_malicious"]),
        "scenario_true_positives": {str(key): int(value) for key, value in hits.items()},
        "static_initial_selection": len(static_selection), "linucb_initial_selection": len(linucb),
        "overlap_removed": len(static_selection) + len(linucb) - len(pd.concat([static_selection, linucb], ignore_index=True).drop_duplicates("id")),
        "static_fill_selection": max(spare, 0),
    }
    ensemble.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[info] wrote {output_csv}")


if __name__ == "__main__":
    main()

"""M5.2: coverage-balanced static-baseline + LinUCB review ensemble.

The static model and LinUCB have complementary scenario coverage: the former
detects Scenario 4 while the online agent largely detects Scenario 3. This
script constructs a fixed-capacity hybrid queue without retraining either
model. The allocation is made separately within every calendar day: it unions
a static-score allocation with an existing LinUCB review queue, then fills any
overlap-created spare capacity with the next-best static scores from that same
day. Labels are never used to select emails; they are joined only for the
offline report.

Usage:
    python 08_hybrid_ensemble.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

import config


def load_baseline_scores(path: Path) -> pd.DataFrame:
    parts = []
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=["id", "date", "user", "is_malicious", "risk_score"], chunksize=200_000, dtype={"id": str}),
        start=1,
    ):
        parts.append(chunk)
        print(f"[hybrid] loaded {chunk_index * 200_000:,} static-score rows")
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
    output_csv = outputs / f"{args.output_prefix}_reviewed_emails.csv"
    output_json = outputs / f"{args.output_prefix}_summary.json"
    if (output_csv.exists() or output_json.exists()) and not args.overwrite:
        parser.error(f"Output exists; use --overwrite: {output_csv}, {output_json}")

    baseline = load_baseline_scores(Path(score_summary["score_output"]))
    linucb_path = outputs / f"{args.linucb_prefix}_reviewed_emails.csv"
    if not linucb_path.exists():
        raise FileNotFoundError(f"LinUCB review log not found: {linucb_path}")
    linucb = pd.read_csv(linucb_path, usecols=["id", "date", "user", "is_malicious", "ucb_score"], dtype={"id": str})
    # Source timestamps include seconds.  Capacity is defined per calendar day,
    # so never group directly on the raw timestamp string.
    baseline["_day"] = baseline["date"].str.slice(0, 10)
    linucb["_day"] = linucb["date"].str.slice(0, 10)
    linucb_by_day = {
        day: rows.assign(selection_source="LinUCB")
        for day, rows in linucb.groupby("_day", sort=False)
    }

    daily_queues = []
    static_initial_selection = linucb_initial_selection = 0
    overlap_removed = static_fill_selection = 0
    for day_index, (day, static_day) in enumerate(baseline.groupby("_day", sort=False), start=1):
        target_quota = max(1, math.ceil(len(static_day) * args.target_budget))
        static_quota = max(1, math.ceil(len(static_day) * args.static_budget))
        static_selection = static_day.drop(columns="_day").nlargest(static_quota, "risk_score").assign(
            selection_source="Static baseline"
        )
        linucb_selection = linucb_by_day.get(day, pd.DataFrame(columns=linucb.columns))
        static_initial_selection += len(static_selection)
        linucb_initial_selection += len(linucb_selection)

        # Static is admitted first, then non-overlapping adaptive selections.
        # This makes the allocation deterministic if the half-budgets round up.
        combined = pd.concat([static_selection, linucb_selection], ignore_index=True, sort=False)
        deduplicated = combined.drop_duplicates("id", keep="first")
        overlap_removed += len(combined) - len(deduplicated)
        queue = deduplicated.head(target_quota)
        spare = target_quota - len(queue)
        if spare > 0:
            filler = static_day.drop(columns="_day").loc[~static_day["id"].isin(queue["id"])].nlargest(spare, "risk_score")
            filler = filler.assign(selection_source="Static baseline fill")
            queue = pd.concat([queue, filler], ignore_index=True, sort=False)
            static_fill_selection += len(filler)
        daily_queues.append(queue)
        if day_index % 25 == 0:
            print(f"[hybrid] {day_index} calendar days combined")

    ensemble = pd.concat(daily_queues, ignore_index=True)
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
        "static_initial_selection": static_initial_selection,
        "linucb_initial_selection": linucb_initial_selection,
        "overlap_removed": overlap_removed,
        "static_fill_selection": static_fill_selection,
    }
    ensemble.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[info] wrote {output_csv}")


if __name__ == "__main__":
    main()

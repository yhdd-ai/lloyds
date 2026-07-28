"""Audit the static baseline with the same daily capacity convention as LinUCB.

The original baseline uses global held-out score cut-offs. This script instead
selects the highest static scores within each calendar day at 0.1%, 0.5%, and
1.0% capacities, matching the online agent's operational review convention.
It reuses ``baseline_scored_emails.csv`` and does not refit any model.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config

BUDGETS = (0.001, 0.005, 0.01)


def evaluate_day(day: pd.DataFrame, scenario_by_id: dict[str, int], totals: dict, daily_rows: list[dict]) -> None:
    scores = day["risk_score"].to_numpy(float)
    labels = pd.to_numeric(day["is_malicious"], errors="coerce").fillna(0).astype(int).to_numpy()
    for budget in BUDGETS:
        quota = max(1, math.ceil(len(day) * budget))
        selected = np.argpartition(scores, -quota)[-quota:]
        selected_labels = labels[selected]
        t = totals[budget]
        t["emails"] += len(day); t["reviewed"] += quota
        t["true_positives"] += int(selected_labels.sum())
        t["false_positives"] += int(quota - selected_labels.sum())
        t["available_malicious"] += int(labels.sum())
        for email_id in day.iloc[selected]["id"].iloc[np.flatnonzero(selected_labels)].tolist():
            t["scenario_hits"][scenario_by_id.get(email_id, 0)] += 1
        daily_rows.append({
            "date": day["date"].iloc[0][:10], "budget": budget, "emails": len(day),
            "reviewed": quota, "true_positives": int(selected_labels.sum()),
            "available_malicious": int(labels.sum()),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-path", type=Path, default=config.OUTPUT_DIR / "baseline_scored_emails.csv")
    parser.add_argument("--summary-output", type=Path, default=config.OUTPUT_DIR / "daily_baseline_evaluation.json")
    parser.add_argument("--daily-output", type=Path, default=config.OUTPUT_DIR / "daily_baseline_metrics.csv")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    existing = [str(path) for path in [args.summary_output, args.daily_output] if path.exists()]
    if existing and not args.overwrite:
        parser.error("Output already exists; use --overwrite: " + ", ".join(existing))
    lookup = pd.read_csv(config.GROUND_TRUTH_LOOKUP, dtype={"id": str})
    scenario_by_id = dict(zip(lookup["id"], lookup["scenario"].astype(int), strict=False))
    totals = {budget: {"emails": 0, "reviewed": 0, "true_positives": 0, "false_positives": 0, "available_malicious": 0, "scenario_hits": Counter()} for budget in BUDGETS}
    pending, current_day, daily_rows = [], None, []
    for chunk_index, chunk in enumerate(pd.read_csv(args.score_path, usecols=["id", "date", "is_malicious", "risk_score"], chunksize=args.chunksize, dtype={"id": str})):
        chunk["_day"] = chunk["date"].str.slice(0, 10)
        for day_name, group in chunk.groupby("_day", sort=False):
            group = group.drop(columns="_day")
            if current_day is None:
                current_day = day_name
            if day_name != current_day:
                evaluate_day(pd.concat(pending, ignore_index=True), scenario_by_id, totals, daily_rows)
                pending, current_day = [], day_name
            pending.append(group)
        print(f"[daily baseline] chunk {chunk_index + 1}: {len(daily_rows):,} completed days")
    if pending:
        evaluate_day(pd.concat(pending, ignore_index=True), scenario_by_id, totals, daily_rows)
    summaries = {}
    for budget, total in totals.items():
        summaries[f"{budget:.1%}"] = {
            "n_test": total["emails"], "n_reviewed": total["reviewed"],
            "true_positives": total["true_positives"], "available_malicious": total["available_malicious"],
            "precision_at_budget": total["true_positives"] / total["reviewed"],
            "recall_at_budget": total["true_positives"] / total["available_malicious"],
            "false_positive_rate_at_budget": total["false_positives"] / max(total["emails"] - total["available_malicious"], 1),
            "scenario_true_positives": {str(key): int(value) for key, value in sorted(total["scenario_hits"].items())},
        }
    result = {"method": "static anomaly baseline with daily review capacity", "budget_metrics": summaries}
    args.summary_output.write_text(json.dumps(result, indent=2))
    pd.DataFrame(daily_rows).to_csv(args.daily_output, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

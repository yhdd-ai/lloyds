"""Paired calendar-day bootstrap comparison of static scoring and LinUCB.

The interval is descriptive: calendar days are resampled as paired units so
that each sampled static outcome stays paired with the LinUCB outcome observed
on the same day. It does not assert independent deployment replications.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


OUTPUTS = Path(__file__).resolve().parent / "outputs"
CONFIGURATIONS = ((0.001, "0p1"), (0.005, "0p5"), (0.01, "1p0"))


def load_true_positives(path: Path, budget: float | None = None) -> dict[str, int]:
    with path.open(newline="") as handle:
        return {
            row["date"]: int(row["true_positives"])
            for row in csv.DictReader(handle)
            if budget is None or abs(float(row["budget"]) - budget) < 1e-12
        }


def percentile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows: list[dict[str, int | float]] = []
    for budget, suffix in CONFIGURATIONS:
        static = load_true_positives(OUTPUTS / "daily_baseline_metrics.csv", budget)
        linucb = load_true_positives(OUTPUTS / f"linucb_{suffix}_daily_metrics.csv")
        dates = sorted(set(static) & set(linucb))
        if len(dates) != len(static) or len(dates) != len(linucb):
            raise ValueError(f"calendar-day mismatch at review budget {budget}")
        day_differences = [linucb[date] - static[date] for date in dates]
        samples = [sum(rng.choice(day_differences) for _ in dates) for _ in range(args.replicates)]
        rows.append({
            "review_budget": budget,
            "n_days": len(dates),
            "static_true_positives": sum(static.values()),
            "linucb_true_positives": sum(linucb.values()),
            "linucb_minus_static_true_positives": sum(day_differences),
            "bootstrap_95ci_lower": round(percentile(samples, 0.025), 2),
            "bootstrap_95ci_upper": round(percentile(samples, 0.975), 2),
            "bootstrap_probability_linucb_greater": round(sum(value > 0 for value in samples) / args.replicates, 4),
        })
    fieldnames = list(rows[0])
    with (OUTPUTS / "bootstrap_policy_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUTS / "bootstrap_policy_comparison.json").open("w") as handle:
        json.dump({"method": "paired calendar-day percentile bootstrap", "replicates": args.replicates, "seed": args.seed, "results": rows}, handle, indent=2)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

"""M3 static, interpretable anomaly baseline for CERT r6.2 email features.

Fits label-free per-user statistics through the inclusive ``--train-end``
(default 2010-07-31, before the first email-based malicious scenario), freezes
them, then scores later emails. Labels are used only after scoring to evaluate
the ranking at 0.1%, 0.5%, and 1% analyst-review budgets.

Usage:
    python 04_anomaly_baseline.py --limit-chunks 3 --no-score-output
    python 04_anomaly_baseline.py
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config

NUMERIC_COLUMNS = ["size", "attachment_count", "n_recipients_internal", "n_recipients_external", "semantic_deviation"]
RARITY_COLUMNS = ["has_external_recipient", "is_after_hours", "is_weekend"]
MIN_USER_HISTORY, EPSILON, MAX_Z_SCORE, RARITY_WEIGHT = 30, 1e-3, 6.0, 0.5
DEFAULT_BUDGETS = (0.001, 0.005, 0.01)


def transformed_numeric_features(chunk: pd.DataFrame) -> np.ndarray:
    """Log-transform skewed count features; semantic deviation is unchanged."""
    cols = [
        np.log1p(pd.to_numeric(chunk["size"], errors="coerce").to_numpy(float)),
        np.log1p(pd.to_numeric(chunk["attachment_count"], errors="coerce").to_numpy(float)),
        np.log1p(pd.to_numeric(chunk["n_recipients_internal"], errors="coerce").to_numpy(float)),
        np.log1p(pd.to_numeric(chunk["n_recipients_external"], errors="coerce").to_numpy(float)),
        pd.to_numeric(chunk["semantic_deviation"], errors="coerce").to_numpy(float),
    ]
    return np.column_stack(cols)


class FeatureMoments:
    """Streaming per-user count/sum/sum-of-squares feature statistics."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.count = defaultdict(lambda: np.zeros(n_features, dtype=np.int64))
        self.sum = defaultdict(lambda: np.zeros(n_features, dtype=float))
        self.sum_sq = defaultdict(lambda: np.zeros(n_features, dtype=float))

    def update(self, users: np.ndarray, values: np.ndarray) -> None:
        for user, value in zip(users, values, strict=False):
            valid = np.isfinite(value)
            if valid.any():
                self.count[user][valid] += 1
                self.sum[user][valid] += value[valid]
                self.sum_sq[user][valid] += np.square(value[valid])

    def mean_std(self, user: str, fallback: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        count = self.count.get(user)
        if count is None or np.any(count < MIN_USER_HISTORY):
            return fallback
        mean = self.sum[user] / count
        variance = np.maximum(self.sum_sq[user] / count - np.square(mean), EPSILON)
        return mean, np.sqrt(variance)


def global_mean_std(stats: FeatureMoments) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(stats.n_features, dtype=np.int64)
    sums = np.zeros(stats.n_features)
    sum_sqs = np.zeros(stats.n_features)
    for user in stats.count:
        counts += stats.count[user]
        sums += stats.sum[user]
        sum_sqs += stats.sum_sq[user]
    if np.any(counts == 0):
        raise RuntimeError("At least one scoring feature has no usable training values.")
    mean = sums / counts
    variance = np.maximum(sum_sqs / counts - np.square(mean), EPSILON)
    return mean, np.sqrt(variance)


def fit_statistics(path: Path, train_end: pd.Timestamp, chunksize: int, limit: int | None):
    numeric, rarity = FeatureMoments(len(NUMERIC_COLUMNS)), FeatureMoments(len(RARITY_COLUMNS))
    n_train = n_test = n_positive = 0
    usecols = ["date", "user", "is_malicious", *NUMERIC_COLUMNS, *RARITY_COLUMNS]
    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize)):
        dates = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        train_mask = (dates <= train_end).to_numpy()
        if train_mask.any():
            train = chunk.loc[train_mask]
            numeric.update(train["user"].to_numpy(), transformed_numeric_features(train))
            rarity.update(train["user"].to_numpy(), train[RARITY_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float))
            n_train += len(train)
        test_mask = ~train_mask
        n_test += int(test_mask.sum())
        n_positive += int(pd.to_numeric(chunk.loc[test_mask, "is_malicious"], errors="coerce").fillna(0).sum())
        print(f"[fit] chunk {i + 1}: {n_train:,} training rows, {n_test:,} test rows")
        if limit is not None and i + 1 >= limit:
            print(f"[info] --limit-chunks={limit} reached during fitting")
            break
    if n_train == 0:
        raise RuntimeError(f"No rows on or before {train_end.date()} were found; use a later --train-end or the complete feature table.")
    return numeric, rarity, n_train, n_test, n_positive


def score_chunk(chunk, numeric_stats, rarity_stats, numeric_fallback, rarity_fallback) -> pd.DataFrame:
    values = transformed_numeric_features(chunk)
    rarity_values = chunk[RARITY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    numeric_score, rarity_score, global_fallback = np.zeros(len(chunk)), np.zeros(len(chunk)), np.zeros(len(chunk), dtype=int)
    for row, user in enumerate(chunk["user"].to_numpy()):
        means, stds = numeric_stats.mean_std(user, numeric_fallback)
        counts = numeric_stats.count.get(user)
        global_fallback[row] = int(counts is None or np.any(counts < MIN_USER_HISTORY))
        valid = np.isfinite(values[row])
        if valid.any():
            z = np.minimum(np.abs((values[row, valid] - means[valid]) / stds[valid]), MAX_Z_SCORE)
            numeric_score[row] = z.mean()
        rarity_means, _ = rarity_stats.mean_std(user, rarity_fallback)
        rarity_score[row] = np.sum(rarity_values[row] * -np.log(np.maximum(rarity_means, EPSILON)))
    result = chunk[["id", "date", "user", "is_malicious"]].copy()
    result["numeric_anomaly"] = numeric_score
    result["rarity_anomaly"] = rarity_score
    result["risk_score"] = numeric_score + RARITY_WEIGHT * rarity_score
    result["uses_global_fallback"] = global_fallback
    return result


def budget_metrics(top_rows, n_test, n_positive, budgets):
    top_rows.sort(reverse=True)
    output = {}
    for budget in budgets:
        k = max(1, math.ceil(n_test * budget))
        selected = top_rows[:k]
        tp = sum(label for _, label in selected)
        output[f"{budget:.1%}"] = {
            "n_reviewed": k,
            "true_positives": tp,
            "precision_at_budget": tp / k,
            "recall_at_budget": tp / n_positive if n_positive else None,
            "false_positive_rate_at_budget": (k - tp) / max(n_test - n_positive, 1),
        }
    return output


def run_baseline(feature_path, train_end, chunksize, limit, score_output, summary_output):
    print(f"[info] fitting a label-free baseline through {train_end.date()} ...")
    numeric, rarity, n_train, n_test, n_positive = fit_statistics(feature_path, train_end, chunksize, limit)
    numeric_fallback, rarity_fallback = global_mean_std(numeric), global_mean_std(rarity)
    top_k = max(1, math.ceil(n_test * max(DEFAULT_BUDGETS)))
    top_rows, header_written = [], False
    if score_output is not None:
        score_output.parent.mkdir(parents=True, exist_ok=True)
        if score_output.exists():
            score_output.unlink()
    usecols = ["id", "date", "user", "is_malicious", *NUMERIC_COLUMNS, *RARITY_COLUMNS]
    for i, chunk in enumerate(pd.read_csv(feature_path, usecols=usecols, chunksize=chunksize)):
        dates = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        test = chunk.loc[(dates > train_end).to_numpy()]
        if not test.empty:
            scored = score_chunk(test, numeric, rarity, numeric_fallback, rarity_fallback)
            labels = pd.to_numeric(scored["is_malicious"], errors="coerce").fillna(0).astype(int).to_numpy()
            for score, label in zip(scored["risk_score"].to_numpy(float), labels, strict=False):
                item = (float(score), int(label))
                if len(top_rows) < top_k:
                    heapq.heappush(top_rows, item)
                elif item[0] > top_rows[0][0]:
                    heapq.heapreplace(top_rows, item)
            if score_output is not None:
                scored.to_csv(score_output, mode="a", header=not header_written, index=False)
                header_written = True
        print(f"[score] chunk {i + 1}: top {len(top_rows):,}/{top_k:,} candidate alerts retained")
        if limit is not None and i + 1 >= limit:
            print(f"[info] --limit-chunks={limit} reached during scoring")
            break
    result = {
        "method": "label-free per-user static anomaly baseline",
        "feature_file": str(feature_path), "train_end": train_end.strftime("%Y-%m-%d"),
        "n_train": n_train, "n_test": n_test, "n_test_malicious": n_positive,
        "min_user_history": MIN_USER_HISTORY, "numeric_features": NUMERIC_COLUMNS,
        "rarity_features": RARITY_COLUMNS, "rarity_weight": RARITY_WEIGHT,
        "budget_metrics": budget_metrics(top_rows, n_test, n_positive, DEFAULT_BUDGETS),
        "score_output": str(score_output) if score_output is not None else None,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=config.FEATURES_OUTPUT)
    parser.add_argument("--train-end", default="2010-07-31", help="Inclusive training end date (YYYY-MM-DD).")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit-chunks", type=int, default=None, help="Development-only limit for both passes.")
    parser.add_argument("--no-score-output", action="store_true", help="Do not write per-email scores.")
    parser.add_argument("--score-output", type=Path, default=config.OUTPUT_DIR / "baseline_scored_emails.csv")
    parser.add_argument("--summary-output", type=Path, default=config.OUTPUT_DIR / "baseline_evaluation.json")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing output files.")
    args = parser.parse_args()
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be at least 1")
    train_end = pd.Timestamp(args.train_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    score_output = None if args.no_score_output else args.score_output
    outputs = [args.summary_output] + ([score_output] if score_output is not None else [])
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error("Output already exists; use --overwrite: " + ", ".join(existing))
    result = run_baseline(args.feature_path, train_end, args.chunksize, args.limit_chunks, score_output, args.summary_output)
    print("\n" + "=" * 70 + "\nStatic anomaly baseline summary\n" + "=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

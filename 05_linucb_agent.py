"""M4.1: budget-constrained LinUCB replay for CERT r6.2 email features.

The first pass obtains feature scaling statistics using only emails through
``--train-end`` (default 2010-07-31). The second pass replays later emails in
chronological, one-day batches. At the start of each day, the agent ranks all
that day's emails by an upper-confidence-bound score and sends only the top
``--review-budget`` fraction to review. It learns only from labels returned
for reviewed emails; labels of released emails are used after decisions solely
for offline evaluation.

Usage
-----
    python 05_linucb_agent.py --limit-chunks 3 --no-output
    python 05_linucb_agent.py --review-budget 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config


BASE_COLUMNS = [
    "size", "attachment_count", "n_recipients_internal",
    "n_recipients_external", "semantic_deviation", "hour_of_day",
    "is_after_hours", "is_weekend", "has_external_recipient",
]
EMBED_COLUMNS = [f"embed_pc_{i}" for i in range(config.PCA_COMPONENTS)]
CONTEXT_COLUMNS = [*BASE_COLUMNS, *EMBED_COLUMNS]


def feature_matrix(chunk: pd.DataFrame) -> np.ndarray:
    """Build compact continuous contexts; count columns receive log transforms."""
    data = chunk[CONTEXT_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    for index in range(4):  # size, attachments, internal/external recipients
        data[:, index] = np.log1p(data[:, index])
    return data


def fit_scaler(path: Path, train_end: pd.Timestamp, chunksize: int, limit: int | None):
    """Label-free global train-period standardisation statistics."""
    count = np.zeros(len(CONTEXT_COLUMNS), dtype=np.int64)
    total = np.zeros(len(CONTEXT_COLUMNS), dtype=float)
    total_sq = np.zeros(len(CONTEXT_COLUMNS), dtype=float)
    n_train = 0
    usecols = ["date", *CONTEXT_COLUMNS]
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize)):
        dates = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        train = chunk.loc[(dates <= train_end).to_numpy()]
        if not train.empty:
            values = feature_matrix(train)
            valid = np.isfinite(values)
            count += valid.sum(axis=0)
            total += np.where(valid, values, 0).sum(axis=0)
            total_sq += np.where(valid, np.square(values), 0).sum(axis=0)
            n_train += len(train)
        print(f"[scale] chunk {chunk_index + 1}: {n_train:,} training rows")
        if limit is not None and chunk_index + 1 >= limit:
            break
    if n_train == 0 or np.any(count == 0):
        raise RuntimeError("Training period has no usable values for one or more context features.")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-6)
    return mean, np.sqrt(variance), n_train


def scaled_context(chunk: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = feature_matrix(chunk)
    # First emails have no semantic history; train-period mean represents
    # "typical/unknown" rather than treating missingness as an extreme value.
    values = np.where(np.isfinite(values), values, mean)
    standardised = (values - mean) / std
    return np.column_stack((np.ones(len(chunk)), standardised))  # intercept


class LinUCB:
    """One-action linear UCB model: expected value of sending an email to review."""

    def __init__(self, dimensions: int, alpha: float, ridge: float):
        self.alpha = alpha
        self.a_inv = np.eye(dimensions) / ridge
        self.b = np.zeros(dimensions)

    def score(self, contexts: np.ndarray) -> np.ndarray:
        theta = self.a_inv @ self.b
        mean_reward = contexts @ theta
        uncertainty = np.sqrt(np.einsum("ij,jk,ik->i", contexts, self.a_inv, contexts))
        return mean_reward + self.alpha * uncertainty

    def update(self, context: np.ndarray, reward: float) -> None:
        # Sherman-Morrison update avoids a matrix inversion for each review.
        projected = self.a_inv @ context
        self.a_inv -= np.outer(projected, projected) / (1.0 + context @ projected)
        self.b += reward * context


def process_day(day: pd.DataFrame, agent: LinUCB, review_budget: float, reward_true_positive: float, review_cost: float):
    contexts = np.vstack(day.pop("_context").to_numpy())
    scores = agent.score(contexts)
    quota = max(1, math.ceil(len(day) * review_budget))
    chosen = np.argpartition(scores, -quota)[-quota:]
    chosen = chosen[np.argsort(chosen)]  # feedback enters in original within-day order
    labels = pd.to_numeric(day["is_malicious"], errors="coerce").fillna(0).astype(int).to_numpy()
    reviewed_labels = labels[chosen]
    rewards = np.where(reviewed_labels == 1, reward_true_positive, -review_cost)
    for row, reward in zip(chosen, rewards, strict=False):
        agent.update(contexts[row], float(reward))
    reviewed = day.iloc[chosen][["id", "date", "user", "is_malicious"]].copy()
    reviewed["ucb_score"] = scores[chosen]
    reviewed["review_reward"] = rewards
    oracle_true_positives = min(quota, int(labels.sum()))
    metrics = {
        "date": day["date"].iloc[0][:10],
        "emails": len(day), "reviewed": quota,
        "true_positives": int(reviewed_labels.sum()),
        "false_positives": int(quota - reviewed_labels.sum()),
        "available_malicious": int(labels.sum()),
        "oracle_true_positives": oracle_true_positives,
        "reward": float(rewards.sum()),
        "oracle_reward": float(oracle_true_positives * reward_true_positive - (quota - oracle_true_positives) * review_cost),
    }
    return metrics, reviewed


def replay(path: Path, train_end: pd.Timestamp, mean: np.ndarray, std: np.ndarray, args) -> tuple[list[dict], pd.DataFrame]:
    usecols = ["id", "date", "user", "is_malicious", *CONTEXT_COLUMNS]
    agent = LinUCB(len(CONTEXT_COLUMNS) + 1, args.alpha, args.ridge)
    daily_metrics, reviewed_parts, current_date, pending = [], [], None, []
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=args.chunksize)):
        timestamps = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.loc[(timestamps > train_end).to_numpy()].copy()
        if not chunk.empty:
            chunk["_day"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S").dt.strftime("%Y-%m-%d")
            for day_name, group in chunk.groupby("_day", sort=False):
                group = group.drop(columns="_day")
                if current_date is None:
                    current_date = day_name
                if day_name != current_date:
                    day = pd.concat(pending, ignore_index=True)
                    day["_context"] = list(scaled_context(day, mean, std))
                    metrics, reviewed = process_day(day, agent, args.review_budget, args.true_positive_reward, args.review_cost)
                    daily_metrics.append(metrics)
                    reviewed_parts.append(reviewed)
                    pending, current_date = [], day_name
                pending.append(group)
        print(f"[replay] chunk {chunk_index + 1}: {len(daily_metrics):,} completed days")
        if args.limit_chunks is not None and chunk_index + 1 >= args.limit_chunks:
            break
    if pending:
        day = pd.concat(pending, ignore_index=True)
        day["_context"] = list(scaled_context(day, mean, std))
        metrics, reviewed = process_day(day, agent, args.review_budget, args.true_positive_reward, args.review_cost)
        daily_metrics.append(metrics)
        reviewed_parts.append(reviewed)
    return daily_metrics, pd.concat(reviewed_parts, ignore_index=True)


def summarise(daily_metrics: list[dict], args, n_train: int) -> dict:
    total = {key: sum(day[key] for day in daily_metrics) for key in ["emails", "reviewed", "true_positives", "false_positives", "available_malicious", "oracle_true_positives", "reward", "oracle_reward"]}
    return {
        "method": "budget-constrained LinUCB daily replay",
        "train_end": args.train_end, "n_train": n_train, "n_test": total["emails"],
        "review_budget": args.review_budget, "alpha": args.alpha, "ridge": args.ridge,
        "review_cost": args.review_cost, "true_positive_reward": args.true_positive_reward,
        "n_reviewed": total["reviewed"], "true_positives": total["true_positives"],
        "available_malicious": total["available_malicious"],
        "precision_at_budget": total["true_positives"] / total["reviewed"] if total["reviewed"] else None,
        "recall_at_budget": total["true_positives"] / total["available_malicious"] if total["available_malicious"] else None,
        "false_positive_rate_at_budget": total["false_positives"] / max(total["emails"] - total["available_malicious"], 1),
        "cumulative_reward": total["reward"], "oracle_reward": total["oracle_reward"],
        "cumulative_regret": total["oracle_reward"] - total["reward"], "n_days": len(daily_metrics),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=config.FEATURES_OUTPUT)
    parser.add_argument("--train-end", default="2010-07-31")
    parser.add_argument("--review-budget", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0, help="UCB exploration multiplier.")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--true-positive-reward", type=float, default=10.0)
    parser.add_argument("--review-cost", type=float, default=1.0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--output-prefix", default="linucb", help="Prefix for result files in outputs/.")
    parser.add_argument("--daily-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--reviewed-output", type=Path, default=None, help="Per-review audit trail, including scenario labels.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.review_budget <= 1:
        parser.error("--review-budget must be in (0, 1].")
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be at least 1.")
    args.daily_output = args.daily_output or config.OUTPUT_DIR / f"{args.output_prefix}_daily_metrics.csv"
    args.summary_output = args.summary_output or config.OUTPUT_DIR / f"{args.output_prefix}_summary.json"
    args.reviewed_output = args.reviewed_output or config.OUTPUT_DIR / f"{args.output_prefix}_reviewed_emails.csv"
    outputs = [] if args.no_output else [args.daily_output, args.summary_output, args.reviewed_output]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error("Output already exists; use --overwrite: " + ", ".join(existing))
    train_end = pd.Timestamp(args.train_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if not args.feature_path.exists():
        parser.error(f"Feature file not found: {args.feature_path}")
    mean, std, n_train = fit_scaler(args.feature_path, train_end, args.chunksize, args.limit_chunks)
    daily_metrics, reviewed_emails = replay(args.feature_path, train_end, mean, std, args)
    summary = summarise(daily_metrics, args, n_train)
    if not args.no_output:
        scenarios = pd.read_csv(config.GROUND_TRUTH_LOOKUP, dtype={"id": str})[["id", "scenario"]]
        reviewed_emails = reviewed_emails.merge(scenarios, on="id", how="left")
        reviewed_emails["scenario"] = reviewed_emails["scenario"].fillna(0).astype(int)
        scenario_hits = reviewed_emails.loc[reviewed_emails["is_malicious"].eq(1), "scenario"].value_counts().sort_index()
        summary["scenario_true_positives"] = {str(k): int(v) for k, v in scenario_hits.items()}
        summary["reviewed_output"] = str(args.reviewed_output)
        pd.DataFrame(daily_metrics).to_csv(args.daily_output, index=False)
        reviewed_emails.to_csv(args.reviewed_output, index=False)
        args.summary_output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

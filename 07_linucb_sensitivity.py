"""M5.1: one-pass LinUCB sensitivity analysis at a fixed review budget.

Four policies are replayed in parallel over the same chronologically ordered
emails, so the 7.8GB feature file is read only twice (one scaler pass and one
replay pass):

* alpha=0.25, reward=10: low exploration
* alpha=1.00, reward=10: default policy
* alpha=2.00, reward=10: high exploration
* alpha=1.00, reward=100: high value for a confirmed threat

Each policy only learns from labels of emails it chose to review. Labels for
released emails are retained only for post-decision evaluation.

Usage:
    python 07_linucb_sensitivity.py --review-budget 0.01
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import config

linucb = importlib.import_module("05_linucb_agent")

POLICIES = (
    ("low_exploration", 0.25, 10.0),
    ("default", 1.00, 10.0),
    ("high_exploration", 2.00, 10.0),
    ("high_threat_value", 1.00, 100.0),
)


def totals(metrics: list[dict]) -> dict:
    keys = ["emails", "reviewed", "true_positives", "false_positives", "available_malicious", "oracle_true_positives", "reward", "oracle_reward"]
    return {key: sum(day[key] for day in metrics) for key in keys}


def process_pending_day(pending, agents, policy_args, scenario_by_id, daily, scenario_hits):
    day = pd.concat(pending, ignore_index=True)
    contexts = linucb.scaled_context(day, policy_args["scaler_mean"], policy_args["scaler_std"])
    for name, agent in agents.items():
        policy_day = day.copy()
        policy_day["_context"] = list(contexts)
        settings = policy_args[name]
        metrics, reviewed = linucb.process_day(
            policy_day, agent, settings.review_budget,
            settings.true_positive_reward, settings.review_cost,
        )
        daily[name].append(metrics)
        for email_id in reviewed.loc[reviewed["is_malicious"].eq(1), "id"]:
            scenario_hits[name][scenario_by_id.get(email_id, 0)] += 1


def replay_all(path: Path, train_end: pd.Timestamp, mean, std, args, scenario_by_id):
    policies = {
        name: SimpleNamespace(
            alpha=alpha, true_positive_reward=reward, review_cost=args.review_cost,
            review_budget=args.review_budget,
        )
        for name, alpha, reward in POLICIES
    }
    policy_args = {**policies, "scaler_mean": mean, "scaler_std": std}
    agents = {name: linucb.LinUCB(len(linucb.CONTEXT_COLUMNS) + 1, settings.alpha, args.ridge) for name, settings in policies.items()}
    daily = {name: [] for name in policies}
    scenario_hits = {name: Counter() for name in policies}
    current_date, pending = None, []
    usecols = ["id", "date", "user", "is_malicious", *linucb.CONTEXT_COLUMNS]
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=args.chunksize)):
        dates = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.loc[(dates > train_end).to_numpy()].copy()
        if not chunk.empty:
            chunk["_day"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S").dt.strftime("%Y-%m-%d")
            for day_name, group in chunk.groupby("_day", sort=False):
                group = group.drop(columns="_day")
                if current_date is None:
                    current_date = day_name
                if day_name != current_date:
                    process_pending_day(pending, agents, policy_args, scenario_by_id, daily, scenario_hits)
                    pending, current_date = [], day_name
                pending.append(group)
        print(f"[replay] chunk {chunk_index + 1}: {len(next(iter(daily.values()))):,} completed days")
        if args.limit_chunks is not None and chunk_index + 1 >= args.limit_chunks:
            break
    if pending:
        process_pending_day(pending, agents, policy_args, scenario_by_id, daily, scenario_hits)
    return policies, daily, scenario_hits


def summarise(policies, daily, scenario_hits, args, n_train):
    rows = []
    for name, settings in policies.items():
        result = totals(daily[name])
        rows.append({
            "policy": name, "review_budget": settings.review_budget,
            "alpha": settings.alpha, "true_positive_reward": settings.true_positive_reward,
            "review_cost": settings.review_cost, "n_train": n_train,
            "n_test": result["emails"], "n_reviewed": result["reviewed"],
            "true_positives": result["true_positives"], "available_malicious": result["available_malicious"],
            "precision": result["true_positives"] / result["reviewed"] if result["reviewed"] else None,
            "recall": result["true_positives"] / result["available_malicious"] if result["available_malicious"] else None,
            "false_positive_rate": result["false_positives"] / max(result["emails"] - result["available_malicious"], 1),
            "cumulative_reward": result["reward"], "oracle_reward": result["oracle_reward"],
            "cumulative_regret": result["oracle_reward"] - result["reward"],
            "scenario_2_true_positives": scenario_hits[name][2],
            "scenario_3_true_positives": scenario_hits[name][3],
            "scenario_4_true_positives": scenario_hits[name][4],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=config.FEATURES_OUTPUT)
    parser.add_argument("--train-end", default="2010-07-31")
    parser.add_argument("--review-budget", type=float, default=0.01)
    parser.add_argument("--review-cost", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit-chunks", type=int, default=None, help="Development-only limit for both passes.")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "linucb_sensitivity.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.review_budget <= 1:
        parser.error("--review-budget must be in (0, 1].")
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be at least 1.")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists; use --overwrite: {args.output}")
    train_end = pd.Timestamp(args.train_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mean, std, n_train = linucb.fit_scaler(args.feature_path, train_end, args.chunksize, args.limit_chunks)
    lookup = pd.read_csv(config.GROUND_TRUTH_LOOKUP, dtype={"id": str})
    scenario_by_id = dict(zip(lookup["id"], lookup["scenario"].astype(int), strict=False))
    policies, daily, scenario_hits = replay_all(args.feature_path, train_end, mean, std, args, scenario_by_id)
    result = summarise(policies, daily, scenario_hits, args, n_train)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"[info] wrote {args.output}")


if __name__ == "__main__":
    main()

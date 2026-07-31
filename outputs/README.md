# Compact results data

This directory contains the compact, reportable outputs needed to inspect the
dissertation results without downloading the multi-gigabyte CERT email archive.
All row-level email content, generated feature tables, static-score files, and
reviewed-email lists are intentionally excluded from GitHub.

## Key files

- `model_comparison.csv` — static anomaly and LinUCB results at 0.1%, 0.5%,
  and 1.0% daily review capacity.
- `scenario_coverage.csv` — Scenario 2/3/4 coverage at the 1% budget.
- `linucb_sensitivity.csv` — exploration and reward-scale sensitivity runs.
- `daily_baseline_metrics.csv` and `linucb_*_daily_metrics.csv` — the
  calendar-day metrics used for matched-capacity comparisons.
- `bootstrap_policy_comparison.csv` / `.json` — paired calendar-day percentile
  bootstrap summaries produced by `../10_bootstrap_policy_comparison.py`.
- `*_summary.json` — run-level diagnostics and summary statistics.
- `figures/*.svg` — vector figures used in the report.

## Reproducing from raw data

Obtain the CERT r6.2 email archive separately, place it according to
`../config.py`, install `../requirements.txt`, and run the numbered scripts in
the order listed in `../README.md`. The source archive and full intermediates
are excluded because they exceed practical GitHub repository limits.

## Interpretation

These outputs arise from a chronological offline replay on synthetic CERT data.
The feature pipeline uses causal semantic-deviation scoring and causal
IncrementalPCA: a chunk is transformed with components learned from prior
chunks and only then updates the PCA basis for future chunks. The bootstrap
intervals are descriptive and do not turn calendar days into independent
deployment replications.

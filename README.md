# python_code -- M1 (Data Exploration) + M2 (Feature Engineering)

## GitHub reproducibility bundle

This repository also includes a compact reproducibility bundle:

- `data/answers/` contains the CERT ground-truth answer files used to label
  malicious email records.
- `outputs/` contains compact JSON/CSV summaries and SVG figures supporting
  the reported evaluation. Large row-level intermediates are excluded.

The raw `emails_lloyds.zip` archive (about 3.3GB), generated
`outputs/email_features.csv` (about 7.8GB), and scored-email intermediates
are deliberately not committed because they exceed normal GitHub repository
limits. Place the CERT email archive next to this repository before running
the full pipeline. `config.py` automatically uses `data/answers/` when the
repository is cloned standalone.

Code for milestones M1 and M2 of the PRD (see `../PRD_保险内部威胁检测项目.md`,
section 10).

## Files

- `config.py` -- paths and dataset/feature constants. Edit `DATA_DIR` if your
  folder layout differs from the default (this file assumes it lives at
  `lloyds/python_code/`).
- `01_explore_email_ldap.py` -- streams `email.csv` in chunks (reads directly
  from inside `emails_lloyds.zip`, no need to extract the ~8GB file to disk)
  and combines the 19 monthly `LDAP/*.csv` snapshots. Produces:
  - `outputs/ldap_combined.csv`
  - `outputs/eda_summary.json`
- `02_match_ground_truth.py` -- reads `answers/insiders.csv` + the r6.2 scenario
  detail files, extracts every malicious **email** row's `id`, then scans
  `email.csv` to confirm those ids are present. Produces:
  - `outputs/malicious_email_ids.csv` (the ground-truth label lookup table
    that feature-engineering code joins against, keyed on `id`)
  - `outputs/ground_truth_match_report.json`
- `03_feature_engineering.py` -- **M2**. Single streaming pass over
  `email.csv` that produces, per email: metadata features (size, attachment
  count, internal/external recipient counts, after-hours/weekend flags),
  an LLM-embedding-derived per-user semantic deviation score (causal: computed
  against each user's *prior* emails only, then the baseline is updated), a
  PCA-reduced embedding (32-dim, fitted online via `IncrementalPCA` so the
  whole pass stays single-pass and memory-flat), and the ground-truth label
  joined from `malicious_email_ids.csv`. Produces `outputs/email_features.csv`
  -- the input table for Phase 2 (LinUCB) and Phase 2b (anomaly baseline).
  Requires `sentence-transformers` (not in `requirements.txt` by default,
  since it pulls in `torch`; add it yourself: `pip install sentence-transformers`).
- `04_anomaly_baseline.py` -- **M3**. Fits a label-free, per-user static
  anomaly baseline through 2010-07-31, then scores later emails. It reports
  Precision, Recall, and false-positive rate at 0.1%, 0.5%, and 1% review
  budgets; labels are used only after scoring for evaluation.
- `05_linucb_agent.py` -- **M4.1**. Replays post-training emails one day at a
  time, ranks them with LinUCB under a fixed analyst-review budget, and updates
  only from labels returned for reviewed emails.
- `06_results_analysis.py` -- **M5**. Produces the final static-baseline vs.
  LinUCB comparison tables and budget/scene coverage figures without rerunning
  either model.
- `07_linucb_sensitivity.py` -- **M5.1**. Replays four LinUCB exploration and
  reward configurations in parallel, reading the feature table only once after
  the shared scaling pass.
- `08_hybrid_ensemble.py` -- **M5.2**. Builds a fixed-capacity, coverage-
  balanced ensemble from completed static-baseline and LinUCB result logs.

## Setup

```bash
pip install -r requirements.txt
pip install sentence-transformers   # needed for 03_ (pulls in torch)
```

## Usage

```bash
# quick smoke tests (finish in seconds)
python 01_explore_email_ldap.py --limit-chunks 3
python 02_match_ground_truth.py --limit-chunks 20
python 03_feature_engineering.py --limit-chunks 2 --mock-embeddings   # pipeline test only, not real features
python 04_anomaly_baseline.py --limit-chunks 3 --no-score-output      # M3 pipeline test
python 05_linucb_agent.py --limit-chunks 3 --no-output                # M4.1 pipeline test
python 06_results_analysis.py                                          # M5 comparison tables and figures
python 07_linucb_sensitivity.py --review-budget 0.01                   # M5.1 sensitivity analysis
python 08_hybrid_ensemble.py                                            # M5.2 0.5% static + 0.5% LinUCB
# measures real embedding throughput without touching outputs/email_features.csv
python 03_feature_engineering.py --benchmark-chunks 3

# full runs (email.csv is ~11M rows / ~8GB -- on a normal machine this takes
# minutes, not the artificial few-second cap of a quick interactive sandbox)
python 01_explore_email_ldap.py
python 02_match_ground_truth.py
python 03_feature_engineering.py     # do NOT pass --mock-embeddings here
python 04_anomaly_baseline.py        # run after 03_ completes
python 05_linucb_agent.py --review-budget 0.01  # 1% LinUCB replay
# Three comparable budgets; each command writes a separate result set.
python 05_linucb_agent.py --review-budget 0.001 --output-prefix linucb_0p1
python 05_linucb_agent.py --review-budget 0.005 --output-prefix linucb_0p5
python 05_linucb_agent.py --review-budget 0.01  --output-prefix linucb_1p0
```

## Validated so far

**Full-scale runs (not just smoke tests) have now been completed against the
real data:**

- `01_explore_email_ldap.py`: **10,994,957 email rows**, 4,000 distinct
  users, 7,162,491 View / 3,832,466 Send. 23.57% of emails carry an
  attachment, 39.64% have at least one external recipient (mean 1.77 internal
  / 0.87 external recipients per email). The `content` field is confirmed to
  be **full sentences** (100% of a 300-row sample had sentence-ending
  punctuation and common stopwords, avg 88 words/email) -- matching the
  research proposal, not the "space-separated keyword list" wording in
  `readme.txt`.
- `02_match_ground_truth.py`: all **135/135 malicious email ids matched** in
  `email.csv` across scenarios 2, 3, and 4 (scenarios 1 and 5 have zero email
  rows, confirming the "3 of 5 scenarios" limitation already noted in the
  PRD). The ground-truth join key (`id`) works end-to-end for the full file,
  not just a sample.
- `03_feature_engineering.py`: pipeline logic (causal per-user deviation
  scoring, online PCA, label join, chunked CSV output) was smoke-tested with
  `--mock-embeddings` -- confirmed exactly one `NaN` deviation score per user
  (their first-ever email, correctly treated as cold-start) and correct
  weekday/after-hours flags. **Still needed**: a full run with the real
  `sentence-transformers` embedder (no `--mock-embeddings`) to produce the
  actual `outputs/email_features.csv` used from Phase 2 onward.

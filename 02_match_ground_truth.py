"""
02_match_ground_truth.py

M1 -- Ground-truth label matching for the CERT r6.2 email dataset.

Background
----------
`answers/insiders.csv` indexes every malicious scenario across all CERT
releases. For release r6.2 there are 5 scenarios (see PRD section 5), each
with its own detail file (e.g. `answers/r6.2-2.csv`) listing every raw log
row -- across logon/device/file/email/http -- that the scenario's insider
(or an associated user) generated during the attack window. Each row keeps
its original `id`, which also appears in the corresponding full log file
(email.csv, logon.csv, etc.). That shared `id` is the join key that turns
these answer files into row-level ground truth.

Only 3 of the 5 r6.2 scenarios (2, 3, 4) leave any trace in email.csv --
scenarios 1 and 5 are logon/device/file and http only. This script surfaces
that fact from the data itself rather than hard-coding it, so it stays
correct if the answer files ever change.

What this script does
----------------------
1. Reads `answers/insiders.csv`, filters to release r6.2.
2. For each r6.2 scenario, reads its detail file and keeps only the rows
   whose log_type == "email", recording the malicious email `id`s together
   with which scenario/insider they belong to.
3. Streams email.csv in chunks and flags every row whose `id` is in that
   malicious set.
4. Writes two artefacts to python_code/outputs/:
   - malicious_email_ids.csv   : one row per malicious email id + metadata
                                  (scenario, insider user, expected user/date
                                  from the answer file)
   - ground_truth_match_report.json : sanity-check counts (did every id from
                                  the answer files actually turn up in
                                  email.csv? how many total malicious emails?)

This script deliberately does NOT write a labelled copy of the full email.csv
(that would duplicate an ~8GB file for a handful of positive rows). Later
feature-engineering code should instead left-join its per-email or per-user
feature table against malicious_email_ids.csv on `id` / `user`.

Usage
-----
    python 02_match_ground_truth.py
    python 02_match_ground_truth.py --limit-chunks 3   # quick smoke test
"""
from __future__ import annotations

import argparse
import csv
import json

import pandas as pd

import config
from importlib import import_module

_explore = import_module("01_explore_email_ldap")
open_email_csv_chunks = _explore.open_email_csv_chunks
validate_schema = _explore.validate_schema


def load_r62_scenarios() -> pd.DataFrame:
    """Read answers/insiders.csv and keep only the r6.2 rows."""
    insiders = pd.read_csv(config.INSIDERS_INDEX, dtype=str)
    scenarios = insiders[insiders["dataset"] == config.RELEASE_TAG].copy()
    if scenarios.empty:
        raise RuntimeError(
            f"No scenarios found for dataset=={config.RELEASE_TAG!r} in "
            f"{config.INSIDERS_INDEX}"
        )
    print(f"[info] {len(scenarios)} scenario(s) found for release "
          f"{config.RELEASE_TAG}:")
    for _, row in scenarios.iterrows():
        print(f"        scenario {row['scenario']}: user={row['user']}, "
              f"details={row['details']}, window={row['start']} -> {row['end']}")
    return scenarios


def extract_malicious_email_ids(scenarios: pd.DataFrame) -> pd.DataFrame:
    """
    For every r6.2 scenario, read its detail file and keep only the email
    rows. Returns one row per malicious email id, tagged with scenario
    number, the scenario's insider user (from insiders.csv), the user field
    recorded on the email row itself (may differ -- e.g. scenario 3 involves
    a second user), and the row's date.
    """
    # Answer-file rows are ragged: every row starts with
    # (log_type, id, date, user, pc, ...) but the number of trailing columns
    # differs by log_type (logon/device/file/email/http each have a
    # different schema in readme.txt), and even within "email" rows there is
    # one more trailing column than in email.csv itself (an extra topic/tag
    # field ahead of the free-text content). A fixed-width pd.read_csv() chokes
    # on this, so we parse with csv.reader() and only rely on the first four
    # positional fields (log_type, id, date, user), which are consistent
    # across every log type.
    records = []
    for _, scen in scenarios.iterrows():
        detail_path = config.ANSWERS_DIR / scen["details"]
        if not detail_path.exists():
            print(f"[warn] detail file missing, skipping: {detail_path}")
            continue

        n_total, n_email = 0, 0
        with open(detail_path, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                n_total += 1
                log_type = row[0]
                if log_type != "email":
                    continue
                n_email += 1
                records.append({
                    "id": row[1],
                    "scenario": scen["scenario"],
                    "insider_user": scen["user"],
                    "email_row_user": row[3] if len(row) > 3 else "",
                    "date": row[2] if len(row) > 2 else "",
                })

        print(f"[info] scenario {scen['scenario']} ({scen['details']}): "
              f"{n_total} total logged rows, {n_email} are email rows")

    if not records:
        print("[warn] no malicious email rows found across any r6.2 scenario "
              "-- double check answers/ was extracted correctly.")
        return pd.DataFrame(columns=["id", "scenario", "insider_user", "email_row_user", "date"])

    out = pd.DataFrame(records)
    n_scenarios_with_email = out["scenario"].nunique()
    print(f"[info] {len(out)} malicious email ids collected across "
          f"{n_scenarios_with_email} scenario(s) (of {len(scenarios)} total r6.2 scenarios)")
    return out


def match_against_email_csv(malicious_ids: pd.DataFrame, limit_chunks: int | None = None) -> dict:
    """
    Stream email.csv and check which malicious ids actually appear in it.
    Returns a report dict; also mutates nothing (matching is a pure sanity
    check here -- the malicious_email_ids.csv file itself is the artefact
    downstream code should join against).
    """
    wanted_ids = set(malicious_ids["id"].tolist())
    found_ids = set()
    total_rows = 0

    for i, chunk in enumerate(open_email_csv_chunks()):
        if i == 0:
            validate_schema(chunk)
        total_rows += len(chunk)
        found_ids.update(set(chunk["id"]) & wanted_ids)

        print(f"[progress] chunk {i + 1}: {total_rows:,} rows scanned, "
              f"{len(found_ids)}/{len(wanted_ids)} malicious ids matched so far")

        if len(found_ids) == len(wanted_ids):
            print("[info] all malicious ids matched, stopping scan early")
            break
        if limit_chunks is not None and (i + 1) >= limit_chunks:
            print(f"[info] --limit-chunks={limit_chunks} reached, stopping early")
            break

    missing = wanted_ids - found_ids
    report = {
        "total_rows_scanned": total_rows,
        "n_malicious_ids_expected": len(wanted_ids),
        "n_malicious_ids_found_in_email_csv": len(found_ids),
        "n_missing": len(missing),
        "missing_ids": sorted(missing)[:20],  # cap in case limit_chunks truncated the scan
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit-chunks", type=int, default=None,
        help="Only scan this many chunks of email.csv when matching ids "
             "(for a quick smoke test; full ids will not necessarily be "
             "found). Omit to scan the whole file.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Step 1/3: loading r6.2 scenario index")
    print("=" * 70)
    scenarios = load_r62_scenarios()

    print()
    print("=" * 70)
    print("Step 2/3: extracting malicious email ids from scenario detail files")
    print("=" * 70)
    malicious_ids = extract_malicious_email_ids(scenarios)

    ids_out = config.OUTPUT_DIR / "malicious_email_ids.csv"
    malicious_ids.to_csv(ids_out, index=False)
    print(f"[info] wrote malicious email id lookup -> {ids_out}")

    if malicious_ids.empty:
        print("[warn] nothing to match against email.csv, exiting.")
        return

    print()
    print("=" * 70)
    print("Step 3/3: matching malicious ids against email.csv")
    print("=" * 70)
    report = match_against_email_csv(malicious_ids, limit_chunks=args.limit_chunks)

    report_out = config.OUTPUT_DIR / "ground_truth_match_report.json"
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 70)
    print("Match report")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"\n[info] full report written -> {report_out}")


if __name__ == "__main__":
    main()

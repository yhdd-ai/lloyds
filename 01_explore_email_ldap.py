"""
01_explore_email_ldap.py

M1 -- Data exploration for the CERT r6.2 email dataset and LDAP org data.

What this script does
----------------------
1. Loads and combines the 19 monthly LDAP/*.csv files (org structure + roles).
2. Streams email.csv in chunks (never loading the full ~8GB file at once) to:
   - validate the schema against readme.txt
   - count activity types (Send / View)
   - compute size and attachment-count distributions (running stats, so
     memory use stays flat regardless of file size)
   - compute internal (@dtaa.com) vs external recipient statistics
   - sample the `content` field and report evidence for whether it reads as
     full sentences or as a space-separated keyword list (this was flagged
     as an open question in the PRD -- the research proposal and readme.txt
     described it differently)
3. Writes all findings to python_code/outputs/eda_summary.json and a combined
   LDAP table to python_code/outputs/ldap_combined.csv, so later stages
   (feature engineering, ground-truth matching) don't need to re-scan the
   raw files.

Usage
-----
    python 01_explore_email_ldap.py                # full run over all of email.csv
    python 01_explore_email_ldap.py --limit-chunks 3  # quick smoke test

Run time note: a full pass over email.csv is I/O-bound and will take a while
(the file is ~8GB). Use --limit-chunks while developing, then run without it
for the real EDA numbers you'll quote in the report.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def open_email_csv_chunks(chunksize: int = config.CHUNK_SIZE):
    """
    Yield DataFrame chunks of email.csv, reading either:
      - directly from inside emails_lloyds.zip (streaming, no extraction), or
      - from an already-extracted copy at config.EMAIL_CSV_EXTRACTED,
    whichever is available. Falling back to the zip keeps disk usage low.
    """
    if config.EMAIL_CSV_EXTRACTED.exists():
        print(f"[info] reading extracted file: {config.EMAIL_CSV_EXTRACTED}")
        yield from pd.read_csv(
            config.EMAIL_CSV_EXTRACTED,
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
        )
        return

    if not config.EMAIL_ZIP.exists():
        raise FileNotFoundError(
            f"Neither {config.EMAIL_CSV_EXTRACTED} nor {config.EMAIL_ZIP} exist. "
            "Update config.py to point at your data."
        )

    print(f"[info] streaming {config.EMAIL_CSV_NAME_IN_ZIP} directly from "
          f"{config.EMAIL_ZIP.name} (no extraction to disk)")
    with zipfile.ZipFile(config.EMAIL_ZIP) as zf:
        with zf.open(config.EMAIL_CSV_NAME_IN_ZIP) as f:
            yield from pd.read_csv(
                f,
                dtype=str,
                keep_default_na=False,
                chunksize=chunksize,
            )


def load_ldap_combined() -> pd.DataFrame:
    """
    Read every LDAP/*.csv from inside emails_lloyds.zip and concatenate them
    into a single DataFrame, tagged with the snapshot month each row came
    from (LDAP is a monthly snapshot of the org chart, so the same user_id
    can appear multiple times with different role/team values over time).
    """
    frames = []
    with zipfile.ZipFile(config.EMAIL_ZIP) as zf:
        ldap_members = sorted(
            n for n in zf.namelist()
            if n.startswith(config.LDAP_PREFIX_IN_ZIP) and n.endswith(".csv")
        )
        if not ldap_members:
            raise RuntimeError("No LDAP/*.csv files found inside the zip.")
        for member in ldap_members:
            snapshot_month = Path(member).stem  # e.g. "2010-01"
            with zf.open(member) as f:
                df = pd.read_csv(f, dtype=str, keep_default_na=False)
            df["snapshot_month"] = snapshot_month
            frames.append(df)

    ldap = pd.concat(frames, ignore_index=True)
    print(f"[info] LDAP: {len(ldap_members)} monthly snapshots, "
          f"{len(ldap):,} total rows, "
          f"{ldap['user_id'].nunique():,} distinct user_id values")
    return ldap


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def validate_schema(chunk: pd.DataFrame) -> None:
    expected = config.EMAIL_COLUMNS
    actual = list(chunk.columns)
    if actual != expected:
        raise ValueError(
            f"email.csv schema mismatch.\nExpected: {expected}\nGot:      {actual}"
        )


# ---------------------------------------------------------------------------
# Running (streaming) statistics -- keeps memory flat regardless of file size
# ---------------------------------------------------------------------------
class RunningStats:
    """Incremental count/mean/std/min/max for a single numeric field."""

    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        values = values[~np.isnan(values)]
        if values.size == 0:
            return
        self.n += values.size
        self.sum += values.sum()
        self.sum_sq += np.square(values).sum()
        self.min = min(self.min, values.min())
        self.max = max(self.max, values.max())

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        mean = self.sum / self.n
        var = max(self.sum_sq / self.n - mean ** 2, 0.0)
        return {
            "n": int(self.n),
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "min": float(self.min),
            "max": float(self.max),
        }


def count_attachments_vectorised(attachments_col: pd.Series) -> np.ndarray:
    """
    attachments field looks like:
      "C:\\path\\file1.doc(1119253);C:\\path\\file2.doc(155895)"
    or is empty when there are no attachments. Vectorised over the whole
    column (no per-row Python function calls): count semicolon-separated
    entries, treating an empty string as zero attachments. This matters at
    scale -- a plain `.apply()` or per-row loop over tens of millions of
    rows is the single slowest part of a full run over email.csv.
    """
    non_empty = attachments_col != ""
    counts = attachments_col.str.count(";").to_numpy(dtype=float) + 1
    return np.where(non_empty, counts, 0.0)


_DOMAIN_PATTERN = "@" + re.escape(config.INTERNAL_DOMAIN)


def recipient_counts_vectorised(chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised internal/external recipient counts summed across the to/cc/bcc
    columns. Each field is a semicolon-separated list of addresses (or
    empty); "internal" means the address ends in @dtaa.com. Implemented with
    pandas .str methods (count/contains), not a per-row Python loop, so it
    scales to the full ~8GB file without becoming the bottleneck.
    """
    n_int = np.zeros(len(chunk))
    n_ext = np.zeros(len(chunk))
    for col in ["to", "cc", "bcc"]:
        s = chunk[col]
        non_empty = (s != "").to_numpy()
        total = np.where(non_empty, s.str.count(";").to_numpy(dtype=float) + 1, 0.0)
        internal = s.str.count(_DOMAIN_PATTERN, flags=re.IGNORECASE).to_numpy(dtype=float)
        internal = np.where(non_empty, internal, 0.0)
        internal = np.minimum(internal, total)  # guard against pathological edge cases
        n_int += internal
        n_ext += total - internal
    return n_int, n_ext


# ---------------------------------------------------------------------------
# content field format check
# ---------------------------------------------------------------------------
_SENTENCE_PUNCT = re.compile(r"[.!?]")
_WORD_RE = re.compile(r"[A-Za-z']+")


def content_style_signal(sample_texts: list[str]) -> dict:
    """
    Heuristic check for whether `content` reads as full sentences (has
    sentence-ending punctuation, capitalised sentence starts, function
    words like "the"/"is"/"and") or as a bare space-separated keyword list
    (no punctuation, mostly nouns/topic words, no repeated function words).
    """
    n = len(sample_texts)
    if n == 0:
        return {"n_sampled": 0}

    has_sentence_punct = sum(1 for t in sample_texts if _SENTENCE_PUNCT.search(t))
    common_stopwords = {"the", "is", "and", "of", "to", "in", "a", "was", "were"}
    stopword_hits = 0
    word_counts = []
    for t in sample_texts:
        words = [w.lower() for w in _WORD_RE.findall(t)]
        word_counts.append(len(words))
        if any(w in common_stopwords for w in words):
            stopword_hits += 1

    return {
        "n_sampled": n,
        "pct_with_sentence_punctuation": round(100 * has_sentence_punct / n, 1),
        "pct_containing_common_stopwords": round(100 * stopword_hits / n, 1),
        "avg_word_count": round(float(np.mean(word_counts)), 1),
        "example_snippets": [t[:160] for t in sample_texts[:5]],
    }


# ---------------------------------------------------------------------------
# Main EDA pass over email.csv
# ---------------------------------------------------------------------------
def run_email_eda(limit_chunks: int | None = None) -> dict:
    activity_counts = Counter()
    size_stats = RunningStats()
    attachment_count_stats = RunningStats()
    pct_with_attachment_num = 0
    pct_with_attachment_den = 0
    recipient_internal_stats = RunningStats()
    recipient_external_stats = RunningStats()
    any_external_recipient = 0
    users = set()
    total_rows = 0
    content_sample: list[str] = []
    rng = np.random.default_rng(42)

    for i, chunk in enumerate(open_email_csv_chunks()):
        if i == 0:
            validate_schema(chunk)

        total_rows += len(chunk)
        activity_counts.update(chunk["activity"].value_counts().to_dict())
        users.update(chunk["user"].unique().tolist())

        size_stats.update(pd.to_numeric(chunk["size"], errors="coerce").to_numpy())

        att_counts = count_attachments_vectorised(chunk["attachments"])
        attachment_count_stats.update(att_counts)
        pct_with_attachment_num += int((att_counts > 0).sum())
        pct_with_attachment_den += len(chunk)

        n_int, n_ext = recipient_counts_vectorised(chunk)
        recipient_internal_stats.update(n_int)
        recipient_external_stats.update(n_ext)
        any_external_recipient += int((n_ext > 0).sum())

        if len(content_sample) < 300:
            take = min(300 - len(content_sample), len(chunk))
            idx = rng.choice(len(chunk), size=take, replace=False) if len(chunk) > take else np.arange(len(chunk))
            content_sample.extend(chunk["content"].iloc[idx].tolist())

        print(f"[progress] processed chunk {i + 1} "
              f"({total_rows:,} rows so far, {len(users):,} distinct users)")

        if limit_chunks is not None and (i + 1) >= limit_chunks:
            print(f"[info] --limit-chunks={limit_chunks} reached, stopping early")
            break

    summary = {
        "total_rows_scanned": total_rows,
        "distinct_users": len(users),
        "activity_counts": dict(activity_counts),
        "size_bytes": size_stats.summary(),
        "attachments_per_email": attachment_count_stats.summary(),
        "pct_emails_with_attachment": round(
            100 * pct_with_attachment_num / max(pct_with_attachment_den, 1), 2
        ),
        "recipients_internal_per_email": recipient_internal_stats.summary(),
        "recipients_external_per_email": recipient_external_stats.summary(),
        "pct_emails_with_external_recipient": round(
            100 * any_external_recipient / max(total_rows, 1), 2
        ),
        "content_field_style_check": content_style_signal(content_sample),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit-chunks", type=int, default=None,
        help="Only process this many chunks (for a quick smoke test). "
             "Omit to run the full EDA over all of email.csv.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Step 1/2: loading and combining LDAP monthly snapshots")
    print("=" * 70)
    ldap = load_ldap_combined()
    ldap_out = config.OUTPUT_DIR / "ldap_combined.csv"
    ldap.to_csv(ldap_out, index=False)
    print(f"[info] wrote combined LDAP table -> {ldap_out}")

    print()
    print("=" * 70)
    print("Step 2/2: streaming EDA over email.csv")
    print("=" * 70)
    summary = run_email_eda(limit_chunks=args.limit_chunks)

    summary_out = config.OUTPUT_DIR / "eda_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("EDA summary")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\n[info] full summary written -> {summary_out}")


if __name__ == "__main__":
    main()

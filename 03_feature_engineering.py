"""
03_feature_engineering.py

M2 -- Feature engineering (PRD section 6, Phase 1) for the CERT r6.2 email
dataset: turns raw email.csv rows into a per-email feature table combining
metadata features and LLM-embedding-derived semantic features, ready to feed
either the LinUCB agent (Phase 2) or the anomaly-detection baseline (Phase 2b).

Design: single streaming pass, causal throughout
--------------------------------------------------
email.csv is processed in chunks, in file order. M1's EDA confirmed the file
is already ordered chronologically, so a single left-to-right pass over the
chunks is also a single pass forward in time -- which is exactly what two of
the features below need to stay leak-free:

1. Per-user semantic deviation score. For every email we compute how far its
   embedding is (cosine distance) from that user's *running mean embedding of
   all their previous emails only*. The running mean is then updated with the
   current email. A user's very first observed email has no prior baseline
   (cold start) and gets a deviation score of NaN, which downstream modelling
   should either impute or treat as "unknown, insufficient history" rather
   than silently coercing to 0.

2. The embedding dimensionality reduction (384 -> config.PCA_COMPONENTS) uses
   scikit-learn's IncrementalPCA. Each chunk is first transformed with PCA
   components fitted on *previous chunks only*, then .partial_fit() updates
   those components for future chunks. The first chunk has no PCA basis and
   therefore receives missing PCA coordinates; downstream modelling treats it
   as an explicitly documented warm-up period. This preserves the strict
   chronological information boundary required for the online-learning
   evaluation.

Metadata features computed per email
-------------------------------------
- size (bytes), attachment_count
- n_recipients_internal, n_recipients_external, has_external_recipient
- hour_of_day, is_after_hours, is_weekend  (readme.txt: after-hours activity
  flagged as a significant behavioural signal)

Ground truth
------------
Joined from outputs/malicious_email_ids.csv (produced by 02_match_ground_truth.py)
on the `id` column. Run 02_ first.

Usage
-----
    python 03_feature_engineering.py                    # full run
    python 03_feature_engineering.py --limit-chunks 3    # quick smoke test
    python 03_feature_engineering.py --benchmark-chunks 3  # real-embedding ETA; preserves production output
    python 03_feature_engineering.py --limit-chunks 3 --mock-embeddings
        # for testing the pipeline (running baseline math, PCA, label join,
        # I/O) on a machine without sentence-transformers/torch installed.
        # NEVER use --mock-embeddings for real feature generation -- the
        # mock embeddings carry no semantic meaning at all.
"""
from __future__ import annotations

import argparse
import re
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

import config

_explore = import_module("01_explore_email_ldap")
open_email_csv_chunks = _explore.open_email_csv_chunks
validate_schema = _explore.validate_schema
count_attachments_vectorised = _explore.count_attachments_vectorised
recipient_counts_vectorised = _explore.recipient_counts_vectorised


# ---------------------------------------------------------------------------
# Embedding backend (pluggable so the pipeline can be tested without torch)
# ---------------------------------------------------------------------------
class RealEmbedder:
    """Wraps sentence-transformers. This is the production path."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME,
            local_files_only=config.EMBEDDING_LOCAL_FILES_ONLY,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,  # tqdm bar per chunk -- without this,
                # a 20k-200k text chunk produces zero output for minutes and
                # looks identical to a hung process.
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


class MockEmbedder:
    """
    Deterministic, dependency-free stand-in for RealEmbedder, used only to
    smoke-test the surrounding pipeline (running baseline, PCA, label join,
    chunked I/O) on machines without sentence-transformers/torch installed.
    Produces a fixed-length vector from a hash of the text -- it carries NO
    semantic meaning and must never be used to generate features for
    modelling or the report.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), config.EMBEDDING_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            out[i] = rng.normal(size=config.EMBEDDING_DIM)
        return out


def get_embedder(mock: bool):
    if mock:
        print("[warn] using MockEmbedder -- features will NOT be semantically "
              "meaningful. This is for pipeline testing only.")
        return MockEmbedder()
    print(f"[info] loading sentence-transformers model "
          f"'{config.EMBEDDING_MODEL_NAME}' ...")
    return RealEmbedder()


# ---------------------------------------------------------------------------
# Per-user running baseline (causal: deviation computed before the update)
# ---------------------------------------------------------------------------
class UserBaselineTracker:
    """
    Maintains, per user, a running mean embedding over all of that user's
    previously seen emails. For each new email, `deviation_and_update`
    returns the cosine distance from the email's embedding to the user's
    prior mean (NaN if this is the user's first observed email), THEN folds
    the new embedding into the running mean.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self._sum: dict[str, np.ndarray] = {}
        self._count: dict[str, int] = {}

    def deviation_and_update(self, user: str, embedding: np.ndarray) -> float:
        n = self._count.get(user, 0)
        if n == 0:
            deviation = np.nan
        else:
            baseline = self._sum[user] / n
            deviation = _cosine_distance(embedding, baseline)

        self._sum[user] = self._sum.get(user, np.zeros(self.dim, dtype=np.float64)) + embedding
        self._count[user] = n + 1
        return deviation

    def n_users_seen(self) -> int:
        return len(self._count)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Metadata features
# ---------------------------------------------------------------------------
_DATE_FMT_CANDIDATES = ("%m/%d/%Y %H:%M:%S",)


def compute_metadata_features(chunk: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")

    attachment_count = count_attachments_vectorised(chunk["attachments"])
    n_int, n_ext = recipient_counts_vectorised(chunk)
    hour = dt.dt.hour
    is_after_hours = (hour < config.WORKDAY_START_HOUR) | (hour >= config.WORKDAY_END_HOUR)
    is_weekend = dt.dt.dayofweek >= 5  # 5=Saturday, 6=Sunday

    return pd.DataFrame({
        "size": pd.to_numeric(chunk["size"], errors="coerce"),
        "attachment_count": attachment_count,
        "n_recipients_internal": n_int,
        "n_recipients_external": n_ext,
        "has_external_recipient": (n_ext > 0).astype(int),
        "hour_of_day": hour,
        "is_after_hours": is_after_hours.astype(int),
        "is_weekend": is_weekend.astype(int),
    })


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_malicious_ids() -> pd.DataFrame:
    if not config.GROUND_TRUTH_LOOKUP.exists():
        raise FileNotFoundError(
            f"{config.GROUND_TRUTH_LOOKUP} not found -- run "
            "02_match_ground_truth.py first."
        )
    return pd.read_csv(config.GROUND_TRUTH_LOOKUP, dtype=str)


# ---------------------------------------------------------------------------
# Main streaming pass
# ---------------------------------------------------------------------------
def run_feature_engineering(
    mock_embeddings: bool,
    limit_chunks: int | None,
    output_path=None,
    replace_output_on_success: bool = True,
) -> dict:
    """Run feature engineering, writing to ``output_path`` when supplied."""
    output_path = Path(output_path or config.FEATURES_OUTPUT)
    # Keep an existing completed feature table intact until a replacement has
    # been generated successfully.  This matters because the full real-embedder
    # run takes many hours and may be interrupted.
    working_output_path = output_path.with_name(
        f"{output_path.stem}.in_progress{output_path.suffix}"
    )
    embedder = get_embedder(mock_embeddings)
    baseline_tracker = UserBaselineTracker()
    pca = IncrementalPCA(n_components=config.PCA_COMPONENTS)

    malicious = load_malicious_ids()
    malicious_ids = set(malicious["id"].tolist()) if not malicious.empty else set()
    print(f"[info] {len(malicious_ids)} malicious email ids loaded for labelling")

    pca_dim_cols = [f"embed_pc_{i}" for i in range(config.PCA_COMPONENTS)]
    total_rows = 0
    total_malicious_written = 0

    # Start a fresh staging file.  The completed output is replaced only after
    # the full streaming pass succeeds; partial resume is not implemented.
    if working_output_path.exists():
        working_output_path.unlink()
    header_written = False

    run_start = time.time()
    for i, chunk in enumerate(open_email_csv_chunks(chunksize=config.FEATURE_CHUNK_SIZE)):
        chunk_start = time.time()
        if i == 0:
            validate_schema(chunk)

        meta = compute_metadata_features(chunk)

        print(f"[info] chunk {i + 1}: embedding {len(chunk):,} emails "
              f"(this is the slow step -- a progress bar should appear below "
              f"for the real embedder; if nothing prints here for a long "
              f"time with --mock-embeddings, something else is wrong)")
        embeddings = embedder.encode(chunk["content"].tolist())

        # PCA: transform with components fitted on prior chunks only, then
        # update the components using this chunk for future chunks.  Updating
        # first would allow each row to influence its own coordinates and the
        # coordinates of later rows in the chunk.
        if hasattr(pca, "components_"):
            reduced = pca.transform(embeddings)
        else:
            # The warm-up chunk has no historical PCA basis.
            reduced = np.full((len(chunk), config.PCA_COMPONENTS), np.nan)
        # IncrementalPCA requires at least n_components samples per update.
        if len(chunk) >= config.PCA_COMPONENTS:
            pca.partial_fit(embeddings)

        deviations = np.empty(len(chunk), dtype=float)
        users = chunk["user"].tolist()
        for j in range(len(chunk)):
            deviations[j] = baseline_tracker.deviation_and_update(users[j], embeddings[j])

        is_malicious = chunk["id"].isin(malicious_ids).astype(int)
        total_malicious_written += int(is_malicious.sum())

        out = pd.DataFrame({
            "id": chunk["id"].values,
            "date": chunk["date"].values,
            "user": chunk["user"].values,
        })
        out = pd.concat([out.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
        out["semantic_deviation"] = deviations
        out = pd.concat(
            [out, pd.DataFrame(reduced, columns=pca_dim_cols)], axis=1
        )
        out["is_malicious"] = is_malicious.values

        out.to_csv(
            working_output_path,
            mode="a",
            header=not header_written,
            index=False,
        )
        header_written = True

        total_rows += len(chunk)
        chunk_elapsed = time.time() - chunk_start
        total_elapsed = time.time() - run_start
        rows_per_sec = total_rows / total_elapsed if total_elapsed > 0 else 0
        remaining_rows = max(config.EXPECTED_TOTAL_ROWS - total_rows, 0)
        eta_minutes = (remaining_rows / rows_per_sec / 60) if rows_per_sec > 0 else float("nan")
        print(f"[progress] chunk {i + 1}: {total_rows:,} rows written "
              f"(chunk took {chunk_elapsed:.1f}s, {rows_per_sec:,.0f} rows/sec overall, "
              f"~{eta_minutes:.0f} min remaining to reach "
              f"~{config.EXPECTED_TOTAL_ROWS:,} total rows), "
              f"{baseline_tracker.n_users_seen():,} users with an established "
              f"baseline so far, {total_malicious_written} malicious rows labelled")

        if limit_chunks is not None and (i + 1) >= limit_chunks:
            print(f"[info] --limit-chunks={limit_chunks} reached, stopping early")
            break

    # A deliberately limited run is normally a smoke test, not a replacement
    # for a completed feature table.  Benchmark mode explicitly opts in to a
    # temporary replacement, which main() deletes immediately afterwards.
    if replace_output_on_success:
        working_output_path.replace(output_path)
    else:
        output_path = working_output_path

    total_elapsed = time.time() - run_start
    rows_per_sec = total_rows / total_elapsed if total_elapsed > 0 else 0
    return {
        "total_rows_written": total_rows,
        "n_users_seen": baseline_tracker.n_users_seen(),
        "n_malicious_rows_labelled": total_malicious_written,
        "elapsed_seconds": round(total_elapsed, 1),
        "rows_per_second": round(rows_per_sec, 1),
        "estimated_full_run_minutes": round(
            config.EXPECTED_TOTAL_ROWS / rows_per_sec / 60, 1
        ) if rows_per_sec else None,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-chunks", type=int, default=None,
                         help="Only process this many chunks (quick smoke test).")
    parser.add_argument("--mock-embeddings", action="store_true",
                         help="Use a fast dependency-free mock embedder for "
                              "pipeline testing. NEVER use this for real "
                              "feature generation.")
    parser.add_argument(
        "--benchmark-chunks", type=int, default=None,
        help="Run this many real-embedding chunks to estimate full-run time. "
             "Writes only a temporary benchmark file and preserves the "
             "production feature CSV.",
    )
    args = parser.parse_args()
    if args.benchmark_chunks is not None and args.benchmark_chunks < 1:
        parser.error("--benchmark-chunks must be at least 1")
    if args.benchmark_chunks is not None and args.limit_chunks is not None:
        parser.error("use either --benchmark-chunks or --limit-chunks, not both")
    if args.benchmark_chunks is not None and args.mock_embeddings:
        parser.error("a runtime benchmark must use real embeddings")

    benchmark_path = None
    if args.benchmark_chunks is not None:
        benchmark_path = config.OUTPUT_DIR / "_benchmark_email_features.csv"
        print(f"[info] benchmark mode: temporary output is {benchmark_path}")
    try:
        result = run_feature_engineering(
            mock_embeddings=args.mock_embeddings,
            limit_chunks=args.benchmark_chunks or args.limit_chunks,
            output_path=benchmark_path,
            replace_output_on_success=(
                args.benchmark_chunks is not None or args.limit_chunks is None
            ),
        )
    finally:
        if benchmark_path is not None and benchmark_path.exists():
            benchmark_path.unlink()
            print("[info] benchmark temporary output removed; production feature CSV was untouched")
    print()
    print("=" * 70)
    print("Feature engineering summary")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

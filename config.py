"""
config.py

Central configuration for the M1 (data exploration) stage of the Lloyds
insider-threat dissertation project.

All paths assume the following layout inside the project's `lloyds` data
folder (adjust DATA_DIR below if your layout differs):

    lloyds/
        emails_lloyds.zip        <- contains email.csv (~8GB uncompressed)
                                     and LDAP/*.csv (19 monthly files)
        answers/                 <- already extracted from answers.tar.bz2
            insiders.csv
            r6.2-1.csv ... r6.2-5.csv
        python_code/
            config.py            <- this file
            01_explore_email_ldap.py
            02_match_ground_truth.py
            outputs/              <- created automatically, holds EDA artefacts

Design notes
------------
email.csv and the LDAP files are NOT extracted to disk. At ~8GB, extracting
email.csv would double the disk footprint for no benefit: Python's `zipfile`
module can open a single member of the zip as a streaming file-like object,
which `pandas.read_csv(..., chunksize=...)` can then read incrementally
without ever materialising the full file in memory or on disk. If you prefer
to work with an extracted copy (e.g. for faster repeated access), extract it
once with `unzip emails_lloyds.zip email.csv -d <dest>` and point
EMAIL_CSV_PATH at the extracted file instead -- both 01_ and 02_ scripts
accept either mode (see `open_email_csv_chunks()` in 01_explore_email_ldap.py).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths -- adjust DATA_DIR if this repo is checked out somewhere else
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent          # .../lloyds/python_code
DATA_DIR = PROJECT_ROOT.parent                          # .../lloyds
REPOSITORY_ANSWERS_DIR = PROJECT_ROOT / "data" / "answers"

EMAIL_ZIP = DATA_DIR / "emails_lloyds.zip"
EMAIL_CSV_NAME_IN_ZIP = "email.csv"
LDAP_PREFIX_IN_ZIP = "LDAP/"

# If you extract email.csv to disk yourself, set this and 01_/02_ will use it
# automatically instead of streaming from the zip.
EMAIL_CSV_EXTRACTED = DATA_DIR / "email.csv"

# Prefer the compact answer files committed inside a standalone Git checkout.
# Retain the original parent-directory layout for the existing local project.
ANSWERS_DIR = REPOSITORY_ANSWERS_DIR if REPOSITORY_ANSWERS_DIR.exists() else DATA_DIR / "answers"
INSIDERS_INDEX = ANSWERS_DIR / "insiders.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset constants (from readme.txt inside emails_lloyds.zip)
# ---------------------------------------------------------------------------
RELEASE_TAG = "6.2"  # CERT r6.2, as used in answers/insiders.csv "dataset" col

# email.csv schema
EMAIL_COLUMNS = [
    "id", "date", "user", "pc", "to", "cc", "bcc", "from",
    "activity", "size", "attachments", "content",
]

# answers/r6.2-N.csv rows are raw log rows with one extra leading column
# (the source log type: logon / device / file / email / http) and NO header.
# We only need the "email" subset, whose remaining columns match EMAIL_COLUMNS.
ANSWER_COLUMNS = ["log_type"] + EMAIL_COLUMNS

# Company email domain used to distinguish internal vs external recipients
INTERNAL_DOMAIN = "dtaa.com"

# Rows to read per chunk when streaming email.csv (~8GB, ~10-11M rows expected
# based on the M1 EDA -- see python_code/README.md)
CHUNK_SIZE = 200_000

# ---------------------------------------------------------------------------
# M2 -- feature engineering constants
# ---------------------------------------------------------------------------
GROUND_TRUTH_LOOKUP = OUTPUT_DIR / "malicious_email_ids.csv"  # produced by 02_
FEATURES_OUTPUT = OUTPUT_DIR / "email_features.csv"           # produced by 03_
PCA_STATE_OUTPUT = OUTPUT_DIR / "incremental_pca_state.joblib"

# Sentence embedding model (see UoE_26_Intro_Notebook.ipynb -- same model used
# there). 384-dim, CPU-friendly, good default for a first pass.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# The model is already cached locally. Keeping this on avoids an unnecessary
# Hugging Face metadata request on every run, which can fail on restricted or
# unstable networks. Set to False only when deliberately downloading/updating.
EMBEDDING_LOCAL_FILES_ONLY = True
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 256

# email.csv chunk size used specifically by 03_feature_engineering.py. Much
# smaller than CHUNK_SIZE (used by 01_/02_, where per-row work is cheap and
# vectorised): running a transformer over 200k texts before printing a single
# line of progress makes the script look hung for minutes. A smaller chunk
# here means progress prints (and the on-disk output file) update far more
# often, at a small, worthwhile overhead cost.
FEATURE_CHUNK_SIZE = 20_000

# Approximate total row count of email.csv, from the full M1 EDA run
# (outputs/eda_summary.json: total_rows_scanned). Used only to print a rough
# ETA while 03_feature_engineering.py is running -- not used for any
# correctness-critical logic.
EXPECTED_TOTAL_ROWS = 10_994_957

# Dimensionality the raw 384-dim embedding is reduced to for storage / as the
# LinUCB context vector (PRD section 6, Phase 1: "dimensionality reduction
# may be applied to improve efficiency"). Fitted online with IncrementalPCA
# as chunks stream past, so it never needs the full embedding matrix in memory.
PCA_COMPONENTS = 32

# "After hours" window used for the timing feature -- outside of this range
# counts as after-hours, matching the readme.txt emphasis on after-hours
# activity as a significant behavioural signal.
WORKDAY_START_HOUR = 7
WORKDAY_END_HOUR = 18

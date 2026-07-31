"""Filesystem layout for the fine-tuning data workstream.

Every path is derived from the repository root so the pipeline runs unchanged on
node0, node1, or a judge's clone. Override the corpus location with
``WESTPAC_DATA_ROOT`` if the datasets are mounted elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# training/ftdata/paths.py -> training/ftdata -> training -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("WESTPAC_DATA_ROOT", REPO_ROOT / "data set"))
RBA_CSV = DATA_ROOT / "RBA Rates" / "RBA-rates.csv"
RBA_JSONL = DATA_ROOT / "RBA Rates" / "RBA-rates.jsonl"
ASX_DIR = DATA_ROOT / "ASX"
AFR_DIR = DATA_ROOT / "AFR"

TRAINING_ROOT = REPO_ROOT / "training"
SCHEMA_DIR = TRAINING_ROOT / "schemas"
OUT_DATA = TRAINING_ROOT / "data"
CACHE_DIR = OUT_DATA / "cache"
PREPARED_DIR = OUT_DATA / "prepared"
EVAL_DIR = TRAINING_ROOT / "eval"
METRICS_DIR = TRAINING_ROOT / "metrics"

PUBLIC_QUESTIONS = REPO_ROOT / "Participant_Package" / "public_questions.jsonl"

# The AFR one-pass scan is the only expensive step (~2 min, 780 MB). Its result is
# cached here; delete the file to force a rescan.
AFR_INDEX_CACHE = CACHE_DIR / "afr_index.json"


def ensure_dirs() -> None:
    for d in (OUT_DATA, CACHE_DIR, PREPARED_DIR, EVAL_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

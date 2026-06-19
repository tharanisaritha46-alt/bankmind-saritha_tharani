"""Shared data loading and preprocessing for training and serving.

Keeping this logic in one place guarantees the model is trained on exactly the
same encoding that the API applies to incoming requests.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bank-full.csv"
MODEL_PATH = ROOT / "model" / "model.pkl"

TARGET = "y"

# Columns that hold text and therefore need encoding before they reach a model.
CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
NUMERIC_COLS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def load_raw() -> pd.DataFrame:
    """Read the semicolon-delimited UCI Bank Marketing file."""
    return pd.read_csv(DATA_PATH, sep=";")


def build_category_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Map each category string to a stable integer code (sorted for determinism)."""
    maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        categories = sorted(df[col].astype(str).unique())
        maps[col] = {cat: code for code, cat in enumerate(categories)}
    return maps


def build_defaults(df: pd.DataFrame) -> dict[str, object]:
    """Per-feature defaults used when the API receives a partial customer record.

    Numeric -> median, categorical -> most frequent value.
    """
    defaults: dict[str, object] = {}
    for col in NUMERIC_COLS:
        defaults[col] = float(df[col].median())
    for col in CATEGORICAL_COLS:
        defaults[col] = str(df[col].mode().iloc[0])
    return defaults


def encode_frame(df: pd.DataFrame, category_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Encode categoricals to their integer codes; unseen values fall back to code 0."""
    out = df.copy()
    for col, mapping in category_maps.items():
        out[col] = out[col].astype(str).map(mapping).fillna(0).astype(int)
    return out[FEATURE_COLS]


def row_from_payload(payload: dict[str, object], defaults: dict[str, object]) -> pd.DataFrame:
    """Build a single-row feature frame from a (possibly partial) request payload."""
    row = {col: payload.get(col, defaults[col]) for col in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)

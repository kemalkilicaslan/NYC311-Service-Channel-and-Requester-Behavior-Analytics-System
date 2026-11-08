"""Shared helpers: database connection, paths, palette, feature preparation, model registry.

Training data is read ORDER BY unique_key: LightGBM bagging and the logistic
regression solver depend on row order, and PostgreSQL returns unordered rows
in whatever order the scan produces.

Training (01_channel_model.py, 04_retrain.py) and scoring (03_batch_scoring.py)
call the same `prepare_features`, so the features seen in production match the
features seen in training. The password is read from ~/.pgpass by libpq.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"
REGISTRY = MODEL_DIR / "registry.json"
for d in (TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Figures go to the project root with the project name as prefix, like the README
# images of the other repositories (same prefix as R/00_setup.R)
FIG_PREFIX = "NYC311-Service-Channel-and-Requester-Behavior-Analytics-"


def fig_path(name: str) -> Path:
    return ROOT / f"{FIG_PREFIX}{name}.png"

# Same palette file as R/00_setup.R
PAL = json.loads((ROOT / "config" / "palette.json").read_text())

CATEGORICAL = ["agency", "complaint_type", "complaint_family", "location_type", "borough", "prev_channel"]
NUMERIC = ["hr", "dow", "is_weekend", "mon", "log_prior_n", "days_since_prev", "prior_digital_rate"]
HISTORY = ["prev_channel", "log_prior_n", "days_since_prev", "prior_digital_rate"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "is_digital"

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  cat_smooth=20, verbose=-1, seed=42, num_threads=8,
                  # same data in the same order -> same model on every run
                  deterministic=True, force_row_wise=True)

# Rare categories fold into OTHER: caps cardinality and absorbs unseen values at scoring time
TOP_K = {"complaint_type": 150, "location_type": 40, "agency": 20}

ML_COLUMNS = """unique_key, created_at, is_digital, channel, agency, complaint_type, complaint_family,
                location_type, borough, hr, dow, is_weekend, yr, mon,
                prior_n, prev_channel, days_since_prev, prior_digital_rate"""


def engine():
    user = os.getenv("PGUSER", "postgres")
    host = os.getenv("PGHOST", "localhost")
    db = os.getenv("PGDATABASE", "nyc311")
    return create_engine(f"postgresql+psycopg2://{user}@{host}:5432/{db}")


def style_axes(ax):
    """Apply the shared chart style (surface, grid, text colors) to a matplotlib axis."""
    ax.set_facecolor(PAL["surface"])
    ax.grid(color=PAL["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PAL["grid"])
    ax.tick_params(colors=PAL["text_secondary"])
    ax.title.set_color(PAL["text_primary"])
    ax.xaxis.label.set_color(PAL["text_primary"])
    ax.yaxis.label.set_color(PAL["text_primary"])


def prepare_features(df: pd.DataFrame, categories: dict | None = None):
    """Turn rows of mart.ml_requests / mart.request_history into model input.

    With categories=None (training) the category lists are learned and returned.
    With categories given (scoring) the same lists are applied and unseen values become OTHER.
    """
    X = pd.DataFrame(index=df.index)
    X["agency"] = df["agency"]
    X["complaint_type"] = df["complaint_type"]
    X["complaint_family"] = df["complaint_family"]
    # Spelling variants such as 'RESIDENTIAL BUILDING' / 'Residential Building' (02_data_quality.sql)
    X["location_type"] = df["location_type"].fillna("missing").str.lower().str.replace(r"[^a-z]", "", regex=True)
    X["borough"] = df["borough"].fillna("missing")
    X["prev_channel"] = df["prev_channel"].fillna("NONE")
    X["hr"] = df["hr"].astype("int16")
    X["dow"] = df["dow"].astype("int16")
    X["is_weekend"] = df["is_weekend"].astype("int8")
    X["mon"] = df["mon"].astype("int16")
    X["log_prior_n"] = np.log1p(df["prior_n"].astype("float64"))
    X["days_since_prev"] = df["days_since_prev"].astype("float64")
    X["prior_digital_rate"] = df["prior_digital_rate"].astype("float64")

    learn = categories is None
    if learn:
        categories = {}
    for col in CATEGORICAL:
        if learn:
            vc = X[col].value_counts()
            keep = vc.index[: TOP_K[col]] if col in TOP_K else vc.index
            categories[col] = sorted(set(keep) | {"OTHER"})
        X[col] = pd.Categorical(X[col].where(X[col].isin(categories[col]), "OTHER"), categories=categories[col])
    return X[FEATURES], categories


def predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """P(digital) from a model bundle, with its Platt calibrator applied when present."""
    p = bundle["model"].predict(X[bundle["features"]], num_iteration=bundle["best_iteration"])
    if bundle.get("calibrator") is not None:
        p = bundle["calibrator"].predict_proba(logit(p).reshape(-1, 1))[:, 1]
    return p


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---- model registry: one file per version, registry.json names the active one
def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"active": None, "models": {}}


def register(bundle: dict, card: dict, activate: bool) -> None:
    joblib.dump(bundle, MODEL_DIR / f"{bundle['version']}.joblib")
    reg = load_registry()
    reg["models"][bundle["version"]] = card
    if activate or reg["active"] is None:
        reg["active"] = bundle["version"]
    REGISTRY.write_text(json.dumps(reg, indent=2, default=str))


def load_model(version: str) -> dict:
    return joblib.load(MODEL_DIR / f"{version}.joblib")


def load_active() -> dict:
    return load_model(load_registry()["active"])

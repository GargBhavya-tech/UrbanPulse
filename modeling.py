"""Modeling core for UrbanPulse B3 (Bible §5 Notebook 03).

Provides the leak-safe plumbing shared by the horizon sweep and full training:
- feature-set selection (all features vs. leak-free)
- temporal target shifting for forecasting, with cross-split leakage guarded
- the Bible's temporal train/val/test split (days 1-10 / 11-12 / 13-14)
- target-encoding of LINK_ID fit on the training split only
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config

# Columns that are identifiers, not model inputs.
NON_FEATURE_COLS: list[str] = ["date", "day_number", "congested"]

# Columns that DEFINE the target (mean_occup>0.5 & mean_queue>238) and their
# direct derivatives — leakage if used to predict the *current* interval.
LEAKY_COLS: list[str] = [
    "mean_occup",
    "max_occup",
    "mean_queue_s",
    "max_queue_s",
    "congestion_index",
    "road_health_score",
]

# Train/val/test day ranges (Bible §5 Notebook 03 — temporal split only).
TRAIN_DAYS: range = range(1, 11)   # 1-10
VAL_DAYS: range = range(11, 13)    # 11-12
TEST_DAYS: range = range(13, 15)   # 13-14


def feature_columns(df: pd.DataFrame, leak_free: bool) -> list[str]:
    """Return the model-input columns.

    Args:
        df: Feature frame.
        leak_free: If True, exclude the target-defining columns (for nowcasting).

    Returns:
        Ordered list of feature column names (LINK_ID kept; it is target-encoded).
    """
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if leak_free:
        cols = [c for c in cols if c not in LEAKY_COLS]
    return cols


def shift_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Shift the target ``horizon`` intervals into the future, per link.

    The label for feature-row t becomes congestion at t + horizon (each interval
    is 5 min). Rows whose future label is missing, or whose future label falls in
    a *different* temporal split than the feature row, are dropped — this is what
    prevents val/test outcomes from leaking into training labels.

    Args:
        df: Feature frame (must contain ``LINK_ID``, ``date``, ``day_number``,
            ``congested``).
        horizon: Number of 5-min intervals ahead. 0 returns the frame unchanged
            (nowcasting).

    Returns:
        Frame with ``congested`` replaced by the future label and tail/boundary
        rows removed.
    """
    if horizon == 0:
        return df.copy()

    out = df.sort_values(["LINK_ID", "date"]).copy()
    grp = out.groupby("LINK_ID", sort=False)
    future_target = grp["congested"].shift(-horizon)
    future_day = grp["day_number"].shift(-horizon)

    out["congested"] = future_target
    # Drop rows with no future label, or where the future label crosses a split
    # boundary (feature day and label day fall in different split segments).
    same_split = future_day.apply(_split_of) == out["day_number"].apply(_split_of)
    out = out[future_target.notna() & same_split].copy()
    out["congested"] = out["congested"].astype("int64")
    return out


def _split_of(day: float) -> str:
    """Map a day number to its split name ('train' / 'val' / 'test' / 'none')."""
    if day in TRAIN_DAYS:
        return "train"
    if day in VAL_DAYS:
        return "val"
    if day in TEST_DAYS:
        return "test"
    return "none"


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by day into train / val / test (Bible temporal split).

    Args:
        df: Feature frame with ``day_number``.

    Returns:
        ``(train, val, test)`` DataFrames.
    """
    train = df[df["day_number"].isin(TRAIN_DAYS)].copy()
    val = df[df["day_number"].isin(VAL_DAYS)].copy()
    test = df[df["day_number"].isin(TEST_DAYS)].copy()
    return train, val, test


def target_encode_link(
    train: pd.DataFrame,
    frames: list[pd.DataFrame],
    smoothing: float = 20.0,
) -> list[pd.Series]:
    """Target-encode ``LINK_ID`` using train-split congestion rate only.

    Smoothed toward the global train mean to stabilise low-support links.

    Args:
        train: Training frame (provides the encoding map).
        frames: Frames to transform (e.g. ``[train, val, test]``).
        smoothing: Smoothing strength (higher = closer to global mean).

    Returns:
        One encoded Series per input frame, in order.
    """
    global_mean = train["congested"].mean()
    stats = train.groupby("LINK_ID")["congested"].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
        stats["count"] + smoothing
    )
    return [f["LINK_ID"].map(smooth).fillna(global_mean) for f in frames]


def prepare_xy(
    df: pd.DataFrame, horizon: int, leak_free: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Full prep: shift target, split, target-encode LINK_ID.

    Args:
        df: Feature frame from B2.
        horizon: Forecast horizon in intervals (0 = nowcast).
        leak_free: Whether to drop target-defining features.

    Returns:
        ``(X_train, X_val, X_test, y)`` where ``y`` is a dict with keys
        ``train``/``val``/``test``.
    """
    shifted = shift_target(df, horizon)
    train, val, test = temporal_split(shifted)

    enc_train, enc_val, enc_test = target_encode_link(train, [train, val, test])
    feat_cols = feature_columns(df, leak_free)

    def _x(frame: pd.DataFrame, enc: pd.Series) -> pd.DataFrame:
        x = frame[feat_cols].copy()
        x["LINK_ID"] = enc.to_numpy()  # replace raw id with encoded value
        return x

    x_train = _x(train, enc_train)
    x_val = _x(val, enc_val)
    x_test = _x(test, enc_test)
    y = {
        "train": train["congested"],
        "val": val["congested"],
        "test": test["congested"],
    }
    return x_train, x_val, x_test, y


def evaluate(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Compute ROC-AUC and PR-AUC.

    Args:
        y_true: Ground-truth binary labels.
        proba: Predicted positive-class probabilities.

    Returns:
        Dict with ``roc_auc``, ``pr_auc`` and the positive rate.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "pos_rate": float(y_true.mean()),
    }

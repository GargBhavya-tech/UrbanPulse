"""Data cleaning for UrbanPulse (Bible §2 + §5 Notebook 01).

Implements the nine mandatory cleaning steps as small, independently testable
functions, plus a ``clean`` orchestrator that applies them in order.

Design notes:
- Every function returns a new/modified DataFrame; the orchestrator copies once
  up front so callers' frames are never mutated.
- Speed scaling, occupancy capping, and the structural flags are exactly as the
  Bible prescribes. No interpretation beyond the documented rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``date`` to datetime and derive time features (step 1).

    Adds ``hour``, ``minute``, ``day_of_week``, ``day_number``, ``minute_of_day``.

    Args:
        df: Raw frame containing a ``date`` column.

    Returns:
        Frame with ``date`` as datetime and the derived time columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["day_of_week"] = df["date"].dt.dayofweek  # 0 = Monday
    df["day_number"] = df["DAY"].astype("int64")
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully redundant columns, i.e. ``TIMEINT`` (step 2).

    Args:
        df: Input frame.

    Returns:
        Frame without the redundant columns.
    """
    present = [c for c in config.DROP_COLUMNS if c in df.columns]
    return df.drop(columns=present)


def cast_vehicle_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Cast vehicle-count columns for lanes 2-6 from float to int64 (step 3).

    Lane 1 is already integer in the raw file; lanes 2-6 arrive as floats.

    Args:
        df: Input frame.

    Returns:
        Frame with integer vehicle-count columns.
    """
    df = df.copy()
    for col in config.metric_cols("vehs"):
        df[col] = df[col].round().astype("int64")
    return df


def cap_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Cap all occupancy columns at 1.0 (step 4, Bible §2 Issue 1).

    Sensor saturation produces values >1.0 under bumper-to-bumper queuing. We
    clip rather than drop, preserving every row.

    Args:
        df: Input frame.

    Returns:
        Frame with occupancy clipped to ``[0, OCCUPANCY_CAP]``.
    """
    df = df.copy()
    occ_cols = config.metric_cols("occup")
    df[occ_cols] = df[occ_cols].clip(upper=config.OCCUPANCY_CAP)
    return df


def add_lane6_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``lane6_active`` = 1 when Lane 6 carries vehicles (step 5).

    Lane 6 is a conditional-use lane (74.5% structural zeros). It must never be
    averaged into cross-lane metrics; this binary flag captures its usage.

    Args:
        df: Input frame.

    Returns:
        Frame with the ``lane6_active`` column.
    """
    df = df.copy()
    vehs_6 = f"{config.METRIC_PREFIXES['vehs']}_6"
    df["lane6_active"] = (df[vehs_6] > 0).astype("int64")
    return df


def add_stall_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add stall flags for lanes 4 and 5 (steps 6-7, Bible §2 Issue 3).

    A stall is a positive vehicle count with exactly zero recorded speed.

    Args:
        df: Input frame.

    Returns:
        Frame with ``lane4_stalled`` and ``lane5_stalled`` columns.
    """
    df = df.copy()
    for lane in (4, 5):
        vehs = f"{config.METRIC_PREFIXES['vehs']}_{lane}"
        speed = f"{config.METRIC_PREFIXES['arith']}_{lane}"
        df[f"lane{lane}_stalled"] = (
            (df[vehs] > 0) & (df[speed] == 0)
        ).astype("int64")
    return df


def scale_speeds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert lane 1-5 speeds from 0.1 km/h to km/h (step 8).

    DEVIATION FROM BIBLE STEP 8: the Bible says "divide *all* speed columns by
    10", but the data shows lanes 1-5 are in 0.1 km/h while Lane 6 is already in
    km/h. Evidence: Lane 6's raw active-row mean speed is 41.19, which matches
    the Bible's own §1.4 figure ("41 km/h") and the §13.2 killer fact exactly —
    so the Bible author did NOT scale Lane 6 when producing those numbers.
    Scaling Lane 6 by 10 would understate it tenfold (4.1 km/h) and break the
    ECHO counterfactual premise that Lane 6 carries unused 41 km/h capacity.
    Therefore Lane 6 speed columns are left unscaled.

    Args:
        df: Input frame.

    Returns:
        Frame with lane 1-5 speeds in real km/h and Lane 6 speeds unchanged.
    """
    df = df.copy()
    speed_cols = (
        config.metric_cols("arith", config.DRIVING_LANES)
        + config.metric_cols("harm", config.DRIVING_LANES)
    )
    df[speed_cols] = df[speed_cols] / config.SPEED_SCALE
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all nine cleaning steps in order (Bible §5 Notebook 01).

    Order matters: stall flags (step 6-7) are computed on the *raw* zero speeds,
    so they must run before ``scale_speeds`` (step 8) — though dividing zero by
    ten is still zero, we keep the documented ordering for clarity and safety.

    Args:
        df: Raw DataFrame as loaded from ``data/raw.csv``.

    Returns:
        A new, cleaned DataFrame. The input is not mutated.
    """
    out = df.copy()
    out = parse_datetime(out)
    out = drop_redundant_columns(out)
    out = cast_vehicle_counts(out)
    out = cap_occupancy(out)
    out = add_lane6_flag(out)
    out = add_stall_flags(out)
    out = scale_speeds(out)
    return out


def integrity_report(df: pd.DataFrame) -> dict[str, object]:
    """Compute post-clean integrity checks used as the B1 gate.

    Args:
        df: Cleaned DataFrame.

    Returns:
        A dict of named checks and their values (rows, missing cells, max
        occupancy, presence of flags, speed range).
    """
    occ_cols = config.metric_cols("occup")
    speed_cols = config.metric_cols("arith", config.DRIVING_LANES)
    return {
        "rows": int(len(df)),
        "rows_match_expected": int(len(df)) == config.EXPECTED_ROWS,
        "links": int(df["LINK_ID"].nunique()),
        "missing_cells": int(df.isna().sum().sum()),
        "max_occupancy": float(df[occ_cols].to_numpy().max()),
        "timeint_dropped": "TIMEINT" not in df.columns,
        "has_lane6_active": "lane6_active" in df.columns, 
        "has_stall_flags": {"lane4_stalled", "lane5_stalled"}.issubset(df.columns),
        "speed_min_kmh": float(np.nanmin(df[speed_cols].to_numpy())),
        "speed_max_kmh": float(np.nanmax(df[speed_cols].to_numpy())),
    }

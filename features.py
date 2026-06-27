"""Feature engineering for UrbanPulse (Bible §5 Notebook 02).

Operates on the cleaned frame produced by B1. IMPORTANT unit note: the Bible's
feature formulas (e.g. "mean SPEEDAVGARITH lanes 1-5 / 10") assume *raw* input,
but ``cleaned.parquet`` already has lanes 1-5 speeds in km/h and occupancy
capped. So this module does NOT re-divide speeds by 10.

Feature groups:
  A  Cross-lane aggregates
  B  Speed-quality features
  C  Time features
  D  Custom KPI features (congestion_index, road_health_score)
plus the binary classification target.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import config

# Columns carried through as identifiers / grouping / split keys.
ID_COLS: list[str] = ["LINK_ID", "date", "day_number", "day_of_week", "hour"]
# Per-lane flags created in B1 that are kept as model features.
FLAG_COLS: list[str] = ["lane6_active", "lane4_stalled", "lane5_stalled"]


# --------------------------------------------------------------------------- #
# Group A — cross-lane aggregates
# --------------------------------------------------------------------------- #
def group_a_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Road-level cross-lane aggregates (Bible §5 Group A).

    Args:
        df: Cleaned frame.

    Returns:
        Frame of Group-A features aligned to ``df``'s index.
    """
    vehs = config.metric_cols("vehs")
    queue = config.metric_cols("queue")
    occ = config.metric_cols("occup", config.DRIVING_LANES)
    arith = config.metric_cols("arith", config.DRIVING_LANES)
    return pd.DataFrame(
        {
            "total_vehs": df[vehs].sum(axis=1),
            "mean_speed_kmh": df[arith].mean(axis=1),  # already km/h
            "mean_queue_s": df[queue].mean(axis=1),
            "max_queue_s": df[queue].max(axis=1),
            "mean_occup": df[occ].mean(axis=1),
            "max_occup": df[occ].max(axis=1),
            "lane_active_count": (df[vehs] > 0).sum(axis=1),
        }
    )


# --------------------------------------------------------------------------- #
# Group B — speed-quality features
# --------------------------------------------------------------------------- #
def group_b_speed_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Speed heterogeneity / stop-go features (Bible §5 Group B).

    Arith-harmonic gap measures stop-go intensity. Speeds are already km/h.

    Args:
        df: Cleaned frame.

    Returns:
        Frame of Group-B features.
    """
    arith = config.metric_cols("arith", config.DRIVING_LANES)
    harm = config.metric_cols("harm", config.DRIVING_LANES)
    div = df[arith].to_numpy() - df[harm].to_numpy()  # per-lane gap, lanes 1-5
    return pd.DataFrame(
        {
            "speed_div_L1": df["SPEEDAVGARITH(ALL)_1"] - df["SPEEDAVGHARM(ALL)_1"],
            "mean_speed_div": div.mean(axis=1),
            "speed_var_across_lanes": df[arith].var(axis=1),
        }
    )


# --------------------------------------------------------------------------- #
# Group C — time features
# --------------------------------------------------------------------------- #
def group_c_time(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal features incl. cyclical hour encoding (Bible §5 Group C).

    Args:
        df: Cleaned frame (already has ``hour``, ``day_of_week``, ``minute_of_day``).

    Returns:
        Frame of Group-C features.
    """
    hour = df["hour"].to_numpy()
    return pd.DataFrame(
        {
            "is_am_peak": df["hour"].isin(config.AM_PEAK_HOURS).astype("int64"),
            "is_pm_peak": df["hour"].isin(config.PM_PEAK_HOURS).astype("int64"),
            "is_weekend": (df["day_of_week"] >= config.WEEKEND_DOW_START).astype("int64"),
            "minute_of_day": df["minute_of_day"],
            "sin_hour": np.sin(2 * np.pi * hour / 24),
            "cos_hour": np.cos(2 * np.pi * hour / 24),
        }
    )


# --------------------------------------------------------------------------- #
# Group D — custom KPI features
# --------------------------------------------------------------------------- #
def _minmax(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Min-max scale a series to [0, 1], returning the constants used.

    When hi == lo (zero-variance feature, e.g. complete sensor outage on a
    link), span defaults to 1.0 so the output is uniformly 0.0. A warning is
    emitted so this silent behaviour is visible during debugging.
    """
    lo = float(series.min())
    hi = float(series.max())
    if hi > lo:
        span = hi - lo
    else:
        warnings.warn(
            f"_minmax: zero-variance feature '{series.name}' "
            f"(min=max={lo:.4f}). Normalised value will be 0.0 for all rows. "
            "This may indicate a sensor outage or constant-value input.",
            RuntimeWarning,
            stacklevel=2,
        )
        span = 1.0
    return (series - lo) / span, lo, hi


def transform_minmax(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Apply pre-fitted min-max scaling using constants from training time.

    This is the scoring-time counterpart to :func:`_minmax`. Use this instead
    of ``_minmax`` whenever you are transforming a subset of rows (per-link,
    single-observation, etc.) so the normalisation is consistent with what the
    model was trained on.

    Args:
        series: Values to normalise.
        lo:     Training-time minimum (from ``feature_norms.json``).
        hi:     Training-time maximum (from ``feature_norms.json``).

    Returns:
        Normalised Series clipped to [0, 1].
    """
    span = hi - lo if hi > lo else 1.0
    return ((series - lo) / span).clip(0.0, 1.0)


def load_feature_norms() -> dict[str, float]:
    """Load the KPI normalisation constants written by :func:`build_features`.

    Returns:
        Dict with keys ``queue_min``, ``queue_max``, ``speed_min``,
        ``speed_max`` — the global training-time min/max values.

    Raises:
        FileNotFoundError: If B2 has not been run yet.
    """
    path = config.FEATURE_NORMS_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"feature_norms.json not found at {path}. Run B2 (build_features) first."
        )
    return json.loads(path.read_text())



def group_d_kpis(group_a: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Composite congestion index and road health score (Bible §5 Group D).

    congestion_index = 0.40*norm_occup + 0.35*norm_queue + 0.25*norm_inv_speed,
    where norm_occup is already in [0,1] (capped occupancy), and queue / speed
    are min-max scaled. road_health_score = 100 - 100*congestion_index.

    Args:
        group_a: Output of :func:`group_a_aggregates` (provides occup/queue/speed).

    Returns:
        Tuple of (KPI frame, normalization constants for queue and speed).
    """
    norm_occup = group_a["mean_occup"]  # already 0-1
    norm_queue, q_lo, q_hi = _minmax(group_a["mean_queue_s"])
    norm_speed, s_lo, s_hi = _minmax(group_a["mean_speed_kmh"])
    norm_inv_speed = 1.0 - norm_speed

    w = config.CONGESTION_INDEX_WEIGHTS
    congestion_index = (
        w["occup"] * norm_occup
        + w["queue"] * norm_queue
        + w["inv_speed"] * norm_inv_speed
    ).clip(0.0, 1.0)

    kpis = pd.DataFrame(
        {
            "congestion_index": congestion_index,
            "road_health_score": 100.0 - 100.0 * congestion_index,
        }
    )
    norms = {
        "queue_min": q_lo,
        "queue_max": q_hi,
        "speed_min": s_lo,
        "speed_max": s_hi,
    }
    return kpis, norms


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
def make_target(group_a: pd.DataFrame) -> pd.Series:
    """Binary congestion target (Bible §5 Option A, recalibrated — decision #11).

    congested = 1 when mean_occup > 0.5 AND mean_queue_s > 238 s. The queue cut
    is recalibrated from the Bible's literal 400 s (which yields only 0.55%
    positive dataset-wide) so the positive rate lands at the intended ~13%.

    Args:
        group_a: Output of :func:`group_a_aggregates`.

    Returns:
        Integer Series named ``congested``.
    """
    target = (
        (group_a["mean_occup"] > config.TARGET_OCCUP_THRESHOLD)
        & (group_a["mean_queue_s"] > config.TARGET_QUEUE_THRESHOLD_S)
    ).astype("int64")
    target.name = "congested"
    return target


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, save_norms: bool = True) -> pd.DataFrame:
    """Build the full model-ready feature frame (Bible §5 Notebook 02).

    Args:
        df: Cleaned frame from B1.
        save_norms: If True, persist KPI normalization constants to
            ``data/feature_norms.json`` for reuse at scoring time.

    Returns:
        Feature frame: identifiers + Group A/B/C/D features + B1 flags + target.
        Contains no NaN.
    """
    group_a = group_a_aggregates(df)
    group_b = group_b_speed_quality(df)
    group_c = group_c_time(df)
    group_d, norms = group_d_kpis(group_a)
    target = make_target(group_a)

    features = pd.concat(
        [
            df[ID_COLS].reset_index(drop=True),
            group_a.reset_index(drop=True),
            group_b.reset_index(drop=True),
            group_c.reset_index(drop=True),
            group_d.reset_index(drop=True),
            df[FLAG_COLS].reset_index(drop=True),
            target.reset_index(drop=True),
        ],
        axis=1,
    )

    if save_norms:
        config.FEATURE_NORMS_JSON.parent.mkdir(parents=True, exist_ok=True)
        config.FEATURE_NORMS_JSON.write_text(json.dumps(norms, indent=2))

    return features


def feature_summary(features: pd.DataFrame) -> dict[str, object]:
    """Summary used as the B2 gate.

    Args:
        features: Output of :func:`build_features`.

    Returns:
        Dict with feature count, NaN count, and target positive rate.
    """
    feature_cols = [c for c in features.columns if c not in ID_COLS + ["congested"]]
    return {
        "rows": int(len(features)),
        "n_features": len(feature_cols),
        "nan_cells": int(features.isna().sum().sum()),
        "target_positive_rate": float(features["congested"].mean()),
    }

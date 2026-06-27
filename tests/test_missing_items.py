"""Tests for M9/M10 modeling changes and M2 features additions.

M9: hour must NOT appear in feature_columns() output.
M10: day_of_week IS kept (carries Mon/Tue signal beyond is_weekend).
M2:  transform_minmax and load_feature_norms are importable and correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import features as feat
import modeling


def _minimal_features_frame() -> pd.DataFrame:
    """A tiny features frame with all expected columns."""
    n = 10
    return pd.DataFrame({
        "LINK_ID": [36] * n, "date": pd.date_range("2024-07-01", periods=n, freq="5min"),
        "day_number": [1] * n, "day_of_week": [0] * n, "hour": [9] * n,
        "minute_of_day": range(540, 540 + n * 5, 5),
        "total_vehs": [100.0] * n, "mean_speed_kmh": [40.0] * n,
        "mean_queue_s": [300.0] * n, "max_queue_s": [400.0] * n,
        "mean_occup": [0.55] * n, "max_occup": [0.7] * n,
        "lane_active_count": [4] * n, "speed_std": [5.0] * n,
        "speed_divergence_mean": [3.0] * n, "lane6_active": [0] * n,
        "sin_hour": [0.9] * n, "cos_hour": [0.4] * n,
        "is_am_peak": [1] * n, "is_pm_peak": [0] * n, "is_weekend": [0] * n,
        "congestion_index": [0.45] * n, "road_health_score": [55.0] * n,
        "lane4_stalled": [0] * n, "lane5_stalled": [0] * n,
        "congested": [1] * n,
    })


# --------------------------------------------------------------------------- #
# M9: hour exclusion
# --------------------------------------------------------------------------- #

def test_m9_hour_not_in_non_leak_free_features():
    """M9: raw 'hour' integer must not appear in feature columns (any mode)."""
    df = _minimal_features_frame()
    cols = modeling.feature_columns(df, leak_free=False)
    assert "hour" not in cols, "hour must be excluded (redundant with sin_hour/cos_hour)"


def test_m9_hour_not_in_leak_free_features():
    df = _minimal_features_frame()
    cols = modeling.feature_columns(df, leak_free=True)
    assert "hour" not in cols


def test_m9_sin_cos_hour_still_present():
    """sin_hour and cos_hour (cyclical) must remain as actual features."""
    df = _minimal_features_frame()
    cols = modeling.feature_columns(df, leak_free=False)
    assert "sin_hour" in cols
    assert "cos_hour" in cols


def test_m9_hour_in_non_feature_cols():
    assert "hour" in modeling.NON_FEATURE_COLS


# --------------------------------------------------------------------------- #
# M10: day_of_week intentionally kept
# --------------------------------------------------------------------------- #

def test_m10_day_of_week_present_in_features():
    """M10: day_of_week carries Mon/Tue granularity beyond is_weekend; it stays."""
    df = _minimal_features_frame()
    cols = modeling.feature_columns(df, leak_free=False)
    assert "day_of_week" in cols, "day_of_week should remain (non-redundant)"


# --------------------------------------------------------------------------- #
# M2: transform_minmax
# --------------------------------------------------------------------------- #

def test_m2_transform_minmax_correct_normalisation():
    s = pd.Series([0.0, 500.0, 1000.0], name="mean_queue_s")
    result = feat.transform_minmax(s, lo=0.0, hi=2000.0)
    np.testing.assert_allclose(result.values, [0.0, 0.25, 0.5], atol=1e-9)


def test_m2_transform_minmax_clips_out_of_range():
    """Values outside [lo, hi] must be clipped to [0, 1]."""
    s = pd.Series([-100.0, 3000.0], name="mean_queue_s")
    result = feat.transform_minmax(s, lo=0.0, hi=2000.0)
    assert result.iloc[0] == 0.0, "below lo must clip to 0"
    assert result.iloc[1] == 1.0, "above hi must clip to 1"


def test_m2_transform_minmax_zero_variance_returns_zero():
    s = pd.Series([500.0, 500.0], name="const_feature")
    result = feat.transform_minmax(s, lo=500.0, hi=500.0)
    assert (result == 0.0).all()


def test_m2_load_feature_norms_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_NORMS_JSON", tmp_path / "ghost.json")
    with pytest.raises(FileNotFoundError):
        feat.load_feature_norms()

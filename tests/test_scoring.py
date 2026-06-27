"""Tests for the single-observation scoring API (MISSING-5).

Validates:
- score_link_interval works on a synthetic observation
- transform_minmax (M2) is used for KPI normalisation (not _minmax)
- score_batch handles errors gracefully
- load_feature_norms raises FileNotFoundError when norms are absent
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import features as feat
from scoring import score_link_interval, score_batch


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _mock_model(proba: float = 0.72):
    """Minimal sklearn-compatible classifier stub."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba, proba]])
    return model


def _minimal_reading() -> dict:
    """Minimum synthetic sensor reading compatible with cleaned.parquet schema."""
    row = {
        "LINK_ID": 36,
        "date": pd.Timestamp("2024-07-01 09:45:00"),
        "day_number": 1,
        "hour": 9,
        "day_of_week": 0,
        "minute_of_day": 585,
        "lane6_active": 0,
        "lane4_stalled": 0,
        "lane5_stalled": 0,
    }
    # Add per-lane columns with realistic values
    for lane in (1, 2, 3, 4, 5):
        arith_col = f"{config.METRIC_PREFIXES['arith']}_{lane}"
        harm_col  = f"{config.METRIC_PREFIXES['harm']}_{lane}"
        vehs_col  = f"{config.METRIC_PREFIXES['vehs']}_{lane}"
        queue_col = f"{config.METRIC_PREFIXES['queue']}_{lane}"
        occ_col   = f"{config.METRIC_PREFIXES['occup']}_{lane}"
        row[arith_col] = 30.0
        row[harm_col]  = 28.0
        row[vehs_col]  = 120
        row[queue_col] = 400.0
        row[occ_col]   = 0.65
    # Lane 6
    row[f"{config.METRIC_PREFIXES['arith']}_6"] = 50.0
    row[f"{config.METRIC_PREFIXES['harm']}_6"]  = 48.0
    row[f"{config.METRIC_PREFIXES['vehs']}_6"]  = 0
    row[f"{config.METRIC_PREFIXES['queue']}_6"] = 0.0
    row[f"{config.METRIC_PREFIXES['occup']}_6"] = 0.0
    return row


_SYNTHETIC_NORMS = {
    "queue_min": 0.0, "queue_max": 2000.0,
    "speed_min": 0.0, "speed_max": 120.0,
}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_score_link_interval_returns_required_keys():
    model = _mock_model(0.72)
    result = score_link_interval(_minimal_reading(), model, norms=_SYNTHETIC_NORMS)
    required_keys = {
        "link_id", "timestamp", "congestion_prob", "state",
        "health_score", "congestion_index", "risk_score", "archetype", "features",
    }
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"


def test_score_link_interval_prob_matches_model():
    model = _mock_model(0.85)
    result = score_link_interval(_minimal_reading(), model, norms=_SYNTHETIC_NORMS)
    assert abs(result["congestion_prob"] - 0.85) < 0.01


def test_score_link_interval_health_in_bounds():
    model = _mock_model(0.5)
    result = score_link_interval(_minimal_reading(), model, norms=_SYNTHETIC_NORMS)
    assert 0.0 <= result["health_score"] <= 100.0


def test_score_link_interval_state_is_valid():
    model = _mock_model(0.5)
    result = score_link_interval(_minimal_reading(), model, norms=_SYNTHETIC_NORMS)
    valid_states = {"Healthy", "Stressed", "Saturated", "Collapsed"}
    assert result["state"] in valid_states


def test_score_link_interval_congestion_index_in_bounds():
    model = _mock_model(0.5)
    result = score_link_interval(_minimal_reading(), model, norms=_SYNTHETIC_NORMS)
    assert 0.0 <= result["congestion_index"] <= 1.0


def test_score_link_interval_uses_transform_minmax_not_local_scale():
    """With training norms, KPI must be consistent regardless of single-row range."""
    model = _mock_model(0.5)
    reading = _minimal_reading()
    reading[f"{config.METRIC_PREFIXES['queue']}_1"] = 500.0  # high queue
    # Use fixed norms: global max = 2000s -> normalised = 500/2000 = 0.25
    result = score_link_interval(reading, model, norms=_SYNTHETIC_NORMS)
    # If _minmax was used on 1 row: queue normalised = 0.0 (hi==lo). CI would differ.
    # With transform_minmax: CI > 0 (queue contributes 0.25 * 0.35 > 0)
    assert result["congestion_index"] > 0.0, "transform_minmax should give non-zero CI"


def test_score_batch_handles_error_gracefully():
    """score_batch must not raise even when one reading is malformed."""
    model = _mock_model(0.5)
    readings = [_minimal_reading(), {"LINK_ID": 99}]  # second is missing cols
    results = score_batch(readings, model, norms=_SYNTHETIC_NORMS)
    assert len(results) == 2
    assert results[0]["congestion_prob"] > 0
    assert "error" in results[1]


def test_load_feature_norms_raises_when_missing(tmp_path, monkeypatch):
    """load_feature_norms raises FileNotFoundError if norms file is absent."""
    monkeypatch.setattr(config, "FEATURE_NORMS_JSON", tmp_path / "nonexistent.json")
    with pytest.raises(FileNotFoundError, match="feature_norms.json not found"):
        feat.load_feature_norms()

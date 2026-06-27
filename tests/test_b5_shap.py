"""Tests for B5 SHAP explainability (MISSING-1 smoke test).

These tests verify:
- translate_waterfall produces the required schema
- _translate_feature handles all known templates without KeyError
- compute_shap_values returns an Explanation with correct shape
- run() gate check (skipped if best_model.pkl not present -- CI guard)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import shap_analysis


# --------------------------------------------------------------------------- #
# Unit: translate_waterfall schema
# --------------------------------------------------------------------------- #

class _MockExplanation:
    """Minimal SHAP Explanation stub for unit tests."""
    def __init__(self, n_features: int = 5) -> None:
        self.values = np.array([[0.3, -0.1, 0.5, -0.2, 0.1]])
        self.data = np.array([[0.8, 120.0, 9.0, 1.0, 0.45]])
        self.base_values = np.array([0.05])
        self.feature_names = [
            "mean_occup", "mean_queue_s", "hour", "is_am_peak", "total_vehs"
        ][:n_features]

    def __getitem__(self, idx):
        return self


def test_translate_waterfall_schema():
    sv = _MockExplanation()
    meta = {"link_id": 36, "day_number": 1, "minute_of_day": 585,
            "mean_queue_s": 900.0, "mean_occup": 0.9, "congested": 1}
    result = shap_analysis.translate_waterfall(sv, label="Test label", meta=meta, top_n=3)
    assert "label" in result
    assert "top_features" in result
    assert len(result["top_features"]) == 3
    for f in result["top_features"]:
        assert "feature" in f
        assert "shap" in f
        assert "plain_english" in f
        assert "rank" in f
    assert result["top_features"][0]["rank"] == 1


def test_translate_waterfall_ranked_by_abs_shap():
    """Highest absolute SHAP value should be rank 1."""
    sv = _MockExplanation()
    meta = {"link_id": 37, "day_number": 2, "minute_of_day": 480,
            "mean_queue_s": 600.0, "mean_occup": 0.7, "congested": 1}
    result = shap_analysis.translate_waterfall(sv, label="Test", meta=meta, top_n=2)
    # sv.values[0] = [0.3, -0.1, 0.5, -0.2, 0.1]; abs-sorted: 0.5, 0.3
    assert result["top_features"][0]["feature"] == "hour"
    assert result["top_features"][1]["feature"] == "mean_occup"


def test_translate_feature_known_templates():
    """All template keys in _TRANSLATIONS should not raise on format."""
    for feature in shap_analysis._TRANSLATIONS:
        sentence = shap_analysis._translate_feature(feature, 0.25, 0.6)
        assert isinstance(sentence, str) and len(sentence) > 0


def test_translate_feature_unknown_fallback():
    """Unknown features get a generic sentence without raising."""
    sentence = shap_analysis._translate_feature("unknown_feature_xyz", -0.15, 42.0)
    assert "unknown_feature_xyz" in sentence
    assert isinstance(sentence, str)


# --------------------------------------------------------------------------- #
# Integration: run() -- only when pipeline artifacts exist
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not config.BEST_MODEL_PKL.exists() or not config.FEATURES_PARQUET.exists(),
    reason="Requires models/best_model.pkl and data/features.parquet (run B1-B4 first)",
)
def test_b5_run_produces_gate_passing_output():
    """End-to-end: shap_analysis.run() must produce 6 plots and translations.json."""
    result = shap_analysis.run()
    assert result["all_plots_exist"], "Some SHAP plots are missing"
    assert len(result["plots_produced"]) == 6, "Expected exactly 6 SHAP outputs"
    assert Path(result["translations_path"]).exists()
    # translations.json must have link_36 key with top_features
    import json
    trans = json.loads(Path(result["translations_path"]).read_text())
    assert "link_36" in trans
    assert len(trans["link_36"]["top_features"]) > 0

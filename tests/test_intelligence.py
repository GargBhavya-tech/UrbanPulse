"""Tests for the Traffic Intelligence Engine (Bible §6)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from engine import intelligence as eng


def test_road_state_bands() -> None:
    assert eng.road_state(85) == "Healthy"
    assert eng.road_state(55) == "Stressed"
    assert eng.road_state(30) == "Saturated"
    assert eng.road_state(10) == "Collapsed"


def test_severity_mapping() -> None:
    assert eng.severity_for_state("Healthy") == "NONE"
    assert eng.severity_for_state("Collapsed") == "CRITICAL"


def test_risk_score_percentile() -> None:
    hist = {5: np.array([0.1, 0.2, 0.3, 0.4, 0.5])}
    # value at/above all history -> 100th percentile
    assert eng.congestion_risk_score(5, 0.5, hist) == 100.0
    # below all -> 0
    assert eng.congestion_risk_score(5, 0.05, hist) == 0.0
    # unknown link -> neutral 50
    assert eng.congestion_risk_score(999, 0.3, hist) == 50.0


def test_rule_any_speed_divergence_fires() -> None:
    row = pd.Series({"mean_speed_div": 6.0, "mean_queue_s": 350, "is_am_peak": 0,
                     "max_occup": 0.5, "lane5_stalled": 0, "max_queue_s": 350})
    recs = eng.recommend(row, risk_score=10)
    assert any("traffic police" in r.recommendation.lower() for r in recs)


def test_archetype_gate_blocks_when_mismatched() -> None:
    row = pd.Series({"max_occup": 0.95, "mean_queue_s": 600, "is_am_peak": 0,
                     "mean_speed_div": 0, "lane5_stalled": 0, "max_queue_s": 600})
    # Saturator rule should fire only for Saturator (or unknown) archetype
    assert eng.recommend(row, 10, archetype="Saturator")
    assert not any(
        r.archetype == "Saturator" for r in eng.recommend(row, 10, archetype="Commuter")
    )


def test_every_recommendation_has_reasoning() -> None:
    # A link has ONE archetype: Landmine here. Triggers Any rules (speed_div,
    # surge, cascade) + the Landmine rule -> all must carry reasoning.
    row = pd.Series({"mean_speed_div": 6.0, "mean_queue_s": 600, "is_am_peak": 1,
                     "max_occup": 0.95, "lane5_stalled": 1, "max_queue_s": 600,
                     "health_lt30_3consec": True})
    recs = eng.recommend(row, risk_score=80, archetype="Landmine",
                         prev_queue_s=100, cascade_propagating=True)
    assert len(recs) >= 4
    assert all(r.reasoning.strip() for r in recs)


def test_archetype_rules_silent_when_unknown() -> None:
    # With archetype unknown, only "Any" rules may fire (no archetype flood).
    row = pd.Series({"mean_speed_div": 6.0, "mean_queue_s": 350, "is_am_peak": 1,
                     "max_occup": 0.95, "lane5_stalled": 1, "max_queue_s": 600,
                     "health_lt30_3consec": True})
    recs = eng.recommend(row, risk_score=90, archetype=None)
    assert all(r.archetype == "Any" for r in recs)


def test_queue_surge_rule() -> None:
    row = pd.Series({"mean_queue_s": 400, "mean_speed_div": 0, "is_am_peak": 0,
                     "max_occup": 0.4, "lane5_stalled": 0, "max_queue_s": 400})
    recs = eng.recommend(row, 10, prev_queue_s=150)  # surge of 250 > 200
    assert any("downstream" in r.recommendation.lower() for r in recs)


def test_critical_alert_on_high_prob() -> None:
    alerts = eng.build_alerts("Healthy", prob=0.9, mean_queue_s=100)
    assert any(a.severity == "CRITICAL" for a in alerts)


@pytest.mark.skipif(
    not (config.BEST_MODEL_PKL.exists() and config.FEATURES_PARQUET.exists()),
    reason="model/features not built",
)
def test_snapshot_end_to_end() -> None:
    import joblib
    import io_utils

    model = joblib.load(config.BEST_MODEL_PKL)
    features = io_utils.load_parquet(config.FEATURES_PARQUET)
    result = eng.analyze_snapshot(features, model, day_number=1, minute_of_day=585)
    assert result["n_links"] == config.EXPECTED_LINKS
    assert len(result["hotspot_ranking"]) == config.EXPECTED_LINKS
    # ranking is worst-first (non-decreasing health)
    hs = [r["health_score"] for r in result["hotspot_ranking"]]
    assert hs == sorted(hs)

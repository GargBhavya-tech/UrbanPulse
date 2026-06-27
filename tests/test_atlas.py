"""Tests for ECHO Stage A — Personality Atlas (Bible §7)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from echo import personality_atlas as pa


def _mini_features() -> pd.DataFrame:
    """Tiny synthetic feature frame with the columns the fingerprint needs."""
    rng = np.random.default_rng(0)
    rows = []
    for link in range(1, 7):
        for h in range(24):
            for k in range(2):
                rows.append({
                    "LINK_ID": link,
                    "date": pd.Timestamp("2024-07-01") + pd.Timedelta(hours=h, minutes=30 * k),
                    "day_number": 1,
                    "hour": h,
                    "minute_of_day": h * 60 + 30 * k,
                    "mean_queue_s": rng.uniform(0, 800),
                    "mean_occup": rng.uniform(0, 1),
                    "mean_speed_div": rng.uniform(0, 6),
                    "lane6_active": rng.integers(0, 2),
                    "lane4_stalled": 0,
                    "lane5_stalled": 0,
                })
    return pd.DataFrame(rows)


def test_fingerprint_shape_and_columns() -> None:
    fp = pa.extract_fingerprints(_mini_features())
    assert list(fp.columns) == pa.FINGERPRINT_COLS
    assert fp.isna().sum().sum() == 0
    assert len(fp) == 6


def test_adjacency_is_symmetric_binary() -> None:
    adj, ids = pa.build_adjacency(_mini_features())
    assert adj.shape == (6, 6)
    assert set(np.unique(adj)) <= {0.0, 1.0}
    assert np.allclose(adj, adj.T)
    assert np.all(np.diag(adj) == 0)


@pytest.mark.skipif(
    not config.FEATURES_PARQUET.exists(), reason="features.parquet not built"
)
def test_run_produces_honest_archetypes() -> None:
    import json

    out = pa.run()
    assert out["n_archetypes"] in (5, 6, 7)
    assert config.ROAD_ARCHETYPES_JSON.exists()
    d = json.loads(config.ROAD_ARCHETYPES_JSON.read_text())
    assert len(d) == config.EXPECTED_LINKS
    # Anchored exemplars hold; Landmine is data-driven (may be absent).
    assert d["37"]["archetype"] == "Chronic"
    assert d["5"]["archetype"] == "Saturator"
    # Every record has the required fields.
    for rec in d.values():
        assert {"archetype", "confidence", "stability_score", "stable"} <= rec.keys()

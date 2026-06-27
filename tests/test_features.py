"""Unit tests for the features module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import features as feat


@pytest.fixture
def cleaned() -> pd.DataFrame:
    """Small synthetic cleaned frame (post-B1 schema)."""
    rng = np.random.default_rng(0)
    n = 50
    data: dict[str, object] = {
        "LINK_ID": rng.integers(1, 67, n),
        "date": pd.date_range("2024-07-01", periods=n, freq="5min"),
        "day_number": 1,
        "day_of_week": rng.integers(0, 7, n),
        "hour": rng.integers(0, 24, n),
        "minute_of_day": rng.integers(0, 1440, n),
        "lane6_active": rng.integers(0, 2, n),
        "lane4_stalled": 0,
        "lane5_stalled": 0,
    }
    for lane in config.LANES:
        data[f"VEHS(ALL)_{lane}"] = rng.integers(0, 250, n)
        data[f"SPEEDAVGARITH(ALL)_{lane}"] = rng.uniform(5, 40, n)
        data[f"SPEEDAVGHARM(ALL)_{lane}"] = rng.uniform(5, 38, n)
        data[f"QUEUEDELAY(ALL)_{lane}"] = rng.uniform(0, 800, n)
        data[f"OCCUPRATE(ALL)_{lane}"] = rng.uniform(0, 1, n)
    return pd.DataFrame(data)


def test_group_a_has_expected_columns(cleaned: pd.DataFrame) -> None:
    a = feat.group_a_aggregates(cleaned)
    assert {"total_vehs", "mean_speed_kmh", "mean_queue_s", "mean_occup"} <= set(a.columns)
    assert (a["max_occup"] >= a["mean_occup"]).all()


def test_speeds_not_rescaled(cleaned: pd.DataFrame) -> None:
    # mean_speed_kmh must equal the plain mean of lanes 1-5 (no extra /10)
    a = feat.group_a_aggregates(cleaned)
    expected = cleaned[config.metric_cols("arith", config.DRIVING_LANES)].mean(axis=1)
    pd.testing.assert_series_equal(a["mean_speed_kmh"], expected, check_names=False)


def test_kpis_in_bounds(cleaned: pd.DataFrame) -> None:
    a = feat.group_a_aggregates(cleaned)
    kpis, _ = feat.group_d_kpis(a)
    assert kpis["congestion_index"].between(0, 1).all()
    assert kpis["road_health_score"].between(0, 100).all()


def test_target_is_binary(cleaned: pd.DataFrame) -> None:
    a = feat.group_a_aggregates(cleaned)
    t = feat.make_target(a)
    assert set(t.unique()) <= {0, 1}


def test_build_features_no_nan(cleaned: pd.DataFrame) -> None:
    f = feat.build_features(cleaned, save_norms=False)
    assert f.isna().sum().sum() == 0


@pytest.mark.skipif(
    not config.FEATURES_PARQUET.exists(), reason="features.parquet not built yet"
)
def test_target_rate_on_real_data() -> None:
    import io_utils

    f = io_utils.load_parquet(config.FEATURES_PARQUET)
    rate = f["congested"].mean()
    assert 0.10 <= rate <= 0.16, f"target rate {rate:.4f} outside ~13% band"
    assert f.isna().sum().sum() == 0

"""Tests for the modeling core (leak-safe target shifting + split)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import modeling


def _seq_frame() -> pd.DataFrame:
    """Two links, ordered intervals spanning the train→val day boundary (10→11)."""
    rows = []
    for link in (1, 2):
        for day in (10, 11):
            for i in range(5):
                rows.append(
                    {
                        "LINK_ID": link,
                        "date": pd.Timestamp("2024-07-01")
                        + pd.Timedelta(days=day - 1, minutes=5 * i),
                        "day_number": day,
                        "congested": (i % 2),  # 0,1,0,1,0
                        "feat_a": float(i),
                        "mean_occup": 0.6,
                        "mean_queue_s": 300.0,
                        "max_occup": 0.7,
                        "max_queue_s": 320.0,
                        "congestion_index": 0.5,
                        "road_health_score": 50.0,
                    }
                )
    return pd.DataFrame(rows)


def test_feature_columns_leak_free_drops_defining_cols() -> None:
    df = _seq_frame()
    cols = modeling.feature_columns(df, leak_free=True)
    assert not (set(modeling.LEAKY_COLS) & set(cols))
    assert "feat_a" in cols
    # non-leak-free keeps them
    assert set(modeling.LEAKY_COLS) <= set(modeling.feature_columns(df, leak_free=False))


def test_shift_target_uses_future_label() -> None:
    df = _seq_frame()
    shifted = modeling.shift_target(df, horizon=1)
    # within link 1, day 11 (clean, no boundary): congested seq 0,1,0,1,0
    # shifting -1 makes labels 1,0,1,0 for first four rows
    link1_day11 = shifted[(shifted.LINK_ID == 1) & (shifted.day_number == 11)]
    assert list(link1_day11["congested"]) == [1, 0, 1, 0]


def test_shift_target_blocks_cross_split_leakage() -> None:
    df = _seq_frame()
    shifted = modeling.shift_target(df, horizon=2)
    # No train-day (10) row may carry a label sourced from a val-day (11) row.
    # The last 2 intervals of day 10 would pull from day 11 -> must be dropped.
    day10 = shifted[shifted.day_number == 10]
    # original day-10 has 5 rows/link; last 2 drop -> at most 3 remain per link
    assert (day10.groupby("LINK_ID").size() <= 3).all()


def test_temporal_split_day_ranges() -> None:
    df = _seq_frame()
    train, val, test = modeling.temporal_split(df)
    assert set(train["day_number"].unique()) <= set(modeling.TRAIN_DAYS)
    assert set(val["day_number"].unique()) <= set(modeling.VAL_DAYS)
    assert len(test) == 0  # no test-day rows in this fixture


def test_target_encode_link_fit_on_train_only() -> None:
    df = _seq_frame()
    train = df[df.day_number == 10].copy()
    enc_train, = modeling.target_encode_link(train, [train])
    # encoded values are bounded in [0, 1] (smoothed rates)
    assert enc_train.between(0, 1).all()


@pytest.mark.skipif(
    not config.FEATURES_PARQUET.exists(), reason="features.parquet not built"
)
def test_prepare_xy_no_nan_and_shapes() -> None:
    import io_utils

    df = io_utils.load_parquet(config.FEATURES_PARQUET)
    x_train, x_val, x_test, y = modeling.prepare_xy(df, horizon=3, leak_free=False)
    assert x_train.isna().sum().sum() == 0
    assert len(x_train) == len(y["train"]) > 0
    assert x_train.shape[1] == x_test.shape[1]

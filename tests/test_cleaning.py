"""Unit tests for the cleaning module.

Most tests run on a tiny synthetic frame so they are fast and deterministic. One
integration test runs the integrity gate on the real cleaned parquet if it has
been produced (skipped otherwise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cleaning
import config


def _raw_row() -> dict[str, object]:
    """One synthetic raw row with all 34 columns populated."""
    row: dict[str, object] = {
        "TIMEINT": "900-1200",
        "date": "2024-07-01 9:45",
        "LINK_ID": 36,
        "DAY": 1,
    }
    for lane in config.LANES:
        row[f"VEHS(ALL)_{lane}"] = 100.0
        row[f"SPEEDAVGARITH(ALL)_{lane}"] = 200.0
        row[f"SPEEDAVGHARM(ALL)_{lane}"] = 180.0
        row[f"QUEUEDELAY(ALL)_{lane}"] = 300.0
        row[f"OCCUPRATE(ALL)_{lane}"] = 0.5
    return row


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame([_raw_row()])


def test_drop_redundant_removes_timeint(raw: pd.DataFrame) -> None:
    out = cleaning.drop_redundant_columns(raw)
    assert "TIMEINT" not in out.columns


def test_cap_occupancy_clips_above_one() -> None:
    df = pd.DataFrame({"OCCUPRATE(ALL)_2": [0.3, 1.5, 3.02]})
    # add the other occupancy cols so metric_cols selection works
    for lane in config.LANES:
        col = f"OCCUPRATE(ALL)_{lane}"
        if col not in df:
            df[col] = 0.0
    out = cleaning.cap_occupancy(df)
    assert out["OCCUPRATE(ALL)_2"].max() == config.OCCUPANCY_CAP


def test_lane6_flag_is_binary(raw: pd.DataFrame) -> None:
    raw.loc[0, "VEHS(ALL)_6"] = 0
    out = cleaning.add_lane6_flag(raw)
    assert out["lane6_active"].iloc[0] == 0
    raw.loc[0, "VEHS(ALL)_6"] = 3
    out = cleaning.add_lane6_flag(raw)
    assert out["lane6_active"].iloc[0] == 1


def test_stall_flag_fires_on_zero_speed_with_vehicles(raw: pd.DataFrame) -> None:
    raw.loc[0, "VEHS(ALL)_5"] = 10
    raw.loc[0, "SPEEDAVGARITH(ALL)_5"] = 0
    out = cleaning.add_stall_flags(raw)
    assert out["lane5_stalled"].iloc[0] == 1


def test_speed_scaling_excludes_lane6(raw: pd.DataFrame) -> None:
    out = cleaning.scale_speeds(raw)
    # lanes 1-5 divided by 10
    assert out["SPEEDAVGARITH(ALL)_1"].iloc[0] == pytest.approx(20.0)
    # lane 6 left unscaled (already km/h)
    assert out["SPEEDAVGARITH(ALL)_6"].iloc[0] == pytest.approx(200.0)


def test_clean_does_not_mutate_input(raw: pd.DataFrame) -> None:
    before = raw.copy()
    cleaning.clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_produces_all_flags(raw: pd.DataFrame) -> None:
    out = cleaning.clean(raw)
    for col in ("lane6_active", "lane4_stalled", "lane5_stalled", "hour"):
        assert col in out.columns


@pytest.mark.skipif(
    not config.CLEANED_PARQUET.exists(), reason="cleaned.parquet not built yet"
)
def test_integrity_gate_on_real_data() -> None:
    import io_utils

    df = io_utils.load_parquet(config.CLEANED_PARQUET)
    report = cleaning.integrity_report(df)
    assert report["rows_match_expected"]
    assert report["missing_cells"] == 0
    assert report["max_occupancy"] <= config.OCCUPANCY_CAP
    assert report["links"] == config.EXPECTED_LINKS
    assert report["has_lane6_active"]
    assert report["has_stall_flags"]

"""Single-observation scoring API for UrbanPulse (MISSING-5).

Provides a minimal streaming/online inference path that takes ONE raw sensor
reading (one link, one 5-min interval) and returns the full structured
prediction without requiring the full 266k-row features parquet in memory.

The key difference from the batch pipeline:
  - `build_features` fits min-max norms on the full 266k-row dataset.
  - `score_link_interval` loads pre-fitted norms from `feature_norms.json`
    (written by B2) and applies them via `transform_minmax`, so the
    normalisation is identical to what the model saw at training time.

Usage::

    import joblib
    from scoring import score_link_interval

    model = joblib.load("models/best_model.pkl")
    reading = {
        "LINK_ID": 36,
        "date": pd.Timestamp("2024-07-01 09:45:00"),
    }
    result = score_link_interval(reading, model)
    print(result["congestion_prob"], result["state"], result["health_score"])
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd

import config
import features as feat
import modeling


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def score_link_interval(
    reading: dict[str, Any],
    model: Any,
    *,
    link_congestion_rate: Optional[float] = None,
    norms: Optional[dict[str, float]] = None,
    archetype: Optional[str] = None,
    historical_risk_percentiles: Optional[dict[int, np.ndarray]] = None,
) -> dict[str, Any]:
    """Score a single cleaned sensor reading for one link and one 5-min interval.

    Args:
        reading: Dict of cleaned sensor columns (same schema as cleaned.parquet rows).
        model: Fitted sklearn-compatible classifier with predict_proba.
        link_congestion_rate: Smoothed target-encoded congestion rate for this link.
            When omitted, global training-mean (~0.131) is used.
        norms: Pre-loaded feature_norms.json dict.  Loaded automatically if omitted.
        archetype: Road archetype string from B7 (optional).
        historical_risk_percentiles: Output of link_risk_percentiles for risk score.

    Returns:
        Dict with link_id, timestamp, congestion_prob, state, health_score,
        congestion_index, risk_score, archetype, features.
    """
    row_df = pd.DataFrame([reading])

    if "date" in row_df.columns:
        row_df["date"] = pd.to_datetime(row_df["date"])

    if "hour" not in row_df.columns:
        row_df["hour"] = row_df["date"].dt.hour
    if "minute_of_day" not in row_df.columns:
        row_df["minute_of_day"] = row_df["date"].dt.hour * 60 + row_df["date"].dt.minute
    if "day_of_week" not in row_df.columns:
        row_df["day_of_week"] = row_df["date"].dt.dayofweek
    if "day_number" not in row_df.columns:
        row_df["day_number"] = 1

    if "lane6_active" not in row_df.columns:
        vehs_6_col = f"{config.METRIC_PREFIXES['vehs']}_6"
        if vehs_6_col in row_df.columns:
            row_df["lane6_active"] = (row_df[vehs_6_col] > 0).astype(int)
        else:
            row_df["lane6_active"] = 0

    for lane in (4, 5):
        col = f"lane{lane}_stalled"
        if col not in row_df.columns:
            vehs_col = f"{config.METRIC_PREFIXES['vehs']}_{lane}"
            arith_col = f"{config.METRIC_PREFIXES['arith']}_{lane}"
            if vehs_col in row_df.columns and arith_col in row_df.columns:
                row_df[col] = ((row_df[vehs_col] > 0) & (row_df[arith_col] == 0)).astype(int)
            else:
                row_df[col] = 0

    group_a = feat.group_a_aggregates(row_df)
    group_b = feat.group_b_speed_quality(row_df)
    group_c = feat.group_c_time(row_df)

    if norms is None:
        norms = feat.load_feature_norms()

    norm_occup = group_a["mean_occup"].clip(0.0, 1.0)
    norm_queue = feat.transform_minmax(group_a["mean_queue_s"], lo=norms["queue_min"], hi=norms["queue_max"])
    norm_speed = feat.transform_minmax(group_a["mean_speed_kmh"], lo=norms["speed_min"], hi=norms["speed_max"])
    norm_inv_speed = 1.0 - norm_speed

    w = config.CONGESTION_INDEX_WEIGHTS
    congestion_index = float(
        (w["occup"] * norm_occup + w["queue"] * norm_queue + w["inv_speed"] * norm_inv_speed)
        .clip(0.0, 1.0).iloc[0]
    )
    health_score = round(100.0 - 100.0 * congestion_index, 2)

    group_d = pd.DataFrame({"congestion_index": [congestion_index], "road_health_score": [health_score]})
    flag_cols = [c for c in ["lane6_active", "lane4_stalled", "lane5_stalled"] if c in row_df.columns]

    full_row = pd.concat([
        row_df[feat.ID_COLS].reset_index(drop=True),
        group_a.reset_index(drop=True),
        group_b.reset_index(drop=True),
        group_c.reset_index(drop=True),
        group_d.reset_index(drop=True),
        row_df[flag_cols].reset_index(drop=True),
    ], axis=1)
    full_row["congested"] = 0

    feat_cols = modeling.feature_columns(full_row, leak_free=False)
    x = full_row[feat_cols].copy()
    x["link_congestion_rate"] = link_congestion_rate if link_congestion_rate is not None else 0.131

    proba = float(model.predict_proba(x)[:, 1][0])

    from engine.intelligence import road_state, congestion_risk_score
    state = road_state(health_score)

    risk = 50.0
    if historical_risk_percentiles is not None:
        link_id = int(reading.get("LINK_ID", 0))
        risk = congestion_risk_score(link_id, congestion_index, historical_risk_percentiles)

    return {
        "link_id": int(reading.get("LINK_ID", -1)),
        "timestamp": str(reading.get("date", "")),
        "congestion_prob": round(proba, 4),
        "state": state,
        "health_score": health_score,
        "congestion_index": round(congestion_index, 4),
        "risk_score": risk,
        "archetype": archetype,
        "features": x.iloc[0].to_dict(),
    }


def score_batch(
    readings: list[dict[str, Any]],
    model: Any,
    *,
    link_congestion_rates: Optional[dict[int, float]] = None,
    norms: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """Score a list of cleaned sensor readings.

    Loads norms once from disk and reuses across all readings for efficiency.
    """
    if norms is None:
        norms = feat.load_feature_norms()
    results = []
    for r in readings:
        link_id = int(r.get("LINK_ID", 0))
        lcr = (link_congestion_rates or {}).get(link_id)
        try:
            result = score_link_interval(r, model, link_congestion_rate=lcr, norms=norms)
        except Exception as exc:
            result = {"link_id": link_id, "error": str(exc)}
        results.append(result)
    return results

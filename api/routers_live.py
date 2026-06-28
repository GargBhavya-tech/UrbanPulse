"""B11 — live engine snapshot router.

The one endpoint that runs the B6 Traffic Intelligence Engine on demand for a
chosen (day, minute) snapshot, using the B4 best model for the +15 min
congestion probability. Requires both the features parquet and the trained
model on disk; if either is missing it returns 503 with a clear hint, so the
read-only artifact endpoints still work after a partial pipeline run.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import APIRouter, HTTPException, Query

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from api import store
from api.schemas import SnapshotResponse
from engine import intelligence

router = APIRouter()

_MODEL_CACHE: dict[str, object] = {}


def _load_model() -> object:
    if "model" not in _MODEL_CACHE:
        if not config.BEST_MODEL_PKL.exists():
            raise HTTPException(
                503,
                "Best model not found. Run `python train.py` and `python compare.py` "
                "to produce models/best_model.pkl before using /snapshot.",
            )
        _MODEL_CACHE["model"] = joblib.load(config.BEST_MODEL_PKL)
    return _MODEL_CACHE["model"]


@router.get("/snapshot", tags=["engine"], response_model=SnapshotResponse)
def snapshot(
    day: int = Query(config.API_DEMO_DAY, ge=1, le=14),
    minute: int = Query(config.API_DEMO_MINUTE, ge=0, le=1435),
) -> SnapshotResponse:
    """B6 — full engine analysis for one 5-min interval across all 66 links.

    Health score, metabolic state, risk percentile, +15min congestion prob,
    hotspot ranking, severity alerts, and archetype-aware recommendations.
    """
    if minute % config.INTERVAL_MINUTES != 0:
        raise HTTPException(422, f"minute must be a multiple of {config.INTERVAL_MINUTES}")

    feats = store.features()
    if feats is None:
        raise HTTPException(
            503,
            "features.parquet not found. Run `python scripts/02_features.py` first.",
        )

    model = _load_model()
    eco = {
        int(lid): rec.get("regime") == "backpressure"
        for lid, rec in store.ecosystem_state().get("links", {}).items()
    }
    result = intelligence.analyze_snapshot(
        features=feats,
        model=model,
        day_number=day,
        minute_of_day=minute,
        archetypes=store.archetype_map(),
        ecosystem_state=eco,
    )
    if not result.get("links"):
        raise HTTPException(404, f"No data for day={day}, minute={minute}")
    return SnapshotResponse(**result)

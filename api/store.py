"""B11 — artifact store.

The API is read-only over the precomputed pipeline artifacts (B1-B10). This
module loads each artifact once, caches it, and degrades gracefully when an
artifact is missing (so the API still boots after a partial pipeline run).

Nothing here recomputes anything — recomputation is the job of the B1-B10
scripts. The store only reads what those scripts already wrote to disk.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# --------------------------------------------------------------------------- #
# Low-level loaders (cached). Each returns None if the artifact is absent.
# --------------------------------------------------------------------------- #

def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def archetypes() -> dict[int, dict]:
    """B7 — LINK_ID -> {archetype, confidence, stability_score, stable}."""
    raw = _read_json(config.ROAD_ARCHETYPES_JSON) or {}
    return {int(k): v for k, v in raw.items()}


@lru_cache(maxsize=1)
def causal_graph() -> dict:
    """B8 — {nodes: [...], edges: [{source, target, lag_minutes, ...}]}."""
    return _read_json(config.CAUSAL_GRAPH_JSON) or {"nodes": [], "edges": []}


@lru_cache(maxsize=1)
def ecosystem_state() -> dict:
    """B8 — {day_number, minute_of_day, links: {LINK_ID: {state, regime, ...}}}."""
    return _read_json(config.ECOSYSTEM_STATE_JSON) or {
        "day_number": config.API_DEMO_DAY,
        "minute_of_day": config.API_DEMO_MINUTE,
        "links": {},
    }


@lru_cache(maxsize=1)
def counterfactuals() -> dict:
    """B9 — {july1_centrepiece, all_links, meta}."""
    return _read_json(config.COUNTERFACTUAL_RESULTS_JSON) or {
        "july1_centrepiece": None,
        "all_links": [],
        "meta": {},
    }


@lru_cache(maxsize=1)
def model_metrics() -> list[dict]:
    """B3/B4 — per-model metrics rows."""
    path = config.MODEL_METRICS_CSV
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


@lru_cache(maxsize=1)
def model_comparison() -> list[dict]:
    """B4 — operating-threshold comparison rows (best model selection)."""
    path = config.MODEL_COMPARISON_CSV
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


@lru_cache(maxsize=1)
def best_model_meta() -> Optional[dict]:
    """B4 — chosen model name + metrics, if exported."""
    return _read_json(config.BEST_MODEL_META_JSON)


@lru_cache(maxsize=1)
def cascade_events() -> pd.DataFrame:
    """B8 — validated cascade propagation events (empty frame if absent)."""
    path = config.CASCADE_EVENTS_CSV
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def cascade_details() -> list[dict]:
    """B8 — cascade events with the downstream_json parsed into a list.

    Each event: {day_number, minute_of_day, source_link, source_state,
    n_downstream, n_validated, validation_rate, downstream: [{link_id,
    lag_minutes, predicted_state, actual_state, validated}, ...]}.
    The frontend animates a pulse from source_link to each downstream link
    arriving after its lag_minutes.
    """
    import json as _json
    df = cascade_events()
    if df.empty:
        return []
    out: list[dict] = []
    for r in df.itertuples(index=False):
        try:
            downstream = _json.loads(r.downstream_json)
        except Exception:
            downstream = []
        out.append({
            "day_number": int(r.day_number),
            "minute_of_day": int(r.minute_of_day),
            "source_link": int(r.source_link),
            "source_state": r.source_state,
            "n_downstream": int(r.n_downstream),
            "n_validated": int(r.n_validated),
            "validation_rate": float(r.validation_rate),
            "downstream": downstream,
        })
    return out


# --------------------------------------------------------------------------- #
# Features parquet — loaded lazily and kept in memory (used for live snapshot).
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def features() -> Optional[pd.DataFrame]:
    """B2 — full feature frame. None if the parquet is absent."""
    path = config.FEATURES_PARQUET
    if not path.exists():
        return None
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _timeline_indexed() -> Optional[pd.DataFrame]:
    """features.parquet reduced to the timeline columns, state precomputed,
    indexed by (day, minute) for fast frame slicing. Built once, ~0.5s."""
    f = features()
    if f is None:
        return None
    bands = config.ROAD_STATE_BANDS
    cuts = [lo for _, lo in bands]
    names = [name for name, _ in bands]

    def to_state(h):
        for name, lo in bands:
            if h >= lo:
                return name
        return names[-1]

    cols = ["LINK_ID", "day_number", "minute_of_day",
            "road_health_score", "mean_queue_s", "mean_occup"]
    sub = f[cols].copy()
    # vectorized state via cut on the descending bands
    sub["state"] = sub["road_health_score"].map(to_state)
    sub = sub.set_index(["day_number", "minute_of_day"]).sort_index()
    return sub


def timeline_axis() -> Optional[dict]:
    """Days + minute ticks available to the scrubber."""
    idx = _timeline_indexed()
    if idx is None:
        return None
    days = sorted({int(d) for d, _ in idx.index})
    minutes = sorted({int(m) for _, m in idx.index})
    return {"days": days, "minutes": minutes, "interval_minutes": config.INTERVAL_MINUTES}


def timeline_frame(day: int, minute: int) -> Optional[list[dict]]:
    """All 66 links' state at one (day, minute). None if parquet absent,
    empty list if that exact frame has no rows."""
    idx = _timeline_indexed()
    if idx is None:
        return None
    try:
        g = idx.loc[(day, minute)]
    except KeyError:
        return []
    if isinstance(g, pd.Series):  # single row edge case
        g = g.to_frame().T
    return [
        {
            "link_id": int(r.LINK_ID),
            "state": r.state,
            "health": round(float(r.road_health_score), 2),
            "queue_s": round(float(r.mean_queue_s), 1),
            "occup": round(float(r.mean_occup), 4),
        }
        for r in g.itertuples(index=False)
    ]


# --------------------------------------------------------------------------- #
# Derived / convenience views
# --------------------------------------------------------------------------- #

def link_ids() -> list[int]:
    """All known link ids, preferring the causal graph node set."""
    nodes = causal_graph().get("nodes") or []
    if nodes:
        return sorted(int(n) for n in nodes)
    return sorted(archetypes().keys())


def archetype_for(link_id: int) -> Optional[str]:
    rec = archetypes().get(link_id)
    return rec.get("archetype") if rec else None


def archetype_map() -> dict[int, str]:
    """LINK_ID -> archetype name (for the engine / LLM context)."""
    return {lid: rec["archetype"] for lid, rec in archetypes().items()}


def status() -> dict[str, bool]:
    """Which artifacts are present — surfaced by the /health endpoint."""
    return {
        "features_parquet": config.FEATURES_PARQUET.exists(),
        "archetypes": config.ROAD_ARCHETYPES_JSON.exists(),
        "causal_graph": config.CAUSAL_GRAPH_JSON.exists(),
        "ecosystem_state": config.ECOSYSTEM_STATE_JSON.exists(),
        "counterfactuals": config.COUNTERFACTUAL_RESULTS_JSON.exists(),
        "model_metrics": config.MODEL_METRICS_CSV.exists(),
        "best_model": config.BEST_MODEL_PKL.exists(),
        "cascade_events": config.CASCADE_EVENTS_CSV.exists(),
    }


def clear_cache() -> None:
    """Drop all cached artifacts (used by tests and a manual reload endpoint)."""
    for fn in (
        archetypes, causal_graph, ecosystem_state, counterfactuals,
        model_metrics, model_comparison, best_model_meta, cascade_events,
        cascade_details, _timeline_indexed, features,
    ):
        fn.cache_clear()


# --------------------------------------------------------------------------- #
# SHAP (B5)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def shap_translations() -> Optional[dict]:
    """Load reports/shap/translations.json (precomputed for Link 36 + 37)."""
    p = config.REPORTS_DIR / "shap" / "translations.json"
    return _read_json(p)


@lru_cache(maxsize=1)
def shap_gate() -> Optional[dict]:
    """Load reports/shap/b5_gate.json (global sample stats + plot list)."""
    p = config.REPORTS_DIR / "shap" / "b5_gate.json"
    return _read_json(p)

"""B11 — read-only artifact routers.

Each endpoint serves a precomputed artifact (B5-B9 + B3/B4) in the wire shape
the frontend consumes. No recomputation, no model inference here — that lives
in routers_live.py.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from api import store
from api.schemas import (
    ArchetypeRecord,
    CausalGraphResponse,
    CounterfactualResponse,
    EcosystemStateResponse,
    ModelMetricsResponse,
)
from llm.context import _ARCHETYPE_DESCRIPTIONS

router = APIRouter()


# --------------------------------------------------------------------------- #
# Network — links + archetypes (B7)
# --------------------------------------------------------------------------- #

@router.get("/links", tags=["network"])
def list_links() -> list[int]:
    """All road link ids in the network."""
    return store.link_ids()


@router.get("/archetypes", tags=["network"], response_model=list[ArchetypeRecord])
def list_archetypes() -> list[ArchetypeRecord]:
    """B7 Personality Atlas — archetype for every link, with description."""
    out: list[ArchetypeRecord] = []
    for lid, rec in sorted(store.archetypes().items()):
        meta = _ARCHETYPE_DESCRIPTIONS.get(rec["archetype"], {})
        out.append(
            ArchetypeRecord(
                link_id=lid,
                archetype=rec.get("archetype"),
                confidence=rec.get("confidence"),
                stability_score=rec.get("stability_score"),
                stable=rec.get("stable"),
                description=meta.get("description"),
                policy_class=meta.get("policy_class"),
            )
        )
    return out


@router.get("/archetypes/{link_id}", tags=["network"], response_model=ArchetypeRecord)
def get_archetype(link_id: int) -> ArchetypeRecord:
    rec = store.archetypes().get(link_id)
    if rec is None:
        raise HTTPException(404, f"No archetype for link {link_id}")
    meta = _ARCHETYPE_DESCRIPTIONS.get(rec["archetype"], {})
    return ArchetypeRecord(
        link_id=link_id,
        archetype=rec.get("archetype"),
        confidence=rec.get("confidence"),
        stability_score=rec.get("stability_score"),
        stable=rec.get("stable"),
        description=meta.get("description"),
        policy_class=meta.get("policy_class"),
    )


# --------------------------------------------------------------------------- #
# ECHO B — causal graph + ecosystem state + cascades (B8)
# --------------------------------------------------------------------------- #

@router.get("/echo/causal-graph", tags=["echo"], response_model=CausalGraphResponse)
def get_causal_graph() -> CausalGraphResponse:
    """B8 — the deseasonalized-residual causal highway graph (three.js edges)."""
    g = store.causal_graph()
    nodes = [int(n) for n in g.get("nodes", [])]
    edges = g.get("edges", [])
    return CausalGraphResponse(
        nodes=nodes, edges=edges, n_nodes=len(nodes), n_edges=len(edges),
    )


@router.get("/echo/ecosystem-state", tags=["echo"], response_model=EcosystemStateResponse)
def get_ecosystem_state() -> EcosystemStateResponse:
    """B8 — per-link metabolic state + causal regime at the demo snapshot."""
    es = store.ecosystem_state()
    arch = store.archetype_map()
    links = [
        {
            "link_id": int(lid),
            "state": rec["state"],
            "regime": rec["regime"],
            "road_health_score": rec["road_health_score"],
            "archetype": arch.get(int(lid)),
        }
        for lid, rec in es.get("links", {}).items()
    ]
    links.sort(key=lambda r: r["link_id"])
    return EcosystemStateResponse(
        day_number=es.get("day_number", config.API_DEMO_DAY),
        minute_of_day=es.get("minute_of_day", config.API_DEMO_MINUTE),
        links=links,
    )


@router.get("/echo/cascades", tags=["echo"])
def get_cascades(limit: int = 50) -> list[dict]:
    """B8 — validated cascade propagation events (animate Link 36->16 lag)."""
    df = store.cascade_events()
    if df.empty:
        return []
    return df.head(limit).to_dict(orient="records")


@router.get("/echo/cascades/detailed", tags=["echo"])
def get_cascades_detailed(source_link: int | None = None, day: int | None = None) -> list[dict]:
    """B8 — cascade events with parsed downstream propagation (source->targets+lags).

    Optionally filter by source_link and/or day. Drives the Stage 2 cascade
    animation: a pulse fires from source_link and reaches each downstream link
    after its lag_minutes.
    """
    events = store.cascade_details()
    if source_link is not None:
        events = [e for e in events if e["source_link"] == source_link]
    if day is not None:
        events = [e for e in events if e["day_number"] == day]
    return events


@router.get("/echo/timeline", tags=["echo"])
def get_timeline_frame(
    day: int = config.API_DEMO_DAY,
    minute: int = config.API_DEMO_MINUTE,
) -> dict:
    """One scrubber frame: every link's metabolic state at (day, minute).

    503 if features.parquet is absent. 404 if the exact frame has no data.
    """
    frame = store.timeline_frame(day, minute)
    if frame is None:
        raise HTTPException(
            503, "features.parquet not found. Run `python scripts/02_features.py` first.",
        )
    if not frame:
        raise HTTPException(404, f"No timeline frame for day={day}, minute={minute}")
    return {"day_number": day, "minute_of_day": minute, "links": frame}


@router.get("/echo/timeline/axis", tags=["echo"])
def get_timeline_axis() -> dict:
    """The scrubber's available days + minute ticks (for building the slider)."""
    axis = store.timeline_axis()
    if axis is None:
        raise HTTPException(
            503, "features.parquet not found. Run `python scripts/02_features.py` first.",
        )
    return axis


# --------------------------------------------------------------------------- #
# ECHO C — counterfactual (B9)
# --------------------------------------------------------------------------- #

@router.get("/echo/counterfactual", tags=["echo"], response_model=CounterfactualResponse)
def get_counterfactual() -> CounterfactualResponse:
    """B9 — full counterfactual results incl. the July 1 Link 36 centrepiece."""
    cf = store.counterfactuals()
    return CounterfactualResponse(
        july1_centrepiece=cf.get("july1_centrepiece"),
        all_links=cf.get("all_links", []),
        meta=cf.get("meta", {}),
    )


@router.get("/echo/counterfactual/{link_id}", tags=["echo"])
def get_counterfactual_link(link_id: int) -> dict:
    cf = store.counterfactuals()
    centre = cf.get("july1_centrepiece")
    if centre and int(centre.get("link_id", -1)) == link_id:
        return centre
    for rec in cf.get("all_links", []):
        if int(rec.get("link_id", -1)) == link_id:
            return rec
    raise HTTPException(404, f"No counterfactual result for link {link_id}")


# --------------------------------------------------------------------------- #
# Models — B3/B4 metrics
# --------------------------------------------------------------------------- #

@router.get("/models/metrics", tags=["models"], response_model=ModelMetricsResponse)
def get_model_metrics() -> ModelMetricsResponse:
    """B3/B4 — per-model metrics + operating-threshold comparison + best model."""
    return ModelMetricsResponse(
        metrics=store.model_metrics(),
        comparison=store.model_comparison(),
        best_model=store.best_model_meta(),
    )


# --------------------------------------------------------------------------- #
# SHAP explainability (B5)
# --------------------------------------------------------------------------- #

@router.get("/shap/summary", tags=["shap"])
def get_shap_summary() -> dict:
    """B5 — global SHAP feature importance + PNG plot URLs."""
    gate = store.shap_gate()
    translations = store.shap_translations()
    if gate is None:
        raise HTTPException(503, "SHAP artifacts not found. Run `python scripts/b5_shap.py` first.")

    computed_links: list[int] = []
    if translations:
        for rec in translations.values():
            meta = rec.get("meta", {})
            if "link_id" in meta:
                computed_links.append(int(meta["link_id"]))

    return {
        "global_sample_rows": gate.get("global_sample_rows"),
        "global_sample_pos_rate": gate.get("global_sample_pos_rate"),
        "plots": {
            "beeswarm": "/reports/shap/01_beeswarm.png",
            "importance_bar": "/reports/shap/02_importance_bar.png",
            "dependence_hour": "/reports/shap/05_dependence_hour.png",
            "dependence_occup": "/reports/shap/06_dependence_mean_occup.png",
        },
        "computed_links": sorted(computed_links),
    }


@router.get("/shap/link/{link_id}", tags=["shap"])
def get_shap_link(link_id: int) -> dict:
    """B5 — per-link SHAP waterfall: top-3 features + plain-English.

    Precomputed for Links 36 and 37. Returns 404 for others (on-demand
    SHAP requires the trained model pkl on disk).
    """
    translations = store.shap_translations()
    if translations is None:
        raise HTTPException(503, "SHAP translations not found. Run `python scripts/b5_shap.py` first.")

    key = f"link_{link_id}"
    rec = translations.get(key)
    if rec is None:
        raise HTTPException(
            404,
            f"No precomputed SHAP waterfall for link {link_id}. "
            f"Precomputed: {sorted(translations.keys())}",
        )

    waterfall_png = None
    gate = store.shap_gate()
    if gate:
        for plot_file in gate.get("plots_produced", []):
            if f"link{link_id}" in plot_file:
                waterfall_png = f"/reports/shap/{plot_file}"
                break

    return {**rec, "waterfall_png": waterfall_png}

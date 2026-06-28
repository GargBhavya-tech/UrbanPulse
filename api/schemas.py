"""B11 — API response schemas.

Pydantic models that define the backend<->frontend contract. The frontend
(React + three.js + p5.js) is built against exactly these shapes, so they are
the source of truth for the wire format. Kept deliberately flat and JSON-native.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Health / meta
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    artifacts: dict[str, bool]


# --------------------------------------------------------------------------- #
# Network / links
# --------------------------------------------------------------------------- #

class ArchetypeRecord(BaseModel):
    link_id: int
    archetype: Optional[str] = None
    confidence: Optional[float] = None
    stability_score: Optional[float] = None
    stable: Optional[bool] = None
    description: Optional[str] = None
    policy_class: Optional[str] = None


class CausalEdge(BaseModel):
    source: int
    target: int
    lag_minutes: int
    correlation_strength: float
    discovery_method: str


class CausalGraphResponse(BaseModel):
    nodes: list[int]
    edges: list[CausalEdge]
    n_nodes: int
    n_edges: int


class LinkState(BaseModel):
    link_id: int
    state: str
    regime: str
    road_health_score: float
    archetype: Optional[str] = None


class EcosystemStateResponse(BaseModel):
    day_number: int
    minute_of_day: int
    links: list[LinkState]


# --------------------------------------------------------------------------- #
# Snapshot (live engine over the best model, when present)
# --------------------------------------------------------------------------- #

class SnapshotRecord(BaseModel):
    link_id: int
    archetype: Optional[str] = None
    health_score: float
    state: str
    risk_score: float
    congestion_prob: float
    critical: bool
    mean_queue_s: float
    alerts: list[dict]
    recommendations: list[dict]


class SnapshotResponse(BaseModel):
    timestamp: dict
    hotspot_ranking: list[int]
    links: list[SnapshotRecord]


# --------------------------------------------------------------------------- #
# Counterfactual
# --------------------------------------------------------------------------- #

class CounterfactualCentrepiece(BaseModel):
    link_id: int
    intervention: str
    estimation_mode: str
    observed_queue_s: float
    counterfactual_queue_s: float
    queue_reduction_s: float
    queue_reduction_pct: float
    vehicle_hours_saved: float
    cascade_prevented: bool
    causal_mechanism: Optional[str] = None

    model_config = {"extra": "allow"}  # keep all extra B9 fields


class CounterfactualResponse(BaseModel):
    july1_centrepiece: Optional[dict] = None
    all_links: list[dict] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class ModelMetricsResponse(BaseModel):
    metrics: list[dict]
    comparison: list[dict]
    best_model: Optional[dict] = None


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

class LLMRequest(BaseModel):
    link_id: int = 36
    day_number: int = 1
    minute_of_day: int = 585
    output_type: str = Field(
        default="planner_briefing",
        description="citizen_advice | planner_briefing | cascade_alert | "
                    "counterfactual_summary | traffic_summary",
    )


class LLMResponse(BaseModel):
    output_type: str
    text: str
    backend: str
    violations: list[str] = Field(default_factory=list)


class QARequest(BaseModel):
    question: str
    link_id: int = 36
    day_number: int = 1
    minute_of_day: int = 585
    audience: str = "citizen"


class QAResponse(BaseModel):
    question: str
    answer: str
    audience: str
    backend: str
    violations: list[str] = Field(default_factory=list)

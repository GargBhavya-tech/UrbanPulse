"""B11 — LLM layer router.

Exposes the B10 LLM Intelligence Layer over HTTP: the five structured output
types plus free-text Q&A. Context is assembled from artifacts via
llm.context.from_artifacts, so the LLM only ever sees an UrbanPulseContext
(Bible §8 grounding rule), never raw sensor data.

Backend is the config default (template / flan_t5 / gemini). The layer is
constructed lazily and cached, so the first call pays any model-load cost.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from api.schemas import LLMRequest, LLMResponse, QARequest, QAResponse
from llm.context import from_artifacts
from llm.layer import LLMLayer

router = APIRouter()

_LAYER_CACHE: dict[str, LLMLayer] = {}

_GENERATORS = {
    "citizen_advice": "generate_citizen_advice",
    "planner_briefing": "generate_planner_briefing",
    "cascade_alert": "generate_cascade_alert",
    "counterfactual_summary": "generate_counterfactual_summary",
    "traffic_summary": "generate_traffic_summary",
}


def _layer() -> LLMLayer:
    if "layer" not in _LAYER_CACHE:
        _LAYER_CACHE["layer"] = LLMLayer()
    return _LAYER_CACHE["layer"]


@router.get("/llm/backends", tags=["llm"])
def llm_backends() -> dict:
    """Which backend is active and what the options are."""
    return {
        "active_default": config.LLM_DEFAULT_BACKEND,
        "options": ["template", "flan_t5", "gemini"],
        "loaded_backend": _layer().backend_name,
    }


@router.post("/llm/generate", tags=["llm"], response_model=LLMResponse)
def llm_generate(req: LLMRequest) -> LLMResponse:
    """Generate one structured output type for a link/time context."""
    method_name = _GENERATORS.get(req.output_type)
    if method_name is None:
        raise HTTPException(
            422,
            f"Unknown output_type '{req.output_type}'. "
            f"Valid: {sorted(_GENERATORS)}",
        )
    ctx = from_artifacts(
        link_id=req.link_id,
        day_number=req.day_number,
        minute_of_day=req.minute_of_day,
    )
    layer = _layer()
    text = getattr(layer, method_name)(ctx)
    return LLMResponse(
        output_type=req.output_type,
        text=text,
        backend=layer.backend_name,
        violations=layer.last_violations,
    )


@router.post("/llm/ask", tags=["llm"], response_model=QAResponse)
def llm_ask(req: QARequest) -> QAResponse:
    """Free-text grounded Q&A about a road/time (citizen or planner audience)."""
    if req.audience not in ("citizen", "planner"):
        raise HTTPException(422, "audience must be 'citizen' or 'planner'")
    ctx = from_artifacts(
        link_id=req.link_id,
        day_number=req.day_number,
        minute_of_day=req.minute_of_day,
    )
    layer = _layer()
    answer = layer.answer_question(ctx, req.question, audience=req.audience)
    return QAResponse(
        question=req.question,
        answer=answer,
        audience=req.audience,
        backend=layer.backend_name,
        violations=layer.last_violations,
    )

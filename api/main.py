"""B11 — UrbanPulse FastAPI serving layer (DECISION_MAP #4/#5).

The single read-only HTTP data-access path between the precomputed pipeline
artifacts (B1-B10) and the Phase 2 frontend (React + three.js + p5.js).

Run::

    uvicorn api.main:app --reload
    # or
    python scripts/11_serve.py

Then open http://127.0.0.1:8000/docs for the interactive API.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from api import store
from api.schemas import HealthResponse
from api.routers_artifacts import router as artifacts_router
from api.routers_live import router as live_router
from api.routers_llm import router as llm_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        description=(
            "Read-only API over the UrbanPulse pipeline artifacts: ML metrics, "
            "Personality Atlas (B7), causal graph + cascades (B8), "
            "counterfactuals (B9), the live B6 engine snapshot, and the B10 LLM "
            "layer. Serves the React + three.js + p5.js frontend."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.API_CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "name": config.API_TITLE,
            "version": config.API_VERSION,
            "docs": "/docs",
            "endpoints": [
                "/health", "/links", "/archetypes", "/archetypes/{link_id}",
                "/echo/causal-graph", "/echo/ecosystem-state", "/echo/cascades",
                "/echo/cascades/detailed", "/echo/timeline", "/echo/timeline/axis",
                "/echo/counterfactual", "/echo/counterfactual/{link_id}",
                "/models/metrics", "/snapshot",
                "/llm/backends", "/llm/generate", "/llm/ask",
            ],
        }

    @app.get("/health", tags=["meta"], response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", version=config.API_VERSION, artifacts=store.status(),
        )

    @app.post("/admin/reload", tags=["meta"])
    def reload_artifacts() -> dict:
        """Drop cached artifacts so a fresh pipeline run is picked up."""
        store.clear_cache()
        return {"reloaded": True, "artifacts": store.status()}

    app.include_router(artifacts_router)
    app.include_router(live_router)
    app.include_router(llm_router)

    # Serve precomputed report PNGs (SHAP plots, model comparison charts, etc.)
    # at /reports/** so the frontend can <img src="/api/reports/shap/...png">.
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    if reports_dir.exists():
        app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

    return app


app = create_app()

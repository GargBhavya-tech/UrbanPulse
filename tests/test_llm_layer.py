"""Tests for B10 LLM Intelligence Layer (Bible §8).

All tests run fully offline using TemplateClient -- no API key, no model download.
The test suite validates:
  - GroundingValidator behavior
  - Prompt template structure
  - Output type contracts (length, content, forbidden terms)
  - Context assembly (from_artifacts with real pipeline outputs)
  - Client factory and fallback behavior
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm.client import LLMClient, TemplateClient, GeminiClient
from llm.context import (
    UrbanPulseContext, CFResult, CascadeEvent, SHAPFeature, from_artifacts
)
from llm.grounding import GroundingValidator, validated_complete
from llm.layer import LLMLayer
from llm import prompts


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _make_ctx(
    link_id: int = 36,
    state: str = "Saturated",
    health: float = 25.7,
    archetype: str = "Chronic",
    cascade: bool = False,
    with_cf: bool = True,
    queue_s: float = 892.0,
) -> UrbanPulseContext:
    ctx = UrbanPulseContext(
        link_id=link_id,
        day_number=1,
        minute_of_day=585,
        hour=9,
        day_of_week=0,
        road_health_score=health,
        congestion_risk_score=85.0,
        metabolic_state=state,
        congestion_prob=0.91,
        predicted_queue_s=queue_s,
        total_vehs=950,
        is_am_peak=True,
        recommendations=["Review and extend green phase", "Flag for infrastructure review"],
        archetype=archetype,
        archetype_description="maintains elevated congestion 24/7",
        archetype_policy_class="infrastructure audit, capacity redesign",
        stability_score=0.82,
        top_shap=[
            SHAPFeature("mean_occup", 0.38, "Road occupancy near maximum (72%)", "increases"),
            SHAPFeature("hour", 0.22, "Measurement falls in peak hours (hour 9)", "increases"),
            SHAPFeature("LINK_ID", 0.18, "Road 36 has structural tendency toward congestion", "increases"),
        ],
        cascade_active=cascade,
        cascade_event=CascadeEvent(36, [16, 30], 8, 2) if cascade else None,
        counterfactual=CFResult(
            intervention_description="Activate Lane 6 at 09:30 AM",
            observed_queue_s=892.0,
            counterfactual_queue_s=391.0,
            queue_reduction_pct=56.2,
            vehicle_hours_saved=1207.0,
            cascade_prevented=False,
            estimation_mode="policy_simulation",
        ) if with_cf else None,
        historical_baseline_queue_s=320.0,
    )
    return ctx


# --------------------------------------------------------------------------- #
# 1. Client factory tests
# --------------------------------------------------------------------------- #

def test_template_client_factory() -> None:
    """LLMClient.create('template') returns TemplateClient."""
    client = LLMClient.create("template")
    assert isinstance(client, TemplateClient)


def test_template_client_returns_string() -> None:
    """TemplateClient.complete() returns a non-empty string."""
    client = TemplateClient()
    ctx = _make_ctx()
    prompt = prompts.build_citizen_advice_prompt(ctx)
    result = client.complete(prompt)
    assert isinstance(result, str)
    assert len(result) > 0


def test_gemini_client_falls_back_without_key(monkeypatch) -> None:
    """GeminiClient without GEMINI_API_KEY falls back to TemplateClient."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient()
    ctx = _make_ctx()
    prompt = prompts.build_citizen_advice_prompt(ctx)
    result = client.complete(prompt)
    assert isinstance(result, str) and len(result) > 0


def test_invalid_backend_raises() -> None:
    """Unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        LLMClient.create("bert")


# --------------------------------------------------------------------------- #
# 2. Grounding validator tests
# --------------------------------------------------------------------------- #

def test_grounding_blocks_forbidden_citizen_term() -> None:
    """GroundingValidator must flag 'SHAP' in citizen output."""
    ctx = _make_ctx()
    validator = GroundingValidator()
    result = validator.validate(
        "The SHAP value shows high risk.", audience="citizen", ctx=ctx
    )
    assert not result.passed
    assert any("SHAP" in v for v in result.violations)


def test_grounding_passes_clean_citizen_output() -> None:
    """Plain citizen output with no forbidden terms passes."""
    ctx = _make_ctx()
    validator = GroundingValidator()
    result = validator.validate(
        "Road 36 has heavy delays. Expect a 14-minute wait. Leave by 7:45 AM.",
        audience="citizen",
        ctx=ctx,
    )
    assert result.passed, result.violations


def test_grounding_blocks_missing_estimation_language() -> None:
    """Counterfactual output without hedging must fail."""
    ctx = _make_ctx()
    validator = GroundingValidator()
    result = validator.validate(
        "Activating Lane 6 will reduce queue by 56%. The cascade will be prevented.",
        audience="planner",
        ctx=ctx,
        output_type="counterfactual_summary",
    )
    assert not result.passed
    assert any("estimation language" in v for v in result.violations)


def test_grounding_passes_hedged_counterfactual() -> None:
    """Properly hedged counterfactual passes estimation check."""
    ctx = _make_ctx()
    validator = GroundingValidator()
    result = validator.validate(
        "Analysis estimates that activating Lane 6 would reduce queue delay by approximately 56%.",
        audience="planner",
        ctx=ctx,
        output_type="counterfactual_summary",
    )
    # May still have number violations but estimation check passes
    estimation_violations = [v for v in result.violations if "estimation" in v.lower()]
    assert len(estimation_violations) == 0


# --------------------------------------------------------------------------- #
# 3. LLMLayer output contract tests
# --------------------------------------------------------------------------- #

def test_citizen_advice_no_forbidden_terms() -> None:
    """Citizen advice must not contain any forbidden terms."""
    ctx = _make_ctx()
    layer = LLMLayer(backend="template")
    advice = layer.generate_citizen_advice(ctx)
    assert isinstance(advice, str) and len(advice) > 0
    advice_lower = advice.lower()
    for term in config.LLM_CITIZEN_FORBIDDEN:
        assert term.lower() not in advice_lower, (
            f"Forbidden term '{term}' found in citizen advice: {advice}"
        )


def test_planner_briefing_contains_key_fields() -> None:
    """Planner briefing must contain archetype, health score, and state."""
    ctx = _make_ctx()
    layer = LLMLayer(backend="template")
    briefing = layer.generate_planner_briefing(ctx)
    assert "36" in briefing            # link id
    assert "Chronic" in briefing       # archetype
    assert any(c in briefing for c in ["Saturated", "saturated"])  # state


def test_cascade_alert_empty_when_no_cascade() -> None:
    """cascade_alert returns empty string when no cascade is active."""
    ctx = _make_ctx(cascade=False)
    layer = LLMLayer(backend="template")
    alert = layer.generate_cascade_alert(ctx)
    assert alert == ""


def test_cascade_alert_non_empty_with_cascade() -> None:
    """cascade_alert returns content when cascade is active."""
    ctx = _make_ctx(cascade=True)
    layer = LLMLayer(backend="template")
    alert = layer.generate_cascade_alert(ctx)
    assert len(alert) > 10
    # Must mention the source road
    assert "36" in alert


def test_counterfactual_summary_uses_estimated_language() -> None:
    """Counterfactual summary must contain estimation language."""
    ctx = _make_ctx(with_cf=True)
    layer = LLMLayer(backend="template")
    summary = layer.generate_counterfactual_summary(ctx)
    assert isinstance(summary, str) and len(summary) > 10
    hedge_words = {"estimated", "estimate", "approximately", "suggests", "analysis"}
    assert any(w in summary.lower() for w in hedge_words), (
        f"No estimation language found in: {summary}"
    )


def test_counterfactual_empty_without_cf_result() -> None:
    """counterfactual_summary returns empty string when no CF result in context."""
    ctx = _make_ctx(with_cf=False)
    layer = LLMLayer(backend="template")
    summary = layer.generate_counterfactual_summary(ctx)
    assert summary == ""


def test_traffic_summary_non_empty() -> None:
    """traffic_summary produces a non-empty string."""
    ctx = _make_ctx()
    layer = LLMLayer(backend="template")
    summary = layer.generate_traffic_summary(ctx, n_critical=17, worst_link=36, worst_health=25.7)
    assert len(summary) > 10


def test_answer_question_july1_citizen() -> None:
    """Citizen Q&A about July 1 must not contain forbidden terms."""
    ctx = _make_ctx(with_cf=True)
    layer = LLMLayer(backend="template")
    answer = layer.answer_question(
        ctx,
        "What could have prevented the July 1 disaster?",
        audience="citizen",
    )
    assert isinstance(answer, str) and len(answer) > 10
    answer_lower = answer.lower()
    for term in config.LLM_CITIZEN_FORBIDDEN:
        assert term.lower() not in answer_lower, (
            f"Forbidden term '{term}' found in citizen Q&A: {answer}"
        )


def test_answer_question_july1_planner_references_cf() -> None:
    """Planner Q&A about July 1 should reference the counterfactual reduction."""
    ctx = _make_ctx(with_cf=True)
    layer = LLMLayer(backend="template")
    answer = layer.answer_question(
        ctx,
        "What could have prevented the July 1 disaster?",
        audience="planner",
    )
    # Should mention reduction percentage or vehicle-hours
    assert any(keyword in answer for keyword in ["56", "1207", "Lane 6", "estimated"]), (
        f"Planner answer missing CF details: {answer}"
    )


def test_generate_all_returns_all_keys() -> None:
    """generate_all() must return all 7 expected keys."""
    ctx = _make_ctx(cascade=True, with_cf=True)
    layer = LLMLayer(backend="template")
    outputs = layer.generate_all(ctx)
    expected_keys = {
        "citizen_advice", "planner_briefing", "cascade_alert",
        "counterfactual_summary", "traffic_summary",
        "answer_question_citizen", "answer_question_planner",
    }
    assert expected_keys <= set(outputs.keys()), (
        f"Missing keys: {expected_keys - set(outputs.keys())}"
    )
    # All non-cascade outputs are non-empty
    for k in expected_keys - {"cascade_alert"}:
        assert outputs[k], f"Output '{k}' is empty"


# --------------------------------------------------------------------------- #
# 4. Context assembly test (requires real artifacts)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not config.FEATURES_PARQUET.exists(),
    reason="Pipeline artifacts not built -- run pipeline first",
)
def test_from_artifacts_loads_context() -> None:
    """from_artifacts() assembles a valid context from real pipeline outputs."""
    ctx = from_artifacts(link_id=36, day_number=1, minute_of_day=585)
    assert ctx.link_id == 36
    assert ctx.road_health_score > 0
    assert ctx.metabolic_state in ("Healthy", "Stressed", "Saturated", "Collapsed")
    # Archetype must be loaded if B7 ran
    if config.ROAD_ARCHETYPES_JSON.exists():
        assert ctx.archetype is not None
    # Counterfactual must be loaded if B9 ran
    if config.COUNTERFACTUAL_RESULTS_JSON.exists():
        assert ctx.counterfactual is not None
        assert ctx.counterfactual.queue_reduction_pct > 0


@pytest.mark.skipif(
    not config.FEATURES_PARQUET.exists(),
    reason="Pipeline artifacts not built -- run pipeline first",
)
def test_full_llm_pipeline_runs_without_error() -> None:
    """End-to-end: load real context, generate all outputs, validate grounding."""
    ctx = from_artifacts(link_id=36, day_number=1, minute_of_day=585)
    layer = LLMLayer(backend="template")
    outputs = layer.generate_all(ctx)

    # Citizen advice must pass full grounding validation
    validator = GroundingValidator()
    result = validator.validate(
        outputs["citizen_advice"],
        audience="citizen",
        ctx=ctx,
        output_type="citizen_advice",
    )
    assert result.passed, f"Grounding failed: {result.violations}"

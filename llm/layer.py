"""B10 LLM Intelligence Layer -- public API.

This module is the only public entry point for the LLM layer.
All callers import from here, not from client/prompts/grounding.

Bible §8 governs what this layer does:
  - Translates structured ML/ECHO outputs into natural language.
  - Never makes predictions.
  - Never accesses raw sensor data.
  - Only sees UrbanPulseContext.

Usage::

    from llm.layer import LLMLayer
    from llm.context import from_artifacts

    ctx = from_artifacts(link_id=36)
    layer = LLMLayer()                    # uses config default backend
    layer = LLMLayer(backend="template")  # deterministic, no model needed

    print(layer.generate_citizen_advice(ctx))
    print(layer.generate_planner_briefing(ctx))
    print(layer.generate_cascade_alert(ctx))
    print(layer.generate_counterfactual_summary(ctx))
    print(layer.generate_traffic_summary(ctx))
    print(layer.answer_question(ctx, "What could have prevented July 1?"))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm.client import LLMClient
from llm.context import UrbanPulseContext
from llm.grounding import ValidationResult, validated_complete
from llm import prompts


# --------------------------------------------------------------------------- #
# LLMLayer
# --------------------------------------------------------------------------- #

class LLMLayer:
    """The UrbanPulse LLM Intelligence Layer.

    Args:
        backend: One of "template", "flan_t5", "gemini".
                 Defaults to config.LLM_DEFAULT_BACKEND.
        client:  Provide a pre-built LLMClient to skip backend auto-detection.
                 Useful for testing.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        self._client = client or LLMClient.create(backend)
        self._last_violations: list[str] = []

    @property
    def backend_name(self) -> str:
        return type(self._client).__name__

    @property
    def last_violations(self) -> list[str]:
        """Grounding violations from the most recent call (empty if passed)."""
        return self._last_violations

    # ----------------------------------------------------------------------- #
    # Output type 1 — Citizen Travel Advice (Bible §8.2)
    # ----------------------------------------------------------------------- #

    def generate_citizen_advice(self, ctx: UrbanPulseContext) -> str:
        """Max 3 sentences.  No technical jargon.  One actionable tip.

        Passes grounding validation: no forbidden terms, no hallucinated numbers.
        """
        prompt = prompts.build_citizen_advice_prompt(ctx)
        output, result = validated_complete(
            self._client, prompt,
            audience="citizen", ctx=ctx, output_type="citizen_advice",
        )
        self._last_violations = result.violations
        return output.strip()

    # ----------------------------------------------------------------------- #
    # Output type 2 — Planner Briefing
    # ----------------------------------------------------------------------- #

    def generate_planner_briefing(self, ctx: UrbanPulseContext) -> str:
        """Technical summary: cause, severity, archetype, top recommendation.

        Max 150 words.  May include SHAP, archetype, occupancy.
        """
        prompt = prompts.build_planner_briefing_prompt(ctx)
        output, result = validated_complete(
            self._client, prompt,
            audience="planner", ctx=ctx, output_type="planner_briefing",
        )
        self._last_violations = result.violations
        return output.strip()

    # ----------------------------------------------------------------------- #
    # Output type 3 — Cascade Alert
    # ----------------------------------------------------------------------- #

    def generate_cascade_alert(self, ctx: UrbanPulseContext) -> str:
        """Two-sentence CRITICAL alert: source road, downstream roads, lag.

        Returns empty string if no cascade is active on this context.
        """
        if not ctx.cascade_active and ctx.cascade_event is None:
            return ""
        prompt = prompts.build_cascade_alert_prompt(ctx)
        output, result = validated_complete(
            self._client, prompt,
            audience="planner", ctx=ctx, output_type="cascade_alert",
        )
        self._last_violations = result.violations
        return output.strip()

    # ----------------------------------------------------------------------- #
    # Output type 4 — Counterfactual Summary
    # ----------------------------------------------------------------------- #

    def generate_counterfactual_summary(self, ctx: UrbanPulseContext) -> str:
        """100-word summary of a counterfactual scenario.

        Always uses estimated/hedged language (grounding rule).
        Returns empty string if no counterfactual result is in context.
        """
        if ctx.counterfactual is None:
            return ""
        prompt = prompts.build_counterfactual_summary_prompt(ctx)
        output, result = validated_complete(
            self._client, prompt,
            audience="planner", ctx=ctx, output_type="counterfactual_summary",
        )
        self._last_violations = result.violations
        return output.strip()

    # ----------------------------------------------------------------------- #
    # Output type 5 — Traffic Network Summary
    # ----------------------------------------------------------------------- #

    def generate_traffic_summary(
        self,
        ctx: UrbanPulseContext,
        n_critical: int = 0,
        worst_link: int = 36,
        worst_health: float = 25.0,
    ) -> str:
        """80-word network-wide overview for planners or senior officials."""
        prompt = prompts.build_traffic_summary_prompt(
            ctx, n_critical=n_critical, worst_link=worst_link, worst_health=worst_health
        )
        output, result = validated_complete(
            self._client, prompt,
            audience="planner", ctx=ctx, output_type="traffic_summary",
        )
        self._last_violations = result.violations
        return output.strip()

    # ----------------------------------------------------------------------- #
    # Output type 6 — Interactive Q&A
    # ----------------------------------------------------------------------- #

    def answer_question(
        self,
        ctx: UrbanPulseContext,
        question: str,
        audience: str = "citizen",
    ) -> str:
        """Answer a free-text question grounded in ctx.

        Args:
            ctx: The structured context for this road / time.
            question: Free-text question from the user.
            audience: "citizen" (plain language) or "planner" (technical).
        """
        prompt = prompts.build_qa_prompt(ctx, question, audience=audience)
        output, result = validated_complete(
            self._client, prompt,
            audience=audience, ctx=ctx, output_type="question_answer",
        )
        self._last_violations = result.violations
        return output.strip()

    # ----------------------------------------------------------------------- #
    # Batch: all output types for one context
    # ----------------------------------------------------------------------- #

    def generate_all(
        self,
        ctx: UrbanPulseContext,
        question: str = "What could have prevented the July 1 disaster?",
        n_critical: int = 17,
        worst_link: int = 36,
        worst_health: float = 25.7,
    ) -> dict:
        """Generate all 6 output types and return as a dict.

        Used by the demo script and integration test.
        """
        results = {
            "citizen_advice": self.generate_citizen_advice(ctx),
            "planner_briefing": self.generate_planner_briefing(ctx),
            "cascade_alert": self.generate_cascade_alert(ctx),
            "counterfactual_summary": self.generate_counterfactual_summary(ctx),
            "traffic_summary": self.generate_traffic_summary(
                ctx, n_critical=n_critical, worst_link=worst_link, worst_health=worst_health
            ),
            "answer_question_citizen": self.answer_question(ctx, question, audience="citizen"),
            "answer_question_planner": self.answer_question(ctx, question, audience="planner"),
        }
        return results


# --------------------------------------------------------------------------- #
# Persistence helper
# --------------------------------------------------------------------------- #

def save_outputs(outputs: dict, path: Optional[Path] = None) -> Path:
    """Save all LLM output types to a JSON file."""
    out_path = path or (config.LLM_REPORTS_DIR / "llm_outputs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path

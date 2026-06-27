"""GroundingValidator for UrbanPulse LLM layer.

Enforces Bible §8.3 grounding rules on LLM output:
  1. Forbidden-term check: citizen output must never contain technical jargon.
  2. Number hallucination check: every number in the output must appear in
     the context (within a tolerance) or in the prompt.
  3. Estimation-language check: counterfactual outputs must use hedged language.

Usage::

    validator = GroundingValidator()
    result = validator.validate(output, audience="citizen", ctx=ctx, prompt=prompt)
    if not result.passed:
        print(result.violations)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm.context import UrbanPulseContext


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class ValidationResult:
    passed: bool
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #

class GroundingValidator:
    """Validates LLM output against grounding rules.

    All checks are conservative -- a check only fails if we can prove the
    output contains something forbidden.  Ambiguous cases pass.
    """

    def validate(
        self,
        output: str,
        *,
        audience: str = "planner",
        ctx: UrbanPulseContext,
        prompt: str = "",
        output_type: str = "generic",
    ) -> ValidationResult:
        violations: list[str] = []

        # 1. Forbidden terms (citizen only)
        if audience == "citizen":
            self._check_forbidden_terms(output, violations)

        # 2. Number hallucination check
        self._check_numbers(output, ctx, prompt, violations)

        # 3. Counterfactual must use estimated language
        if output_type == "counterfactual_summary":
            self._check_estimation_language(output, violations)

        # 4. Length sanity check
        self._check_length(output, output_type, violations)

        return ValidationResult(passed=len(violations) == 0, violations=violations)

    # -- individual checks -------------------------------------------------- #

    def _check_forbidden_terms(self, output: str, violations: list[str]) -> None:
        output_lower = output.lower()
        for term in config.LLM_CITIZEN_FORBIDDEN:
            if term.lower() in output_lower:
                violations.append(f"Citizen output contains forbidden term: '{term}'")

    def _check_numbers(
        self, output: str, ctx: UrbanPulseContext, prompt: str, violations: list[str]
    ) -> None:
        """Numbers in LLM output must appear in the context or prompt.

        We only flag numbers that look like statistics (>= 3 digits or
        decimal numbers) and that cannot be traced to known context fields.
        """
        # Build a set of allowed numeric values from context
        allowed: set[float] = set()
        for val in [
            ctx.road_health_score,
            ctx.congestion_prob * 100,
            ctx.predicted_queue_s,
            ctx.predicted_queue_s / 60,  # in minutes
            ctx.historical_baseline_queue_s,
            ctx.congestion_risk_score,
            float(ctx.link_id),
            float(ctx.total_vehs),
            float(ctx.hour),
        ]:
            allowed.add(round(val, 0))
            allowed.add(round(val, 1))
        if ctx.counterfactual:
            cf = ctx.counterfactual
            for v in [
                cf.observed_queue_s, cf.counterfactual_queue_s,
                cf.queue_reduction_pct, cf.vehicle_hours_saved,
                cf.observed_queue_s / 60, cf.counterfactual_queue_s / 60,
            ]:
                allowed.add(round(v, 0))
                allowed.add(round(v, 1))
        if ctx.cascade_event:
            allowed.add(float(ctx.cascade_event.lag_minutes))
            allowed.add(float(ctx.cascade_event.n_downstream))

        # Also accept numbers from the prompt (they're grounded by construction)
        prompt_numbers = {float(m) for m in re.findall(r"\b\d+\.?\d*\b", prompt)}
        allowed.update(prompt_numbers)

        # Extract numbers from output (3+ digit integers or decimals)
        output_numbers = re.findall(r"\b(\d{3,}(?:\.\d+)?|\d+\.\d+)\b", output)
        for num_str in output_numbers:
            num = float(num_str)
            # Allow if within 5% of any allowed value
            if not any(
                abs(num - a) <= max(5.0, 0.05 * abs(a))
                for a in allowed
                if a != 0
            ):
                violations.append(
                    f"Possible hallucinated number: {num_str} (not traceable to context)"
                )

    def _check_estimation_language(self, output: str, violations: list[str]) -> None:
        hedge_words = {"estimated", "estimate", "approximately", "suggests", "analysis", "projected"}
        output_lower = output.lower()
        if not any(w in output_lower for w in hedge_words):
            violations.append(
                "Counterfactual output missing estimation language "
                "(e.g., 'estimated', 'approximately', 'analysis suggests')"
            )

    def _check_length(self, output: str, output_type: str, violations: list[str]) -> None:
        word_count = len(output.split())
        limits: dict[str, int] = {
            "citizen_advice": 60,
            "planner_briefing": 200,
            "cascade_alert": 60,
            "counterfactual_summary": 150,
            "traffic_summary": 120,
            "question_answer": 180,
        }
        limit = limits.get(output_type, 250)
        if word_count > limit:
            violations.append(
                f"Output too long: {word_count} words (limit for {output_type}: {limit})"
            )


# --------------------------------------------------------------------------- #
# Retry wrapper
# --------------------------------------------------------------------------- #

def validated_complete(
    client,
    prompt: str,
    *,
    audience: str = "planner",
    ctx: UrbanPulseContext,
    output_type: str = "generic",
    max_retries: int = 1,
) -> tuple[str, ValidationResult]:
    """Call client.complete() with grounding validation and one retry.

    Returns (output_text, final_validation_result).
    On retry failure, returns the last output with its violations noted.
    """
    validator = GroundingValidator()
    output = client.complete(prompt)

    result = validator.validate(
        output, audience=audience, ctx=ctx, prompt=prompt, output_type=output_type
    )
    if result.passed:
        return output, result

    # One retry with stricter instruction prepended
    if max_retries > 0:
        strict_prefix = (
            "IMPORTANT: Your previous response violated grounding rules. "
            "Violations: " + "; ".join(result.violations) + "\n"
            "Try again, strictly following ALL rules:\n\n"
        )
        output2 = client.complete(strict_prefix + prompt)
        result2 = validator.validate(
            output2, audience=audience, ctx=ctx, prompt=prompt, output_type=output_type
        )
        if result2.passed:
            return output2, result2
        # Return last attempt with violations
        return output2, result2

    return output, result

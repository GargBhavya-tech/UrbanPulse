"""Prompt templates for UrbanPulse LLM layer.

Every prompt function builds a string containing:
  1. A system instruction (role + rules)
  2. A [DATA] key=value block (parsed by TemplateClient; read by real LLMs)
  3. A task instruction

The [DATA] block ensures TemplateClient and real LLMs have the same
structured facts.  Real LLMs may produce richer prose; TemplateClient
reads the block deterministically.

Bible §8.3 grounding rules are enforced at prompt-build time:
  - Citizen prompts explicitly forbid a list of technical terms.
  - Counterfactual prompts always include "estimated" framing.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm.context import UrbanPulseContext


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _state_plain(state: str) -> str:
    """Convert metabolic state to citizen-friendly language."""
    return {
        "Healthy": "clear conditions with no significant delays",
        "Stressed": "moderate congestion with some delays",
        "Saturated": "heavy congestion with significant waiting times",
        "Collapsed": "severe gridlock",
    }.get(state, "some delays")


def _data_block(pairs: dict) -> str:
    lines = ["[DATA]"]
    for k, v in pairs.items():
        lines.append(f"{k}={v}")
    lines.append("[/DATA]")
    return "\n".join(lines)


def _shap_lines(ctx: UrbanPulseContext) -> str:
    if not ctx.top_shap:
        return "No SHAP data available."
    return "\n".join(
        f"  - {f.plain_english} ({'+' if f.shap_value > 0 else ''}{f.shap_value:.3f} SHAP)"
        for f in ctx.top_shap
    )


def _cascade_targets_str(ctx: UrbanPulseContext) -> str:
    if ctx.cascade_event and ctx.cascade_event.target_link_ids:
        roads = ", ".join(f"Road {t}" for t in ctx.cascade_event.target_link_ids[:5])
        return roads
    return "downstream roads"


# --------------------------------------------------------------------------- #
# Citizen Travel Advice (Bible §8.2, max 3 sentences, no jargon)
# --------------------------------------------------------------------------- #

def build_citizen_advice_prompt(ctx: UrbanPulseContext) -> str:
    state_plain = _state_plain(ctx.metabolic_state)
    top_rec = ctx.recommendations[0] if ctx.recommendations else "Check live status before departing"
    cascade_warning = ""
    if ctx.cascade_active and ctx.cascade_event:
        lag = ctx.cascade_event.lag_minutes
        targets = _cascade_targets_str(ctx)
        cascade_warning = (
            f"Note: congestion is spreading from Road {ctx.cascade_event.source_link_id} "
            f"and may affect {targets} in approximately {lag} minutes."
        )

    data = _data_block({
        "output_type": "citizen_advice",
        "link_id": ctx.link_id,
        "state_plain": state_plain,
        "queue_min": ctx.queue_minutes_str,
        "top_rec_plain": top_rec,
        "cascade_warning": cascade_warning or "No cascade warning.",
        "is_am_peak": ctx.is_am_peak,
        "archetype": ctx.archetype or "Unknown",
    })

    forbidden = ", ".join(f'"{t}"' for t in config.LLM_CITIZEN_FORBIDDEN)
    return f"""\
You are a friendly travel assistant for Pangyo Smart City.
Your job: give a commuter clear, simple advice about Road {ctx.link_id}.

STRICT RULES:
- Maximum 3 sentences.
- NEVER use any of these technical terms: {forbidden}
- Only refer to facts listed in [DATA] below.
- Use plain everyday language a 12-year-old would understand.
- Express time delays in minutes and seconds only.
- End with one specific, actionable tip.

{data}

Road {ctx.link_id} is currently showing {state_plain}, with an average wait of {ctx.queue_minutes_str}.
{cascade_warning}

Write your 3-sentence travel advisory now:"""


# --------------------------------------------------------------------------- #
# Planner Briefing (technical, full context)
# --------------------------------------------------------------------------- #

def build_planner_briefing_prompt(ctx: UrbanPulseContext) -> str:
    shap_text = _shap_lines(ctx)
    rec_list = "\n".join(f"  - {r}" for r in ctx.recommendations) or "  - None triggered"
    arch_desc = ctx.archetype_description or "Unknown"
    arch_policy = ctx.archetype_policy_class or "General traffic management"

    data = _data_block({
        "output_type": "planner_briefing",
        "link_id": ctx.link_id,
        "health_score": f"{ctx.road_health_score:.1f}",
        "metabolic_state": ctx.metabolic_state,
        "archetype": ctx.archetype or "Unknown",
        "congestion_prob_pct": ctx.congestion_prob_pct,
        "predicted_queue_s": f"{ctx.predicted_queue_s:.0f}",
        "historical_baseline_queue_s": f"{ctx.historical_baseline_queue_s:.0f}",
        "congestion_risk_score": f"{ctx.congestion_risk_score:.0f}",
        "top_rec": ctx.recommendations[0] if ctx.recommendations else "None",
        "shap_summary": ctx.top_shap[0].plain_english if ctx.top_shap else "N/A",
        "is_am_peak": ctx.is_am_peak,
        "stability_score": f"{ctx.stability_score:.2f}",
    })

    return f"""\
You are an expert traffic operations analyst. Produce a concise technical briefing.

RULES:
- Maximum 150 words.
- Reference only facts from [DATA] and the context below.
- Include: health score, metabolic state, archetype, primary drivers, top recommendation.
- Technical language is appropriate (occupancy, SHAP, archetype, cascade).

{data}

CONTEXT:
Road {ctx.link_id} | Health Score: {ctx.road_health_score:.1f}/100 | State: {ctx.metabolic_state}
Archetype: {ctx.archetype} -- {arch_desc}
Policy class: {arch_policy}
Congestion probability: {ctx.congestion_prob_pct}% | Queue: {ctx.predicted_queue_s:.0f}s
Baseline for this hour: {ctx.historical_baseline_queue_s:.0f}s | Risk percentile: {ctx.congestion_risk_score:.0f}th

Primary causal drivers (SHAP):
{shap_text}

Active recommendations:
{rec_list}

Write your technical planner briefing now:"""


# --------------------------------------------------------------------------- #
# Cascade Alert
# --------------------------------------------------------------------------- #

def build_cascade_alert_prompt(ctx: UrbanPulseContext) -> str:
    if not ctx.cascade_event:
        targets_str = "downstream roads"
        lag = 8
        n_down = 0
    else:
        targets_str = _cascade_targets_str(ctx)
        lag = ctx.cascade_event.lag_minutes
        n_down = ctx.cascade_event.n_downstream

    data = _data_block({
        "output_type": "cascade_alert",
        "link_id": ctx.link_id,
        "metabolic_state": ctx.metabolic_state,
        "cascade_targets_str": targets_str,
        "cascade_lag_minutes": lag,
        "n_downstream": n_down,
    })

    return f"""\
You are a real-time traffic alert system. Issue a 2-sentence CRITICAL cascade alert.

RULES:
- Exactly 2 sentences.
- Sentence 1: State that Road {ctx.link_id} has entered {ctx.metabolic_state} state.
- Sentence 2: Name the downstream roads and the predicted time window.
- Use operational language: direct, urgent, specific.
- Only use facts from [DATA].

{data}

Write the cascade alert now:"""


# --------------------------------------------------------------------------- #
# Counterfactual Summary
# --------------------------------------------------------------------------- #

def build_counterfactual_summary_prompt(ctx: UrbanPulseContext) -> str:
    cf = ctx.counterfactual
    if cf is None:
        return "No counterfactual data available for this link."

    cascade_line = (
        "The cascade to downstream roads would have been prevented."
        if cf.cascade_prevented
        else "Cascade prevention was not confirmed at this intervention threshold."
    )
    mode_note = (
        "based on domain-informed policy simulation (intervention not observed in dataset)"
        if cf.estimation_mode == "policy_simulation"
        else "based on observed OLS-estimated causal effect"
    )

    data = _data_block({
        "output_type": "counterfactual_summary",
        "link_id": ctx.link_id,
        "cf_intervention": cf.intervention_description,
        "cf_observed_queue_s": f"{cf.observed_queue_s:.0f}",
        "cf_counterfactual_queue_s": f"{cf.counterfactual_queue_s:.0f}",
        "cf_reduction_pct": f"{cf.queue_reduction_pct:.1f}",
        "cf_vehicle_hours": f"{cf.vehicle_hours_saved:.0f}",
        "cf_cascade_prevented": cf.cascade_prevented,
        "cf_estimation_mode": cf.estimation_mode,
    })

    return f"""\
You are an urban traffic causal intelligence system. Summarize a counterfactual result.

RULES:
- Maximum 100 words.
- Always frame as ESTIMATED, never certain. Use language like "estimated to reduce" or "analysis suggests".
- Include: intervention name, queue delay reduction, vehicle-hours saved, cascade outcome.
- State the estimation mode ({mode_note}).
- Only cite numbers from [DATA].

{data}

Counterfactual scenario:
  Intervention: {cf.intervention_description}
  Observed queue: {cf.observed_queue_s:.0f}s | Counterfactual queue: {cf.counterfactual_queue_s:.0f}s
  Queue reduction: {cf.queue_reduction_pct:.1f}% | Vehicle-hours saved: {cf.vehicle_hours_saved:.0f}
  Cascade prevented: {cf.cascade_prevented}
  Estimation mode: {cf.estimation_mode} ({mode_note})

Write the counterfactual summary now:"""


# --------------------------------------------------------------------------- #
# Traffic Summary (network-wide, both audiences)
# --------------------------------------------------------------------------- #

def build_traffic_summary_prompt(ctx: UrbanPulseContext, n_critical: int = 0,
                                  worst_link: int = 36, worst_health: float = 25.0) -> str:
    data = _data_block({
        "output_type": "traffic_summary",
        "link_id": ctx.link_id,
        "n_critical": n_critical,
        "worst_link": worst_link,
        "worst_health": f"{worst_health:.1f}",
        "metabolic_state": ctx.metabolic_state,
        "health_score": f"{ctx.road_health_score:.1f}",
    })

    return f"""\
You are a traffic network status reporter. Write a network-wide summary.

RULES:
- Maximum 80 words.
- Suitable for both planners and senior officials.
- Mention the number of critical links, the worst performer, and the overall trend.
- No raw sensor values. Only what a decision-maker needs.
- Use only facts from [DATA].

{data}

Write the network summary now:"""


# --------------------------------------------------------------------------- #
# Interactive Q&A
# --------------------------------------------------------------------------- #

def build_qa_prompt(ctx: UrbanPulseContext, question: str,
                    audience: str = "citizen") -> str:
    """Build a Q&A prompt for either citizen or planner audience."""
    cf = ctx.counterfactual
    cf_block = ""
    if cf:
        cf_block = (
            f"Counterfactual result: {cf.intervention_description} is estimated to "
            f"reduce queue delay by {cf.queue_reduction_pct:.1f}%, saving "
            f"{cf.vehicle_hours_saved:.0f} vehicle-hours."
        )

    forbidden = ""
    if audience == "citizen":
        terms = ", ".join(f'"{t}"' for t in config.LLM_CITIZEN_FORBIDDEN)
        forbidden = f"NEVER use these technical terms: {terms}\n"

    arch_context = ""
    if ctx.archetype and audience == "planner":
        arch_context = (
            f"Road {ctx.link_id} archetype: {ctx.archetype} -- "
            f"{ctx.archetype_description}. "
            f"Policy class: {ctx.archetype_policy_class}."
        )
    elif ctx.archetype and audience == "citizen":
        # Translate archetype to plain English for citizen
        plain_arch = {
            "Landmine": "unpredictable during morning rush",
            "Chronic": "almost always congested, even at night",
            "Saturator": "very busy throughout the day",
            "Ghost": "sometimes closed, fast when open",
            "Commuter": "busiest during morning and evening",
            "Chameleon": "unpredictable -- varies day to day",
        }.get(ctx.archetype, "known for frequent delays")
        arch_context = f"Note: Road {ctx.link_id} is {plain_arch}."

    data = _data_block({
        "output_type": "question_answer",
        "link_id": ctx.link_id,
        "question": question,
        "archetype": ctx.archetype or "Unknown",
        "state_plain": _state_plain(ctx.metabolic_state),
        "queue_min": ctx.queue_minutes_str,
        "cf_reduction_pct": f"{cf.queue_reduction_pct:.1f}" if cf else "N/A",
        "cf_vehicle_hours": f"{cf.vehicle_hours_saved:.0f}" if cf else "N/A",
        "audience": audience,
    })

    audience_note = (
        "Use simple, jargon-free language." if audience == "citizen"
        else "Use technical traffic operations language."
    )

    return f"""\
You are the UrbanPulse traffic intelligence assistant.
Answer the user's question using ONLY the context provided below.
Do not invent facts, statistics, or predictions not present in the context.

RULES:
- Maximum 120 words.
- {audience_note}
{forbidden}- End with one specific actionable suggestion.
- If the question cannot be answered from context, say so directly.

{data}

ROAD CONTEXT:
Road {ctx.link_id} | State: {ctx.metabolic_state} | Queue: {ctx.queue_minutes_str}
{arch_context}
{cf_block}

USER QUESTION: {question}

Answer:"""

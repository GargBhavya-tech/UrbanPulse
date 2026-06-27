"""B10 entry point: LLM Intelligence Layer demo + gate.

Usage (from repo root):
    python scripts/10_llm_layer.py [--backend template|flan_t5|gemini]

Requires upstream artifacts (B6, B7, B8, B9):
    reports/engine/snapshot_d1_m585.json
    data/road_archetypes.json
    reports/echo/cascade_events.csv
    data/counterfactual_results.json

Produces:
    reports/llm/llm_outputs.json

Gate conditions (all must pass):
    1. All 6 output types generated without exception
    2. Citizen advice passes grounding (no forbidden terms)
    3. Counterfactual summary uses estimated language
    4. No crash on context assembly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm.context import from_artifacts
from llm.grounding import GroundingValidator
from llm.layer import LLMLayer, save_outputs


def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def _gate(label: str, passed: bool) -> None:
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {label}")
    if not passed:
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="B10 LLM Intelligence Layer demo")
    ap.add_argument(
        "--backend", default=config.LLM_DEFAULT_BACKEND,
        choices=["template", "flan_t5", "gemini"],
        help="LLM backend to use",
    )
    ap.add_argument("--link", type=int, default=36, help="Link ID for demo context")
    args = ap.parse_args()

    _banner("B10 -- LLM Intelligence Layer")
    print(f"  Backend  : {args.backend}")
    print(f"  Link ID  : {args.link}")

    # -- Step 1: Assemble context from artifacts ----------------------------
    print("\nStep 1: Loading context from upstream artifacts ...")
    ctx_ok = False
    ctx = None
    try:
        ctx = from_artifacts(
            link_id=args.link,
            day_number=1,
            minute_of_day=585,
        )
        ctx_ok = True
        print(f"  Link {ctx.link_id} | state={ctx.metabolic_state} | "
              f"health={ctx.road_health_score:.1f} | archetype={ctx.archetype}")
        print(f"  cascade_active={ctx.cascade_active} | "
              f"counterfactual={'yes' if ctx.counterfactual else 'no'}")
    except Exception as exc:
        print(f"  ERROR assembling context: {exc}")
    _gate("Context assembly from B6/B7/B8/B9 artifacts", ctx_ok)
    assert ctx is not None  # for type checker

    # -- Step 2: Initialise LLM layer --------------------------------------
    print(f"\nStep 2: Initialising LLMLayer (backend={args.backend}) ...")
    layer = LLMLayer(backend=args.backend)
    print(f"  Backend class: {layer.backend_name}")

    # -- Step 3: Generate all 6 output types --------------------------------
    print("\nStep 3: Generating all output types ...")
    outputs_ok = True
    outputs = {}
    try:
        outputs = layer.generate_all(
            ctx,
            question="What could have prevented the July 1 disaster on Road 36?",
            n_critical=17,
            worst_link=36,
            worst_health=ctx.road_health_score,
        )
    except Exception as exc:
        print(f"  ERROR during generation: {exc}")
        outputs_ok = False

    _gate("All 6 output types generated without exception", outputs_ok)

    # -- Step 4: Print outputs ----------------------------------------------
    _banner("Generated Outputs")

    sections = [
        ("CITIZEN TRAVEL ADVICE", "citizen_advice", "citizen"),
        ("PLANNER BRIEFING", "planner_briefing", "planner"),
        ("CASCADE ALERT", "cascade_alert", "planner"),
        ("COUNTERFACTUAL SUMMARY", "counterfactual_summary", "planner"),
        ("TRAFFIC SUMMARY", "traffic_summary", "planner"),
        ("Q&A -- CITIZEN", "answer_question_citizen", "citizen"),
        ("Q&A -- PLANNER", "answer_question_planner", "planner"),
    ]

    validator = GroundingValidator()
    all_grounding_pass = True
    grounding_results = {}

    for title, key, audience in sections:
        text = outputs.get(key, "")
        if not text:
            print(f"\n[{title}]\n  (empty -- skipped)")
            continue
        print(f"\n[{title}]")
        print(text)
        vr = validator.validate(
            text, audience=audience, ctx=ctx, output_type=key, prompt=""
        )
        grounding_results[key] = vr
        if not vr.passed:
            all_grounding_pass = False
            print(f"  WARNING -- grounding violations: {vr.violations}")

    # -- Step 5: Gates -------------------------------------------------------
    _banner("B10 Gates")

    citizen_ok = grounding_results.get("citizen_advice", None)
    citizen_grounded = (citizen_ok is not None and citizen_ok.passed)
    _gate("Citizen advice passes grounding (no forbidden terms)", citizen_grounded)

    cf_ok = grounding_results.get("counterfactual_summary", None)
    cf_hedged = True
    if outputs.get("counterfactual_summary"):
        hedge_words = {"estimated", "estimate", "approximately", "suggests", "analysis"}
        cf_hedged = any(w in outputs["counterfactual_summary"].lower() for w in hedge_words)
    _gate("Counterfactual summary uses estimated language", cf_hedged)

    _gate("All grounding checks passed", all_grounding_pass)

    # -- Step 6: Save outputs -----------------------------------------------
    config.LLM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = save_outputs(outputs)
    print(f"\n  Saved -> {saved}")

    print("\n  GATE (B10 overall): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

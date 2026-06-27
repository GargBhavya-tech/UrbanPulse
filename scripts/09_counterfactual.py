"""B9 entry point: ECHO Stage C -- Counterfactual Intervention Engine.

Usage (from repo root):
    python scripts/09_counterfactual.py

Requires upstream artifacts:
    data/features.parquet      (B2)
    data/road_archetypes.json  (B7)
    data/causal_graph.json     (B8)
    reports/echo/cascade_events.csv  (B8)

Produces:
    data/counterfactual_results.json
    reports/echo/scm_coefficients.json
    reports/echo/scm_graph.png
    reports/echo/july1_counterfactual.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from echo import counterfactual


def main() -> int:
    out = counterfactual.run()

    print("\n=== B9 COUNTERFACTUAL INTERVENTION ENGINE ===")
    print(f"  SCMs fitted          : {out['n_scms_fitted']}/{config.EXPECTED_LINKS}")
    print(f"  OLS / policy-sim     : {out.get('n_links_ols', '?')} / {out.get('n_links_policy_simulation', '?')}")
    print(f"  Links processed      : {out['n_links_processed']}")
    print(f"  July 1 CF success    : {out['july1_ok']}")
    print(f"  Narrative produced   : {out['narrative_ok']}")
    print(f"  Archetypes covered   : {out['all_archetypes_covered']}")

    if out["july1_ok"] and "narrative" in out["july1_result"]:
        print("\n--- July 1 Centrepiece Counterfactual ---")
        print(out["july1_result"]["narrative"])
        jr = out["july1_result"]
        print(f"\n  Estimation mode      : {jr.get('estimation_mode', '?')}")
        print(f"  Observed queue       : {jr.get('observed_queue_s', '?')}s")
        print(f"  Counterfactual queue : {jr.get('counterfactual_queue_s', '?')}s")
        print(f"  Reduction            : {jr.get('queue_reduction_s', '?')}s  ({jr.get('queue_reduction_pct', '?')}%)")
        print(f"  Cascade source conf. : {jr.get('cascade_source_confirmed', '?')}")
        print(f"  Cascade prevented    : {jr.get('cascade_prevented', '?')}")
        print(f"  Vehicle-hours saved  : {jr.get('vehicle_hours_saved', '?')}")

    print(f"\n  counterfactual JSON  -> {config.COUNTERFACTUAL_RESULTS_JSON}")
    print(f"  SCM graph            -> {out['scm_graph_path']}")

    passed = out["july1_ok"] and out["narrative_ok"] and out["all_archetypes_covered"]
    print(f"\n  GATE (July-1 CF + narrative + archetypes covered): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


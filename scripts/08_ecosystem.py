"""B8 entry point: ECHO Stage B — Ecosystem State Machine.

Usage (from repo root):
    python scripts/08_ecosystem.py

Produces:
    data/causal_graph.json
    reports/echo/cascade_events.csv
    data/ecosystem_state.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from echo import ecosystem


def main() -> int:
    out = ecosystem.run()
    print("\n=== B8 ECOSYSTEM STATE MACHINE ===")
    print(f"  causal edges       : {out['n_edges']}")
    if out["edge_36_16"]:
        e = out["edge_36_16"]
        print(f"  36 -> 16 edge      : corr={e['correlation_strength']:.3f}  lag={e['lag_minutes']}min")
    else:
        print("  36 -> 16 edge      : NOT FOUND")
    print(f"  cascade events     : {out['n_cascade_events']}  (Day 1: {out['n_cascade_events_day1']})")
    print(f"  validation rate    : {out['validation_rate']:.1%}")
    print(f"  demo event         : {out['demo_event']}")
    print(f"  causal graph       -> {config.CAUSAL_GRAPH_JSON}")
    print(f"  cascade events     -> {config.CASCADE_EVENTS_CSV}")
    print(f"  ecosystem snapshot -> {config.ECOSYSTEM_STATE_JSON}")
    passed = out["edge_36_16"] is not None and out["n_cascade_events_day1"] > 0
    print(f"\n  GATE (36->16 edge + July-1 cascade): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

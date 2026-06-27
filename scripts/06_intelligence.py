"""B6 entry point: run the Traffic Intelligence Engine on a snapshot.

Usage (from repo root):
    python scripts/06_intelligence.py [--day DAY] [--minute MIN]

Without args, auto-picks the B8 demo cascade event (largest fan-out) if
cascade_events.csv exists, so the R7 CASCADE_PROPAGATING rule actually fires.
Falls back to Day 4, 08:10 if B8 hasn't run yet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import io_utils
from echo import ecosystem
from engine import intelligence


def _default_snapshot() -> tuple[int, int, str]:
    if config.CASCADE_EVENTS_CSV.exists():
        events = pd.read_csv(config.CASCADE_EVENTS_CSV)
        if len(events):
            demo = events.sort_values("n_downstream", ascending=False).iloc[0]
            return int(demo["day_number"]), int(demo["minute_of_day"]), "B8 demo cascade event"
    return 4, 490, "fallback (no cascade_events.csv yet)"


def main() -> int:
    default_day, default_minute, default_reason = _default_snapshot()
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=default_day)
    ap.add_argument("--minute", type=int, default=default_minute)
    args = ap.parse_args()
    if args.day == default_day and args.minute == default_minute:
        print(f"  (using default: {default_reason})")

    model = joblib.load(config.BEST_MODEL_PKL)
    features = io_utils.load_parquet(config.FEATURES_PARQUET)

    config.ENGINE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    census = intelligence.rule_activation_census(features)
    census.to_csv(config.ENGINE_REPORTS_DIR / "rule_activation_census.csv", index=False)
    print("\n=== RULE ACTIVATION CENSUS ===")
    print(census.to_string(index=False))

    archetypes = intelligence.load_archetypes()
    if archetypes:
        print(f"  ({len(archetypes)} archetypes loaded — archetype rules active)")

    ecosystem_state: dict[int, bool] = {}
    if config.CASCADE_EVENTS_CSV.exists():
        events = pd.read_csv(config.CASCADE_EVENTS_CSV)
        ecosystem_state = ecosystem.cascade_propagating_map(events, args.day, args.minute)
        if ecosystem_state:
            print(f"  (B8 cascade flags active on: {list(ecosystem_state)})")

    result = intelligence.analyze_snapshot(
        features, model, args.day, args.minute,
        archetypes=archetypes, ecosystem_state=ecosystem_state,
    )

    hh, mm = divmod(args.minute, 60)
    print(f"\n=== TRAFFIC INTELLIGENCE — Day {args.day}, {hh:02d}:{mm:02d} ===")
    print(f"  links={result['n_links']}  critical={result['n_critical']}\n")

    print("  Hotspot ranking (worst 8):")
    for r in result["hotspot_ranking"][:8]:
        print(f"    Link {r['link_id']:>2}  health={r['health_score']:6.2f}  {r['state']}")

    print("\n  Recommendations:")
    issued = 0
    for link in result["links"]:
        for rec in link["recommendations"]:
            issued += 1
            print(f"    [Link {link['link_id']}] ({rec['severity']}) {rec['recommendation']}")
            print(f"        reason: {rec['reasoning']}")
    if issued == 0:
        print("    (none triggered)")

    out_path = config.ENGINE_REPORTS_DIR / f"snapshot_d{args.day}_m{args.minute}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Saved -> {out_path}")

    all_have_reason = all(
        rec["reasoning"].strip()
        for link in result["links"]
        for rec in link["recommendations"]
    )

    # Always also generate the July 1 centrepiece snapshot (d1, m585 = 09:45 AM)
    # so the demo is always current.
    if not (args.day == 1 and args.minute == 585):
        july1_eco: dict[int, bool] = {}
        if config.CASCADE_EVENTS_CSV.exists():
            events_df = pd.read_csv(config.CASCADE_EVENTS_CSV)
            july1_eco = ecosystem.cascade_propagating_map(events_df, 1, 585)
        july1_snap = intelligence.analyze_snapshot(
            features, model, 1, 585, archetypes=archetypes, ecosystem_state=july1_eco,
        )
        july1_path = config.ENGINE_REPORTS_DIR / "snapshot_d1_m585.json"
        july1_path.write_text(json.dumps(july1_snap, indent=2))
        july1_recs = sum(len(l["recommendations"]) for l in july1_snap["links"])
        july1_arch36 = next(
            (l.get("archetype") for l in july1_snap["links"] if l["link_id"] == 36), None
        )
        print(f"\n  July 1 snapshot (09:45 AM): critical={july1_snap['n_critical']}  "
              f"recs={july1_recs}  Link36.archetype={july1_arch36}  -> {july1_path}")

        all_july1_reason = all(
            rec["reasoning"].strip()
            for l in july1_snap["links"]
            for rec in l["recommendations"]
        )
        all_have_reason = all_have_reason and all_july1_reason

    print(f"  GATE (every rec has reasoning): {'PASS' if all_have_reason else 'FAIL'}")
    return 0 if all_have_reason else 1


if __name__ == "__main__":
    raise SystemExit(main())

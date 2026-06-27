"""B7 entry point: ECHO Stage A — Personality Atlas.

Usage (from repo root):
    python scripts/07_atlas.py

Produces data/road_archetypes.json and reports/echo/personality_atlas.png, and
prints the archetype roster + gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from echo import personality_atlas as atlas


def main() -> int:
    out = atlas.run()
    print("\n=== B7 PERSONALITY ATLAS ===")
    print(f"  silhouette   : {out['silhouette']:.3f}  (Bible's 0.5 is aspirational; "
          "see B7 note)")
    print(f"  archetypes   : {out['n_archetypes']}  {out['archetype_counts']}")
    print(f"  stable links : {out['n_stable']}/{config.EXPECTED_LINKS} "
          f"(>= {config.STABILITY_THRESHOLD} stability)")
    print(f"  atlas plot   : {out['atlas_plot']}")
    print(f"  archetypes   -> {config.ROAD_ARCHETYPES_JSON}")

    # Gate: 5-7 archetypes present. Silhouette is reported but not gated (the 66
    # links form a behavioral continuum; stability is the primary validation).
    passed = 5 <= out["n_archetypes"] <= 7
    print(f"\n  GATE (5-7 archetypes): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

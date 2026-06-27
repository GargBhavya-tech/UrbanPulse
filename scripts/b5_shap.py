"""B5 entry point: compute SHAP explainability and write all six outputs.

Usage (from the repo root):
    python scripts/b5_shap.py

Requires:
    data/features.parquet  — from B2
    models/best_model.pkl  — from B4

Produces (all in reports/shap/):
    01_beeswarm.png
    02_importance_bar.png
    03_waterfall_link36.png
    04_waterfall_link37.png
    05_dependence_hour.png
    06_dependence_mean_occup.png
    translations.json
    b5_gate.json
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root importable when run as `python scripts/b5_shap.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shap_analysis


def main() -> int:
    """Run B5. Returns 0 on a passing gate, 1 otherwise."""
    result = shap_analysis.run()

    print("\n=== B5 SHAP EXPLAINABILITY ===")
    print(f"  Plots produced:    {result['plots_produced']}")
    print(f"  Global sample:     {result['global_sample_rows']} rows "
          f"(pos={result['global_sample_pos_rate']:.3f})")
    print(f"  Link 36 meta:      {result['link36_meta']}")
    print(f"  Link 37 meta:      {result['link37_meta']}")
    print(f"  Translations:      {result['translations_path']}")

    passed = result["all_plots_exist"] and len(result["plots_produced"]) == 6
    print(f"\n  GATE: {'PASS — all 6 SHAP outputs produced' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

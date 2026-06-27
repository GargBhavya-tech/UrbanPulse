"""B2 entry point: build the model-ready feature set from cleaned data.

Usage (from the repo root):
    python scripts/02_features.py

Produces:
    data/features.parquet
    data/feature_norms.json
and prints the B2 gate (no NaN, target class balance ~13% positive).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import features as feat
import io_utils


def main() -> int:
    """Run B2. Returns 0 on a passing gate, 1 otherwise."""
    print(f"Loading cleaned parquet <- {config.CLEANED_PARQUET}")
    df = io_utils.load_parquet(config.CLEANED_PARQUET)

    print("Building features ...")
    features = feat.build_features(df)

    print(f"Saving features parquet -> {config.FEATURES_PARQUET}")
    io_utils.save_parquet(features, config.FEATURES_PARQUET)

    summary = feat.feature_summary(features)
    print("\n=== B2 GATE ===")
    for key, value in summary.items():
        print(f"  {key:>22}: {value}")

    # Bible expects ~13% positive; accept a tolerant band around it.
    rate = summary["target_positive_rate"]
    passed = (
        summary["nan_cells"] == 0
        and summary["n_features"] > 0
        and 0.10 <= rate <= 0.16
    )
    print(f"\n  target positive rate: {rate:.4f} (Bible target ~0.13)")
    print(f"  GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

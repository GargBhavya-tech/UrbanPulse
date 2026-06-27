"""B3 horizon sweep — evidence for ticket #12.

Trains a single fast model (HistGradientBoosting, CPU, no extra deps) under four
framings and reports test-set ROC-AUC / PR-AUC so the horizon decision is made
on data rather than anecdote:

    nowcast_leakfree : predict current congestion, target-defining cols dropped
    forecast_+5min   : predict congestion 1 interval ahead, all features
    forecast_+10min  : 2 intervals ahead
    forecast_+15min  : 3 intervals ahead

Usage:
    python scripts/b3_horizon_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import config
import io_utils
import modeling


def _run(df: pd.DataFrame, label: str, horizon: int, leak_free: bool) -> dict:
    x_train, x_val, x_test, y = modeling.prepare_xy(df, horizon, leak_free)
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=6, random_state=42
    )
    model.fit(x_train, y["train"])
    proba = model.predict_proba(x_test)[:, 1]
    metrics = modeling.evaluate(y["test"], proba)
    return {
        "framing": label,
        "n_features": x_train.shape[1],
        "train_rows": len(x_train),
        "test_pos_rate": round(metrics["pos_rate"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "pr_auc": round(metrics["pr_auc"], 4),
    }


def main() -> int:
    df = io_utils.load_parquet(config.FEATURES_PARQUET)
    runs = [
        _run(df, "nowcast_leakfree", horizon=0, leak_free=True),
        _run(df, "forecast_+5min", horizon=1, leak_free=False),
        _run(df, "forecast_+10min", horizon=2, leak_free=False),
        _run(df, "forecast_+15min", horizon=3, leak_free=False),
    ]
    table = pd.DataFrame(runs)
    print("\n=== B3 HORIZON SWEEP (test set, HistGradientBoosting) ===\n")
    print(table.to_string(index=False))
    print(
        "\nNote: nowcast_leakfree drops the 6 target-defining columns; the "
        "forecast rows keep all features (no leakage because the label is in "
        "the future)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

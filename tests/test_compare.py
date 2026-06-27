"""Tests for the comparison/selection logic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare


def test_best_threshold_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 500))
    proba = rng.random(500)
    thr = compare._best_threshold(y, proba)
    assert 0.0 <= thr <= 1.0


def test_select_best_prefers_roc_within_latency() -> None:
    table = pd.DataFrame(
        {
            "model": ["a", "b", "c"],
            "roc_auc": [0.97, 0.99, 0.98],
            "precision@op": [0.7, 0.7, 0.7],
            "infer_ms_per_row": [1.0, 600.0, 1.0],  # b violates latency
        }
    )
    # b has best ROC but is too slow -> c wins
    assert compare.select_best(table) == "c"


def test_select_best_tiebreak_precision() -> None:
    table = pd.DataFrame(
        {
            "model": ["a", "b"],
            "roc_auc": [0.97, 0.97],
            "precision@op": [0.70, 0.80],
            "infer_ms_per_row": [1.0, 1.0],
        }
    )
    assert compare.select_best(table) == "b"


@pytest.mark.skipif(
    not (compare.config.MODELS_DIR / "best_model_meta.json").exists(),
    reason="B4 not run yet",
)
def test_best_meta_passes_gate() -> None:
    import json

    meta = json.loads((compare.config.MODELS_DIR / "best_model_meta.json").read_text())
    assert meta["test_metrics_at_op"]["roc_auc"] > 0.85
    assert meta["horizon_intervals"] == compare.config.HORIZON_INTERVALS

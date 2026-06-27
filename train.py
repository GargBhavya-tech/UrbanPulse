"""Model training for UrbanPulse B3 (Bible §5 Notebook 03).

Trains the Bible's seven models on the +15 min forecast target (decision #12),
records the full metric panel + timings, and persists each fitted model.

Inputs are already model-ready: ``modeling.prepare_xy`` shifts the target,
applies the temporal split, and target-encodes LINK_ID (fit on train only), so
no further preprocessing is needed for the tree models. SVM (Bible priority 8,
"if time") is intentionally skipped: SVC with probabilities is O(n^2) and
impractical on ~190k CPU rows; it can be added later on a subsample if wanted.
"""
from __future__ import annotations

import time
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

import config
import io_utils
import modeling

RANDOM_STATE = 42


def _model_registry() -> dict[str, Callable]:
    """Build the model registry, importing boosting libs lazily.

    Returns:
        Ordered dict of ``name -> zero-arg constructor``. Boosting models are
        included only if their library imports successfully (decision #6:
        degrade gracefully).
    """
    registry: dict[str, Callable] = {
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=12, random_state=RANDOM_STATE
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }
    try:
        from xgboost import XGBClassifier

        registry["xgboost"] = lambda: XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        print("  (xgboost not available — skipping)")
    try:
        from lightgbm import LGBMClassifier

        registry["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.1,
            n_jobs=-1,
            verbose=-1,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        print("  (lightgbm not available — skipping)")
    try:
        from catboost import CatBoostClassifier

        registry["catboost"] = lambda: CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.1,
            verbose=0,
            random_seed=RANDOM_STATE,
        )
    except ImportError:
        print("  (catboost not available — skipping)")
    return registry


def _metrics(
    name: str, model: object, x_test: pd.DataFrame, y_test: pd.Series, train_s: float
) -> dict[str, object]:
    """Compute the full metric panel for one fitted model."""
    t0 = time.perf_counter()
    proba = model.predict_proba(x_test)[:, 1]
    infer_total = time.perf_counter() - t0
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, pred, average="weighted"), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "train_time_s": round(train_s, 2),
        "infer_ms_per_row": round(infer_total / len(x_test) * 1000, 4),
    }


def train_all() -> pd.DataFrame:
    """Train all available models on the +15 min target; persist + score them.

    Returns:
        Metrics DataFrame (one row per model), also written to
        ``reports/model_metrics.csv``. Each fitted model is saved to
        ``models/<name>.pkl``.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = io_utils.load_parquet(config.FEATURES_PARQUET)
    x_train, x_val, x_test, y = modeling.prepare_xy(
        df, horizon=config.HORIZON_INTERVALS, leak_free=False
    )
    print(
        f"train={len(x_train)} val={len(x_val)} test={len(x_test)} "
        f"features={x_train.shape[1]} pos_rate={y['train'].mean():.3f}"
    )

    rows = []
    for name, make in _model_registry().items():
        print(f"  training {name} ...", flush=True)
        model = make()
        t0 = time.perf_counter()
        model.fit(x_train, y["train"])
        train_s = time.perf_counter() - t0
        joblib.dump(model, config.MODELS_DIR / f"{name}.pkl")
        rows.append(_metrics(name, model, x_test, y["test"], train_s))

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(config.MODEL_METRICS_CSV, index=False)
    return metrics


if __name__ == "__main__":
    table = train_all()
    print("\n=== B3 MODEL METRICS (test set, +15 min forecast) ===\n")
    print(table.to_string(index=False))
    n = len(table)
    gate_auc = (table["roc_auc"] > 0.85).all()
    gate_infer = (table["infer_ms_per_row"] < 500).all()
    print(f"\n  models trained: {n}")
    print(f"  all ROC-AUC > 0.85: {gate_auc}")
    print(f"  all inference < 500ms/row: {gate_infer}")
    print(f"  GATE: {'PASS' if (n >= 7 and gate_auc and gate_infer) else 'FAIL'}")

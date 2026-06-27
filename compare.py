"""Model comparison & selection for UrbanPulse B4 (Bible §5 Notebook 04).

Loads the seven fitted models, scores them on the temporal val/test splits,
selects the best by test ROC-AUC (tiebreak: precision at its operating
threshold; constraint: inference < 500 ms/row), and chooses an operating
threshold by maximising F1 on the VALIDATION split (so the threshold is never
fit on test). Produces the Bible's required comparison visuals and persists the
winner to ``models/best_model.pkl`` with metadata.
"""
from __future__ import annotations

import json
import shutil

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
import io_utils
import modeling

COMPARE_DIR = config.REPORTS_DIR / "model_comparison"
MODEL_ORDER = [
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
]


def _load_models() -> dict[str, object]:
    """Load every ``models/<name>.pkl`` that exists, in canonical order."""
    models = {}
    for name in MODEL_ORDER:
        path = config.MODELS_DIR / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
    if not models:
        raise FileNotFoundError(
            "No model pkls found. Run `python train.py` first."
        )
    return models


def _best_threshold(y_val: pd.Series, proba_val: np.ndarray) -> float:
    """Threshold that maximises F1 on the validation split."""
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # thr has len-1 vs prec/rec; align by dropping the last prec/rec point
    return float(thr[int(np.nanargmax(f1[:-1]))])


def score_models() -> tuple[pd.DataFrame, dict, dict]:
    """Score all models on test; pick operating thresholds on val.

    Returns:
        ``(table, probas_test, thresholds)`` where ``table`` has one row per
        model with test metrics at its val-tuned threshold, ``probas_test`` maps
        name -> test probabilities (for curves), and ``thresholds`` maps
        name -> operating threshold.
    """
    df = io_utils.load_parquet(config.FEATURES_PARQUET)
    x_train, x_val, x_test, y = modeling.prepare_xy(
        df, horizon=config.HORIZON_INTERVALS, leak_free=False
    )
    models = _load_models()
    timing = (
        pd.read_csv(config.MODEL_METRICS_CSV).set_index("model")
        if config.MODEL_METRICS_CSV.exists()
        else None
    )

    rows, probas_test, thresholds = [], {}, {}
    for name, model in models.items():
        p_val = model.predict_proba(x_val)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]
        thr = _best_threshold(y["val"], p_val)
        pred = (p_test >= thr).astype(int)
        probas_test[name] = p_test
        thresholds[name] = thr
        rows.append(
            {
                "model": name,
                "roc_auc": round(roc_auc_score(y["test"], p_test), 4),
                "pr_auc": round(average_precision_score(y["test"], p_test), 4),
                "op_threshold": round(thr, 3),
                "precision@op": round(precision_score(y["test"], pred, zero_division=0), 4),
                "recall@op": round(recall_score(y["test"], pred, zero_division=0), 4),
                "f1@op": round(f1_score(y["test"], pred), 4),
                "train_time_s": float(timing.loc[name, "train_time_s"]) if timing is not None else np.nan,
                "infer_ms_per_row": float(timing.loc[name, "infer_ms_per_row"]) if timing is not None else np.nan,
            }
        )
    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return table, probas_test, thresholds


def select_best(table: pd.DataFrame) -> str:
    """Select best model: max test ROC-AUC, tiebreak precision@op, infer<500ms."""
    eligible = table[table["infer_ms_per_row"] < 500]
    eligible = eligible.sort_values(
        ["roc_auc", "precision@op"], ascending=False
    )
    return str(eligible.iloc[0]["model"])


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
def _y_test() -> pd.Series:
    df = io_utils.load_parquet(config.FEATURES_PARQUET)
    _, _, _, y = modeling.prepare_xy(df, config.HORIZON_INTERVALS, leak_free=False)
    return y["test"]


def make_visuals(table: pd.DataFrame, probas: dict, thresholds: dict, best: str) -> None:
    """Produce all Bible §5 Notebook 04 comparison figures."""
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    y_test = _y_test().to_numpy()

    # 1. ROC-AUC bar
    fig, ax = plt.subplots(figsize=(8, 4))
    t = table.sort_values("roc_auc")
    ax.barh(t["model"], t["roc_auc"], color="steelblue")
    ax.set_xlim(0.9, 1.0)
    ax.set_title("Models ranked by test ROC-AUC")
    for i, v in enumerate(t["roc_auc"]):
        ax.text(v, i, f" {v:.3f}", va="center")
    fig.tight_layout(); fig.savefig(COMPARE_DIR / "roc_auc_bar.png", dpi=110); plt.close(fig)

    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC curves")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(COMPARE_DIR / "roc_curves.png", dpi=110); plt.close(fig)

    # 3. PR curves
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, label=f"{name} ({average_precision_score(y_test, p):.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Precision-Recall curves")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(COMPARE_DIR / "pr_curves.png", dpi=110); plt.close(fig)

    # 4. Confusion matrices for top 3 (at each model's operating threshold)
    top3 = table.head(3)["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, top3):
        pred = (probas[name] >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title(f"{name} @thr={thresholds[name]:.2f}")
    fig.tight_layout(); fig.savefig(COMPARE_DIR / "confusion_top3.png", dpi=110); plt.close(fig)

    # 5. Efficiency frontier: train time vs ROC-AUC
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(table["train_time_s"], table["roc_auc"])
    for _, r in table.iterrows():
        ax.annotate(r["model"], (r["train_time_s"], r["roc_auc"]), fontsize=8)
    ax.set_xlabel("train time (s)"); ax.set_ylabel("test ROC-AUC")
    ax.set_title("Efficiency frontier")
    fig.tight_layout(); fig.savefig(COMPARE_DIR / "efficiency_frontier.png", dpi=110); plt.close(fig)

    # 6. Feature importance of the best model
    model = _load_models()[best]
    df = io_utils.load_parquet(config.FEATURES_PARQUET)
    x_train, *_ = modeling.prepare_xy(df, config.HORIZON_INTERVALS, leak_free=False)
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=x_train.columns).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.plot.barh(ax=ax, color="darkorange")
        ax.set_title(f"Top-15 feature importances — {best}")
        fig.tight_layout(); fig.savefig(COMPARE_DIR / "feature_importance_best.png", dpi=110); plt.close(fig)


def run() -> dict:
    """Full B4: score, select, visualize, persist winner. Returns metadata."""
    table, probas, thresholds = score_models()
    best = select_best(table)
    make_visuals(table, probas, thresholds, best)

    table.to_csv(COMPARE_DIR / "comparison_metrics.csv", index=False)
    shutil.copyfile(config.MODELS_DIR / f"{best}.pkl", config.BEST_MODEL_PKL)

    best_row = table[table["model"] == best].iloc[0].to_dict()
    meta = {
        "best_model": best,
        "operating_threshold": thresholds[best],
        "horizon_intervals": config.HORIZON_INTERVALS,
        "test_metrics_at_op": {
            k: best_row[k] for k in ["roc_auc", "pr_auc", "precision@op", "recall@op", "f1@op"]
        },
    }
    (config.MODELS_DIR / "best_model_meta.json").write_text(json.dumps(meta, indent=2))
    return {"table": table, "meta": meta}


if __name__ == "__main__":
    out = run()
    print("\n=== B4 MODEL COMPARISON (test set @ val-tuned threshold) ===\n")
    print(out["table"].to_string(index=False))
    m = out["meta"]
    print(f"\nBEST: {m['best_model']}  (op threshold {m['operating_threshold']:.3f})")
    print(f"  test @op: {json.dumps(m['test_metrics_at_op'])}")
    auc_ok = out["table"]["roc_auc"].max() > 0.85
    infer_ok = (out["table"]["infer_ms_per_row"] < 500).all()
    print(f"\n  GATE: {'PASS' if (auc_ok and infer_ok) else 'FAIL'}")

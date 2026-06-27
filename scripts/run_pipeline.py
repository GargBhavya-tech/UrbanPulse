"""Full pipeline runner: B1 (clean) -> B2 (features) -> B3 (train) -> B4 (compare).

Usage (from repo root):
    python scripts/run_pipeline.py [--data PATH_TO_CSV]

If --data is omitted, defaults to data/raw.csv.
Produces:
    data/cleaned.parquet
    data/features.parquet
    data/feature_norms.json
    models/<name>.pkl  (one per model)
    models/best_model.pkl
    models/best_model_meta.json
    reports/model_metrics.csv
    reports/model_comparison/  (charts + comparison_metrics.csv)
    reports/pipeline_results.txt  (human-readable summary)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Make repo root importable when run as `python scripts/run_pipeline.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import cleaning
import features as feat
import io_utils
import modeling
import train as train_module
import compare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def _gate(label: str, passed: bool) -> None:
    status = "PASS PASS" if passed else "FAIL FAIL"
    print(f"  GATE [{label}]: {status}")
    if not passed:
        print("  -> Aborting: fix this stage before proceeding.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# B1 — Clean
# ---------------------------------------------------------------------------
def run_b1(raw_csv: Path) -> None:
    _banner("B1 — EDA + Cleaning")
    print(f"  Loading raw CSV: {raw_csv}")
    t0 = time.perf_counter()
    raw = io_utils.load_raw(raw_csv)
    print(f"  Raw shape: {raw.shape}")

    df = cleaning.clean(raw)
    io_utils.save_parquet(df, config.CLEANED_PARQUET)
    print(f"  Saved -> {config.CLEANED_PARQUET}  ({time.perf_counter()-t0:.1f}s)")

    report = cleaning.integrity_report(df)
    for k, v in report.items():
        print(f"    {k:>25}: {v}")

    passed = (
        report["rows_match_expected"]
        and report["missing_cells"] == 0
        and report["max_occupancy"] <= config.OCCUPANCY_CAP
        and report["has_lane6_active"]
        and report["has_stall_flags"]
        and report["links"] == config.EXPECTED_LINKS
    )
    _gate("B1", passed)


# ---------------------------------------------------------------------------
# B2 — Features
# ---------------------------------------------------------------------------
def run_b2() -> None:
    _banner("B2 — Feature Engineering")
    t0 = time.perf_counter()
    df = io_utils.load_parquet(config.CLEANED_PARQUET)
    features = feat.build_features(df)
    io_utils.save_parquet(features, config.FEATURES_PARQUET)
    print(f"  Saved -> {config.FEATURES_PARQUET}  ({time.perf_counter()-t0:.1f}s)")

    summary = feat.feature_summary(features)
    for k, v in summary.items():
        print(f"    {k:>25}: {v}")

    rate = summary["target_positive_rate"]
    passed = summary["nan_cells"] == 0 and summary["n_features"] > 0 and 0.10 <= rate <= 0.16
    _gate("B2", passed)
    return features


# ---------------------------------------------------------------------------
# B3 — Train
# ---------------------------------------------------------------------------
def run_b3() -> None:
    _banner("B3 — Model Training (7 models, +15 min forecast)")
    t0 = time.perf_counter()
    metrics = train_module.train_all()
    print(f"\n  Training complete in {time.perf_counter()-t0:.1f}s")
    print(metrics[["model", "roc_auc", "pr_auc", "f1_weighted", "train_time_s",
                    "infer_ms_per_row"]].to_string(index=False))

    gate_auc = (metrics["roc_auc"] > 0.85).all()
    gate_infer = (metrics["infer_ms_per_row"] < 500).all()
    _gate("B3 ROC-AUC > 0.85", gate_auc)
    _gate("B3 infer < 500ms", gate_infer)
    return metrics


# ---------------------------------------------------------------------------
# B4 — Compare + Feature Importance
# ---------------------------------------------------------------------------
def run_b4() -> dict:
    _banner("B4 — Model Comparison + Feature Importance")
    out = compare.run()
    meta = out["meta"]
    table = out["table"]

    print(f"\n  Best model : {meta['best_model']}")
    print(f"  Op threshold: {meta['operating_threshold']:.3f}")
    print(f"\n  Full comparison table:")
    print(table.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Feature importance report
# ---------------------------------------------------------------------------
def print_feature_importance(best_name: str) -> None:
    _banner(f"Feature Importance — {best_name}")
    import joblib
    import pandas as pd

    model = joblib.load(config.MODELS_DIR / f"{best_name}.pkl")
    df = io_utils.load_parquet(config.FEATURES_PARQUET)
    x_train, *_ = modeling.prepare_xy(df, config.HORIZON_INTERVALS, leak_free=False)

    if not hasattr(model, "feature_importances_"):
        print("  (model does not expose feature_importances_ — skipping)")
        return

    imp = (
        pd.Series(model.feature_importances_, index=x_train.columns)
        .rename("importance")
        .sort_values(ascending=False)
    )
    imp_pct = (imp / imp.sum() * 100).round(2)

    print(f"\n  {'Feature':<35} {'Importance':>12}  {'%':>6}")
    print("  " + "-" * 58)
    for feat_name, val in imp.items():
        print(f"  {feat_name:<35} {val:>12.6f}  {imp_pct[feat_name]:>5.2f}%")

    # Save to reports
    imp_df = pd.DataFrame({"feature": imp.index, "importance": imp.values,
                           "importance_pct": imp_pct.values})
    out_path = config.REPORTS_DIR / "feature_importance.csv"
    imp_df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return imp_df


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def print_summary(b3_metrics, b4_out) -> None:
    _banner("FINAL RESULTS SUMMARY")
    meta = b4_out["meta"]
    table = b4_out["table"]
    best = meta["best_model"]
    best_row = table[table["model"] == best].iloc[0]

    lines = [
        "=" * 60,
        "  UrbanPulse Pipeline Results",
        "  Dataset : Pangyo 14-day highway loop-detector data",
        "  Task    : Binary congestion prediction +15 min ahead",
        "  Split   : Temporal (Days 1-10 train / 11-12 val / 13-14 test)",
        "=" * 60,
        "",
        f"  Best model         : {best}",
        f"  Test ROC-AUC       : {best_row['roc_auc']:.4f}",
        f"  Test PR-AUC        : {best_row['pr_auc']:.4f}",
        f"  Test Precision@thr : {best_row['precision@op']:.4f}",
        f"  Test Recall@thr    : {best_row['recall@op']:.4f}",
        f"  Test F1@thr        : {best_row['f1@op']:.4f}",
        f"  Operating threshold: {meta['operating_threshold']:.3f}",
        f"  Infer ms/row       : {best_row['infer_ms_per_row']:.4f}",
        "",
        "  All models (sorted by test ROC-AUC):",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"    {r['model']:<20} AUC={r['roc_auc']:.4f}  "
            f"PR={r['pr_auc']:.4f}  F1={r['f1@op']:.4f}"
        )

    text = "\n".join(lines)
    print(text)

    out_path = config.REPORTS_DIR / "pipeline_results.txt"
    out_path.write_text(text + "\n")
    print(f"\n  Summary saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run UrbanPulse B1->B4 pipeline.")
    parser.add_argument(
        "--data",
        default=str(config.RAW_CSV),
        help=f"Path to raw CSV (default: {config.RAW_CSV})",
    )
    args = parser.parse_args()
    raw_csv = Path(args.data)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # If the user pointed to a different path, copy it to data/raw.csv so
    # all downstream stages find it in the canonical location.
    if raw_csv.resolve() != config.RAW_CSV.resolve():
        print(f"  Copying {raw_csv.name} -> {config.RAW_CSV}")
        config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_csv, config.RAW_CSV)

    run_b1(config.RAW_CSV)
    run_b2()
    b3_metrics = run_b3()
    b4_out = run_b4()
    imp_df = print_feature_importance(b4_out["meta"]["best_model"])
    print_summary(b3_metrics, b4_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Full pipeline runner: B1 -> B2 -> B3 -> B4 -> B5 -> B7 -> B8 -> B6.

Usage:
    python scripts/run_pipeline.py [--data PATH_TO_CSV]
"""
from __future__ import annotations

import argparse
import json as _json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import cleaning
import features as feat
import io_utils
import modeling
import train as train_module
import compare
import shap_analysis


def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def _gate(label: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  GATE [{label}]: {status}")
    if not passed:
        print("  -> Aborting.")
        sys.exit(1)


def run_b1(raw_csv: Path) -> None:
    _banner("B1 — EDA + Cleaning")
    t0 = time.perf_counter()
    raw = io_utils.load_raw(raw_csv)
    df = cleaning.clean(raw)
    io_utils.save_parquet(df, config.CLEANED_PARQUET)
    print(f"  Saved -> {config.CLEANED_PARQUET}  ({time.perf_counter()-t0:.1f}s)")
    report = cleaning.integrity_report(df)
    for k, v in report.items():
        print(f"    {k:>25}: {v}")
    passed = (
        report["rows_match_expected"] and report["missing_cells"] == 0
        and report["max_occupancy"] <= config.OCCUPANCY_CAP
        and report["has_lane6_active"] and report["has_stall_flags"]
        and report["links"] == config.EXPECTED_LINKS
    )
    _gate("B1", passed)


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


def run_b3():
    _banner("B3 — Model Training (7 models, +15 min forecast)")
    t0 = time.perf_counter()
    metrics = train_module.train_all()
    print(f"  Training complete in {time.perf_counter()-t0:.1f}s")
    print(metrics[["model", "roc_auc", "pr_auc", "f1_weighted", "train_time_s",
                    "infer_ms_per_row"]].to_string(index=False))
    _gate("B3 ROC-AUC > 0.85", (metrics["roc_auc"] > 0.85).all())
    _gate("B3 infer < 500ms", (metrics["infer_ms_per_row"] < 500).all())
    return metrics


def run_b4():
    _banner("B4 — Model Comparison + Feature Importance")
    out = compare.run()
    meta = out["meta"]
    print(f"  Best model: {meta['best_model']}")
    print(out["table"].to_string(index=False))
    return out


def run_b5() -> None:
    _banner("B5 — SHAP Explainability")
    result = shap_analysis.run()
    passed = result["all_plots_exist"] and len(result["plots_produced"]) == 6
    _gate("B5 — 6 SHAP plots", passed)


def run_b7() -> None:
    _banner("B7 — ECHO Stage A: Personality Atlas")
    from echo import personality_atlas as atlas
    out = atlas.run()
    print(f"  silhouette={out['silhouette']:.3f}  archetypes={out['archetype_counts']}")
    print(f"  stable={out['n_stable']}/{config.EXPECTED_LINKS}")
    _gate("B7 — 5-7 archetypes", 5 <= out["n_archetypes"] <= 7)


def run_b8() -> None:
    _banner("B8 — ECHO Stage B: Ecosystem State Machine")
    from echo import ecosystem
    out = ecosystem.run()
    print(f"  causal edges    : {out['n_edges']}")
    if out["edge_36_16"]:
        e = out["edge_36_16"]
        print(f"  36->16 edge     : corr={e['correlation_strength']:.3f}  lag={e['lag_minutes']}min")
    else:
        print("  36->16 edge     : NOT FOUND")
    print(f"  cascade events  : {out['n_cascade_events']}  (Day 1: {out['n_cascade_events_day1']})")
    print(f"  validation rate : {out['validation_rate']:.1%}")
    passed = out["edge_36_16"] is not None and out["n_cascade_events_day1"] > 0
    _gate("B8 — 36->16 edge + July-1 cascade", passed)


def run_b9() -> None:
    _banner("B9 — ECHO Stage C: Counterfactual Intervention Engine")
    from echo import counterfactual
    out = counterfactual.run()
    print(f"  SCMs fitted          : {out['n_scms_fitted']}/{config.EXPECTED_LINKS}")
    print(f"  Links processed      : {out['n_links_processed']}")
    print(f"  July 1 CF success    : {out['july1_ok']}")
    print(f"  Narrative produced   : {out['narrative_ok']}")
    if out["july1_ok"] and "narrative" in out["july1_result"]:
        jr = out["july1_result"]
        print(f"  Observed queue       : {jr.get('observed_queue_s', '?')}s")
        print(f"  Counterfactual queue : {jr.get('counterfactual_queue_s', '?')}s")
        print(f"  Queue reduction      : {jr.get('queue_reduction_pct', '?')}%")
        print(f"  Cascade prevented    : {jr.get('cascade_prevented', '?')}")
        print(f"  Vehicle-hours saved  : {jr.get('vehicle_hours_saved', '?')}")
    passed = out["july1_ok"] and out["narrative_ok"] and out["all_archetypes_covered"]
    _gate("B9 — July-1 CF + narrative + archetypes", passed)


def run_b6() -> None:
    _banner("B6 — Traffic Intelligence Engine")
    import joblib
    import pandas as pd
    from echo import ecosystem
    from engine import intelligence

    model = joblib.load(config.BEST_MODEL_PKL)
    features = io_utils.load_parquet(config.FEATURES_PARQUET)
    archetypes = intelligence.load_archetypes()
    config.ENGINE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- July 1 centrepiece snapshot (09:45 AM = day 1, minute 585) --
    # This is the primary demo snapshot: the exact event the B9 counterfactual
    # analyses.  Always generated regardless of cascade data.
    july1_eco: dict[int, bool] = {}
    if config.CASCADE_EVENTS_CSV.exists():
        events = pd.read_csv(config.CASCADE_EVENTS_CSV)
        july1_eco = ecosystem.cascade_propagating_map(events, day_number=1, minute_of_day=585)

    july1_snap = intelligence.analyze_snapshot(
        features, model, day_number=1, minute_of_day=585,
        archetypes=archetypes, ecosystem_state=july1_eco,
    )
    july1_path = config.ENGINE_REPORTS_DIR / "snapshot_d1_m585.json"
    july1_path.write_text(_json.dumps(july1_snap, indent=2))
    print(f"  July 1 snapshot (d1,m585): critical={july1_snap['n_critical']}  "
          f"recs={sum(len(l['recommendations']) for l in july1_snap['links'])}  -> {july1_path}")

    # -- Best cascade-event snapshot (highest downstream impact) --
    cascade_day, cascade_min = 4, 490   # fallback
    cascade_eco: dict[int, bool] = {}
    if config.CASCADE_EVENTS_CSV.exists():
        events = pd.read_csv(config.CASCADE_EVENTS_CSV)
        if len(events):
            demo = events.sort_values("n_downstream", ascending=False).iloc[0]
            cascade_day = int(demo["day_number"])
            cascade_min = int(demo["minute_of_day"])
            cascade_eco = ecosystem.cascade_propagating_map(events, cascade_day, cascade_min)

    cascade_snap = intelligence.analyze_snapshot(
        features, model, day_number=cascade_day, minute_of_day=cascade_min,
        archetypes=archetypes, ecosystem_state=cascade_eco,
    )
    cascade_path = config.ENGINE_REPORTS_DIR / f"snapshot_d{cascade_day}_m{cascade_min}.json"
    cascade_path.write_text(_json.dumps(cascade_snap, indent=2))
    n_recs = sum(len(l["recommendations"]) for l in cascade_snap["links"])
    print(f"  Cascade snapshot (d{cascade_day},m{cascade_min}): critical={cascade_snap['n_critical']}  "
          f"recs={n_recs}  -> {cascade_path}")
    if cascade_eco:
        print(f"  cascade-propagating links: {list(cascade_eco)}")

    # Gate: every recommendation that was generated must carry reasoning.
    # We check across both snapshots. If no recs fired, the gate still passes --
    # the empty-rec case means no rule conditions were met, which is valid.
    all_recs = [
        r for snap in (july1_snap, cascade_snap)
        for l in snap["links"]
        for r in l["recommendations"]
    ]
    all_reason = all(r["reasoning"].strip() for r in all_recs) if all_recs else True
    _gate("B6 — every rec has reasoning", all_reason)



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.RAW_CSV))
    args = parser.parse_args()
    raw_csv = Path(args.data)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if raw_csv.resolve() != config.RAW_CSV.resolve():
        print(f"  Copying {raw_csv.name} -> {config.RAW_CSV}")
        config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_csv, config.RAW_CSV)

    run_b1(config.RAW_CSV)
    run_b2()
    run_b3()
    run_b4()
    run_b5()
    run_b7()
    run_b8()
    run_b9()
    run_b6()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""SHAP explainability layer for UrbanPulse B5 (Bible §5 Notebook 05).

Produces all six required SHAP outputs:
  1. Beeswarm summary plot  — global, top-20 features
  2. Feature importance bar — global, mean |SHAP|
  3. Waterfall — Link 36, worst event (July 1, 09:45 AM)
  4. Waterfall — Link 37, chronic congestion case
  5. Dependence — hour
  6. Dependence — mean_occup

Also writes ``reports/shap/translations.json`` with plain-English SHAP
translations for the top-3 contributors of each waterfall.

All SHAP computation uses TreeExplainer (exact for tree models, ~O(TLD) time).
Global analysis is performed on a stratified sample (``GLOBAL_SAMPLE``) so
wall-time stays tractable on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
import io_utils
import modeling

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SHAP_DIR: Path = config.REPORTS_DIR / "shap"

# Number of rows used for global SHAP summary / importance / dependence.
# Stratified so ~13 % positive class is preserved.
GLOBAL_SAMPLE: int = 5_000

# Random seed for reproducible sampling.
_RNG_SEED: int = 42


# --------------------------------------------------------------------------- #
# Data prep helpers
# --------------------------------------------------------------------------- #

def _load_full_xy() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load features, build X for every row (train + val + test), return
    the matching y and the raw features frame for temporal look-up.

    Target-encoding is fit on the train split only (no leakage) and then
    applied to all rows so we can retrieve individual rows by timestamp.

    Returns:
        (X_all, features_shifted, y_all)
        X_all: feature matrix for every non-boundary shifted row.
        features_shifted: full shifted frame used for look-up filtering.
        y_all: binary target aligned with X_all.
    """
    feat_df = io_utils.load_parquet(config.FEATURES_PARQUET)
    shifted = modeling.shift_target(feat_df, config.HORIZON_INTERVALS)

    train, val, test = modeling.temporal_split(shifted)
    enc_train, enc_val, enc_test = modeling.target_encode_link(
        train, [train, val, test]
    )

    feat_cols = modeling.feature_columns(feat_df, leak_free=False)

    def _x(frame: pd.DataFrame, enc: pd.Series) -> pd.DataFrame:
        x = frame[feat_cols].copy()
        x["link_congestion_rate"] = enc.to_numpy()
        return x

    x_all = pd.concat(
        [_x(train, enc_train), _x(val, enc_val), _x(test, enc_test)],
        ignore_index=True,
    )
    y_all = pd.concat(
        [train["congested"], val["congested"], test["congested"]],
        ignore_index=True,
    )
    features_shifted = pd.concat(
        [train, val, test], ignore_index=True
    )
    return x_all, features_shifted, y_all


def _stratified_sample(
    x: pd.DataFrame, y: pd.Series, n: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a stratified random sample of size n."""
    rng = np.random.default_rng(seed)
    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()
    n_pos = min(int(n * y.mean()), len(pos_idx))
    n_neg = n - n_pos
    idx = np.concatenate(
        [
            rng.choice(pos_idx, n_pos, replace=False),
            rng.choice(neg_idx, n_neg, replace=False),
        ]
    )
    rng.shuffle(idx)
    return x.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def _find_row(
    x_all: pd.DataFrame,
    features_shifted: pd.DataFrame,
    link_id: int,
    day_number: int | None = None,
    minute_of_day: int | None = None,
    high_queue: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Locate a specific feature row by LINK_ID + optional filters.

    For Link 36 worst event: day_number=1, minute_of_day=585 (09:45 AM).
    For Link 37 chronic case: high_queue=True picks the row with maximum
    mean_queue_s for that link.

    Args:
        x_all:            Full X frame (aligned with features_shifted).
        features_shifted: Shifted frame carrying LINK_ID, day_number,
                          minute_of_day, mean_queue_s.
        link_id:          Road link to select.
        day_number:       Day filter (1-indexed, July 1 = 1).
        minute_of_day:    Minute-of-day filter (hour*60 + minute).
        high_queue:       If True, select the row with max mean_queue_s.

    Returns:
        (x_row, meta) where x_row is a 1-row DataFrame ready for SHAP,
        and meta carries human-readable context.
    """
    mask = features_shifted["LINK_ID"] == link_id
    if day_number is not None:
        mask &= features_shifted["day_number"] == day_number
    if minute_of_day is not None:
        mask &= features_shifted["minute_of_day"] == minute_of_day

    sub = features_shifted[mask]
    if sub.empty:
        raise ValueError(
            f"No rows found: LINK_ID={link_id}, day={day_number}, "
            f"minute_of_day={minute_of_day}"
        )

    if high_queue:
        idx = sub["mean_queue_s"].idxmax()
    else:
        idx = sub.index[0]

    meta: dict[str, Any] = {
        "link_id": link_id,
        "day_number": int(features_shifted.at[idx, "day_number"]),
        "minute_of_day": int(features_shifted.at[idx, "minute_of_day"]),
        "mean_queue_s": float(features_shifted.at[idx, "mean_queue_s"]),
        "mean_occup": float(features_shifted.at[idx, "mean_occup"]),
        "congested": int(features_shifted.at[idx, "congested"]),
    }

    # Map from features_shifted positional index → x_all position.
    x_row = x_all.iloc[[features_shifted.index.get_loc(idx)]].copy()
    return x_row, meta


# --------------------------------------------------------------------------- #
# SHAP computation
# --------------------------------------------------------------------------- #

def compute_shap_values(
    model: Any,
    x_sample: pd.DataFrame,
) -> shap.Explanation:
    """Run TreeExplainer on x_sample and return an Explanation object.

    Args:
        model:    Fitted tree model (CatBoost, XGBoost, etc.).
        x_sample: Feature matrix for the rows to explain.

    Returns:
        shap.Explanation with .values, .base_values, .data, .feature_names.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer(x_sample)
    # Some explainers return a 3-D array [rows, features, classes] for
    # binary classifiers. Squeeze to the positive-class slice.
    if len(sv.values.shape) == 3:
        sv.values = sv.values[:, :, 1]
        sv.base_values = sv.base_values[:, 1]
    return sv


# --------------------------------------------------------------------------- #
# Plot 1 — Beeswarm summary
# --------------------------------------------------------------------------- #

def plot_beeswarm(sv: shap.Explanation, out_dir: Path) -> Path:
    """Global beeswarm summary plot — top-20 features (Bible output 1).

    Args:
        sv:      SHAP Explanation for the sample.
        out_dir: Directory to write the PNG.

    Returns:
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv.values,
        sv.data,
        feature_names=sv.feature_names,
        max_display=20,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Beeswarm — Top 20 Features (Global)", fontsize=12)
    path = out_dir / "01_beeswarm.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


# --------------------------------------------------------------------------- #
# Plot 2 — Feature importance bar
# --------------------------------------------------------------------------- #

def plot_importance_bar(sv: shap.Explanation, out_dir: Path) -> Path:
    """Mean |SHAP| feature importance bar chart (Bible output 2).

    Args:
        sv:      SHAP Explanation for the sample.
        out_dir: Directory to write the PNG.

    Returns:
        Path to the saved PNG.
    """
    mean_abs = np.abs(sv.values).mean(axis=0)
    imp = pd.Series(mean_abs, index=sv.feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    imp.tail(20).plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance — Mean |SHAP|", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    path = out_dir / "02_importance_bar.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


# --------------------------------------------------------------------------- #
# Plot 3 & 4 — Waterfall plots
# --------------------------------------------------------------------------- #

def plot_waterfall(
    sv_row: shap.Explanation,
    title: str,
    filename: str,
    out_dir: Path,
    max_display: int = 12,
) -> Path:
    """SHAP waterfall plot for a single row (Bible outputs 3 & 4).

    Args:
        sv_row:      Single-row SHAP Explanation (already sliced to 1 row).
        title:       Chart title string.
        filename:    Output file name (e.g. ``03_waterfall_link36.png``).
        out_dir:     Directory to write the PNG.
        max_display: Number of features to show before collapsing the rest.

    Returns:
        Path to the saved PNG.
    """
    shap.waterfall_plot(sv_row[0], max_display=max_display, show=False)
    plt.title(title, fontsize=11)
    path = out_dir / filename
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


# --------------------------------------------------------------------------- #
# Plot 5 & 6 — Dependence plots
# --------------------------------------------------------------------------- #

def plot_dependence(
    sv: shap.Explanation,
    feature: str,
    filename: str,
    out_dir: Path,
) -> Path:
    """SHAP dependence plot for one feature (Bible outputs 5 & 6).

    Coloured by interaction with the feature that has the highest interaction
    effect (auto-selected by SHAP).

    Args:
        sv:       SHAP Explanation for the sample.
        feature:  Feature name to plot on the X axis.
        filename: Output file name.
        out_dir:  Directory to write the PNG.

    Returns:
        Path to the saved PNG.
    """
    feat_names = list(sv.feature_names)
    if feature not in feat_names:
        raise ValueError(f"Feature '{feature}' not in SHAP explanation.")
    feat_idx = feat_names.index(feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        sv.values,
        sv.data,
        feature_names=feat_names,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP Dependence — {feature}", fontsize=12)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


# --------------------------------------------------------------------------- #
# Human-readable SHAP translations
# --------------------------------------------------------------------------- #

# Plain-English templates keyed by feature name (partial match supported).
_TRANSLATIONS: dict[str, str] = {
    "mean_occup": (
        "Road occupancy was {val:.0%}, {direction} congestion risk "
        "(high fill = less room for more vehicles)."
    ),
    "mean_queue_s": (
        "Current average queue delay was {val:.0f} s, {direction} "
        "congestion risk (longer queues signal an already-stressed road)."
    ),
    "hour": (
        "This reading falls at hour {val:.0f} — "
        "{direction} congestion risk (peak-hour multiplier)."
    ),
    "link_congestion_rate": (
        "This road's historical congestion rate is {val:.1%}, "
        "{direction} congestion risk by {mag:.0%} above the network average."
    ),
    "is_am_peak": (
        "The AM-peak flag is {flag_str}, {direction} congestion risk "
        "(8–10 AM is the hardest demand period in the dataset)."
    ),
    "congestion_index": (
        "The composite congestion index was {val:.2f}, "
        "{direction} congestion risk."
    ),
    "minute_of_day": (
        "Minute-of-day position was {val:.0f} (≈ {hour_approx:02d}:{min_approx:02d}), "
        "{direction} congestion risk."
    ),
    "total_vehs": (
        "Total vehicle count was {val:.0f}, {direction} congestion risk "
        "(volume pressure on the road)."
    ),
    "max_occup": (
        "Worst-lane occupancy was {val:.0%}, {direction} congestion risk "
        "(even one fully saturated lane can block the whole road)."
    ),
    "mean_speed_kmh": (
        "Average speed was {val:.1f} km/h — {direction} congestion risk "
        "(slower = tighter stop-go conditions)."
    ),
}


def _direction(shap_val: float) -> str:
    return "significantly increasing" if shap_val > 0 else "reducing"


def _translate_feature(
    feature: str, shap_val: float, feature_val: float
) -> str:
    """Build a plain-English sentence for one (feature, shap_value, value) triple."""
    direction = _direction(shap_val)
    mag = abs(shap_val)

    # Find matching template (exact, then partial).
    template = _TRANSLATIONS.get(feature)
    if template is None:
        for key, tmpl in _TRANSLATIONS.items():
            if key in feature:
                template = tmpl
                break

    if template is None:
        return (
            f"`{feature}` = {feature_val:.3g} — {direction} "
            f"congestion risk (SHAP {shap_val:+.3f})."
        )

    kwargs: dict[str, Any] = {
        "val": feature_val,
        "direction": direction,
        "mag": mag,
    }
    if "flag_str" in template:
        kwargs["flag_str"] = "ON" if feature_val == 1 else "OFF"
    if "hour_approx" in template:
        kwargs["hour_approx"] = int(feature_val) // 60
        kwargs["min_approx"] = int(feature_val) % 60

    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        return (
            f"`{feature}` = {feature_val:.3g} — {direction} "
            f"congestion risk (SHAP {shap_val:+.3f})."
        )


def translate_waterfall(
    sv_row: shap.Explanation,
    label: str,
    meta: dict[str, Any],
    top_n: int = 3,
) -> dict[str, Any]:
    """Produce a structured plain-English SHAP translation for a single row.

    Args:
        sv_row: Single-row SHAP Explanation.
        label:  Human label for this analysis (e.g. ``"Link 36 — July 1 09:45 AM"``).
        meta:   Context dict from ``_find_row`` (queue, occup, day, minute).
        top_n:  Number of top-contributing features to translate.

    Returns:
        Dict with ``label``, ``meta``, ``base_value``, ``prediction_log_odds``,
        and ``top_features`` list of plain-English sentences.
    """
    shap_vals = sv_row.values[0]
    feat_vals = sv_row.data[0]
    feat_names = list(sv_row.feature_names)

    # Rank by absolute contribution.
    order = np.argsort(np.abs(shap_vals))[::-1]

    top_features = []
    for rank, fi in enumerate(order[:top_n]):
        sentence = _translate_feature(feat_names[fi], shap_vals[fi], feat_vals[fi])
        top_features.append(
            {
                "rank": rank + 1,
                "feature": feat_names[fi],
                "value": float(feat_vals[fi]),
                "shap": float(shap_vals[fi]),
                "plain_english": sentence,
            }
        )

    return {
        "label": label,
        "meta": meta,
        "base_value": float(sv_row.base_values[0]),
        "prediction_log_odds": float(shap_vals.sum() + sv_row.base_values[0]),
        "top_features": top_features,
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def run() -> dict[str, Any]:
    """Full B5 pipeline.

    Loads the best model and features, computes SHAP values, writes all six
    required plots plus a translations JSON.

    Returns:
        Summary dict for the B5 gate check.
    """
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load model + data
    # ------------------------------------------------------------------ #
    print("Loading best model ...")
    model = joblib.load(config.BEST_MODEL_PKL)

    print("Building full feature matrix ...")
    x_all, features_shifted, y_all = _load_full_xy()

    print(f"  total rows available: {len(x_all)}")

    # ------------------------------------------------------------------ #
    # 2. Stratified sample for global analysis
    # ------------------------------------------------------------------ #
    x_sample, y_sample = _stratified_sample(x_all, y_all, GLOBAL_SAMPLE, _RNG_SEED)
    print(
        f"  global sample: {len(x_sample)} rows, "
        f"pos_rate={y_sample.mean():.3f}"
    )

    # ------------------------------------------------------------------ #
    # 3. Compute SHAP — global sample
    # ------------------------------------------------------------------ #
    print("Computing SHAP values (global sample) ...")
    sv_sample = compute_shap_values(model, x_sample)

    # ------------------------------------------------------------------ #
    # 4. Plot 1 — Beeswarm
    # ------------------------------------------------------------------ #
    print("  [1/6] Beeswarm ...")
    p1 = plot_beeswarm(sv_sample, SHAP_DIR)

    # ------------------------------------------------------------------ #
    # 5. Plot 2 — Importance bar
    # ------------------------------------------------------------------ #
    print("  [2/6] Importance bar ...")
    p2 = plot_importance_bar(sv_sample, SHAP_DIR)

    # ------------------------------------------------------------------ #
    # 6. Local rows: Link 36 (July 1, 09:45) and Link 37 (chronic)
    # ------------------------------------------------------------------ #
    # July 1 = day_number = 1; 09:45 AM → minute_of_day = 9*60+45 = 585
    print("  Locating Link 36 worst event (day 1, 09:45 AM) ...")
    x_36, meta_36 = _find_row(
        x_all, features_shifted,
        link_id=36, day_number=1, minute_of_day=585
    )

    print("  Locating Link 37 chronic congestion case ...")
    x_37, meta_37 = _find_row(
        x_all, features_shifted,
        link_id=37, high_queue=True
    )

    # Compute local SHAP for both rows.
    print("Computing SHAP values (local rows) ...")
    sv_36 = compute_shap_values(model, x_36)
    sv_37 = compute_shap_values(model, x_37)

    # ------------------------------------------------------------------ #
    # 7. Plot 3 — Waterfall Link 36
    # ------------------------------------------------------------------ #
    print("  [3/6] Waterfall Link 36 ...")
    hour_36 = meta_36["minute_of_day"] // 60
    min_36 = meta_36["minute_of_day"] % 60
    title_36 = (
        f"SHAP Waterfall — Link 36  |  "
        f"Day {meta_36['day_number']}, {hour_36:02d}:{min_36:02d}  |  "
        f"Queue {meta_36['mean_queue_s']:.0f}s"
    )
    p3 = plot_waterfall(sv_36, title_36, "03_waterfall_link36.png", SHAP_DIR)

    # ------------------------------------------------------------------ #
    # 8. Plot 4 — Waterfall Link 37
    # ------------------------------------------------------------------ #
    print("  [4/6] Waterfall Link 37 ...")
    hour_37 = meta_37["minute_of_day"] // 60
    min_37 = meta_37["minute_of_day"] % 60
    title_37 = (
        f"SHAP Waterfall — Link 37  |  "
        f"Day {meta_37['day_number']}, {hour_37:02d}:{min_37:02d}  |  "
        f"Queue {meta_37['mean_queue_s']:.0f}s (chronic)"
    )
    p4 = plot_waterfall(sv_37, title_37, "04_waterfall_link37.png", SHAP_DIR)

    # ------------------------------------------------------------------ #
    # 9. Plot 5 — Dependence: hour
    # ------------------------------------------------------------------ #
    print("  [5/6] Dependence: hour ...")
    p5 = plot_dependence(sv_sample, "hour", "05_dependence_hour.png", SHAP_DIR)

    # ------------------------------------------------------------------ #
    # 10. Plot 6 — Dependence: mean_occup
    # ------------------------------------------------------------------ #
    print("  [6/6] Dependence: mean_occup ...")
    p6 = plot_dependence(sv_sample, "mean_occup", "06_dependence_mean_occup.png", SHAP_DIR)

    # ------------------------------------------------------------------ #
    # 11. Human-readable translations
    # ------------------------------------------------------------------ #
    print("Building plain-English translations ...")
    trans_36 = translate_waterfall(
        sv_36,
        label=f"Link 36 — Day {meta_36['day_number']} {hour_36:02d}:{min_36:02d} "
              f"(worst event, queue {meta_36['mean_queue_s']:.0f}s)",
        meta=meta_36,
    )
    trans_37 = translate_waterfall(
        sv_37,
        label=f"Link 37 — Day {meta_37['day_number']} {hour_37:02d}:{min_37:02d} "
              f"(chronic case, queue {meta_37['mean_queue_s']:.0f}s)",
        meta=meta_37,
    )
    translations = {"link_36": trans_36, "link_37": trans_37}
    trans_path = SHAP_DIR / "translations.json"
    trans_path.write_text(json.dumps(translations, indent=2))

    # ------------------------------------------------------------------ #
    # 12. Gate summary
    # ------------------------------------------------------------------ #
    plots = [p1, p2, p3, p4, p5, p6]
    all_exist = all(p.exists() for p in plots)
    summary = {
        "plots_produced": [str(p.name) for p in plots],
        "all_plots_exist": all_exist,
        "global_sample_rows": int(len(x_sample)),
        "global_sample_pos_rate": float(y_sample.mean()),
        "link36_meta": meta_36,
        "link37_meta": meta_37,
        "translations_path": str(trans_path),
    }
    (SHAP_DIR / "b5_gate.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    result = run()
    print("\n=== B5 SHAP GATE ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    passed = (
        result["all_plots_exist"]
        and len(result["plots_produced"]) == 6
    )
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'}")

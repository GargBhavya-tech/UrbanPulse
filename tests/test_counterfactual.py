"""Tests for ECHO Stage C — Counterfactual Intervention Engine (Bible §7 Stage C)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from echo.counterfactual import (
    SCM,
    StructuralEquation,
    _intervention_for,
    build_scm_library,
    july1_counterfactual,
    run_all_counterfactuals,
    INTERVENTION_LIBRARY,
    JULY1_LINK,
    JULY1_DAY,
    JULY1_PEAK_MINUTE,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _synthetic_link(n_rows: int = 500, rng_seed: int = 0) -> pd.DataFrame:
    """Create a realistic synthetic feature frame for one link."""
    rng = np.random.default_rng(rng_seed)
    lane6_active = rng.integers(0, 2, n_rows).astype(float)
    total_vehs = rng.integers(300, 1200, n_rows).astype(float)
    hour = (np.arange(n_rows) % 24).astype(float)
    is_am_peak = ((hour >= 8) & (hour <= 9)).astype(float)
    mean_occup = np.clip(0.3 + 0.15 * total_vehs / 1200 - 0.05 * lane6_active + rng.normal(0, 0.05, n_rows), 0, 1)
    mean_speed_div = rng.uniform(0, 8, n_rows)
    mean_queue_s = np.maximum(
        50 + 400 * mean_occup + 20 * mean_speed_div - 30 * lane6_active + rng.normal(0, 30, n_rows),
        0.0,
    )
    return pd.DataFrame({
        "LINK_ID": 36,
        "day_number": (np.arange(n_rows) // 288) + 1,
        "minute_of_day": (np.arange(n_rows) * 5) % 1440,
        "hour": hour,
        "total_vehs": total_vehs,
        "lane6_active": lane6_active,
        "is_am_peak": is_am_peak,
        "mean_occup": mean_occup,
        "mean_speed_div": mean_speed_div,
        "congestion_index": mean_occup * 0.6,
        "road_health_score": np.clip(100 - mean_queue_s / 10, 0, 100),
        "mean_queue_s": mean_queue_s,
        "total_vehs": total_vehs,
    })


def _synthetic_multi_link(n_links: int = 5, n_rows_per_link: int = 200) -> pd.DataFrame:
    """Synthetic frame with multiple LINK_IDs."""
    frames = []
    for i in range(n_links):
        df = _synthetic_link(n_rows_per_link, rng_seed=i)
        df["LINK_ID"] = i + 1
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# StructuralEquation tests
# --------------------------------------------------------------------------- #

def test_structural_equation_fit_and_predict() -> None:
    """OLS fits cleanly on well-conditioned synthetic data."""
    df = _synthetic_link()
    eq = StructuralEquation("mean_occup", ["intervention_value", "total_vehs", "hour"])
    df2 = df.copy()
    df2["intervention_value"] = df2["lane6_active"]
    eq.fit(df2)
    assert eq.coef_ is not None
    assert len(eq.coef_) == 3
    assert 0.0 <= eq.r2_ <= 1.0
    # Predict shape
    X = df2[["intervention_value", "total_vehs", "hour"]].to_numpy()
    y_hat = eq.predict(X)
    assert y_hat.shape == (len(df2),)


def test_structural_equation_r2_positive() -> None:
    """R² should be positive on data with genuine signal."""
    df = _synthetic_link()
    df["intervention_value"] = df["lane6_active"]
    eq = StructuralEquation("mean_occup", ["intervention_value", "total_vehs", "hour"])
    eq.fit(df)
    assert eq.r2_ > 0.0, f"Expected positive R², got {eq.r2_}"


def test_structural_equation_raises_on_missing_col() -> None:
    df = _synthetic_link()
    eq = StructuralEquation("mean_occup", ["nonexistent_col"])
    with pytest.raises(ValueError, match="Missing columns"):
        eq.fit(df)


def test_structural_equation_as_dict_schema() -> None:
    df = _synthetic_link()
    df["intervention_value"] = df["lane6_active"]
    eq = StructuralEquation("mean_occup", ["intervention_value", "total_vehs", "hour"])
    eq.fit(df)
    d = eq.as_dict()
    assert set(d.keys()) >= {"target", "predictors", "intercept", "coefficients", "r2"}
    assert d["target"] == "mean_occup"
    assert len(d["coefficients"]) == 3


def test_structural_equation_predict_before_fit_raises() -> None:
    eq = StructuralEquation("mean_occup", ["intervention_value"])
    with pytest.raises(RuntimeError, match="not been fitted"):
        eq.predict(np.array([[1.0]]))


# --------------------------------------------------------------------------- #
# SCM tests
# --------------------------------------------------------------------------- #

def test_scm_fit_and_do() -> None:
    """SCM fits both stages and do() returns non-negative queue values."""
    df = _synthetic_link()
    scm = SCM(link_id=36, archetype="Landmine")
    scm.fit(df, intervention_col="lane6_active")
    assert scm._fitted
    q = scm.do(df.head(50), "lane6_active", intervention_value=1.0)
    assert q.shape == (50,)
    assert (q >= 0).all(), "Queue delay must be non-negative"


def test_scm_do_intervention_reduces_queue() -> None:
    """Activating lane6 should reduce (or not increase) mean queue in the model."""
    df = _synthetic_link()
    scm = SCM(link_id=36, archetype="Landmine")
    scm.fit(df, intervention_col="lane6_active")
    q0 = scm.do(df, "lane6_active", 0.0).mean()
    q1 = scm.do(df, "lane6_active", 1.0).mean()
    # The synthetic data is designed so lane6 reduces occupancy → reduces queue
    assert q1 < q0, f"Expected q1({q1:.1f}) < q0({q0:.1f})"


def test_scm_ate_sign_and_structure() -> None:
    """ATE dict has correct keys and sensible sign."""
    df = _synthetic_link()
    scm = SCM(link_id=36, archetype="Saturator")
    scm.fit(df, intervention_col="lane6_active")
    effect = scm.ate(df, "lane6_active", value_treated=1.0, value_control=0.0)
    assert set(effect.keys()) == {"ate", "queue_treated", "queue_control"}
    assert effect["queue_treated"] >= 0
    assert effect["queue_control"] >= 0
    # ATE = treated − control; we expect negative ATE (intervention reduces queue)
    assert effect["ate"] < 0, f"Expected negative ATE, got {effect['ate']:.2f}"


def test_scm_do_before_fit_raises() -> None:
    df = _synthetic_link()
    scm = SCM(link_id=1, archetype="Commuter")
    with pytest.raises(RuntimeError, match="SCM must be fitted"):
        scm.do(df, "lane6_active", 1.0)


# --------------------------------------------------------------------------- #
# Intervention library tests
# --------------------------------------------------------------------------- #

def test_intervention_library_covers_all_archetypes() -> None:
    """Every expected archetype has an intervention defined."""
    for arch in ["Landmine", "Chronic", "Saturator", "Ghost", "Commuter", "Chameleon", "Unknown"]:
        col, v_treated, v_control, desc = _intervention_for(arch)
        assert col in {"lane6_active", "is_am_peak"}, f"Unexpected col for {arch}: {col}"
        assert isinstance(desc, str) and len(desc) > 5


def test_intervention_for_unknown_defaults_gracefully() -> None:
    col, v_treated, v_control, desc = _intervention_for("NotARealArchetype")
    assert col == "lane6_active"


# --------------------------------------------------------------------------- #
# build_scm_library tests
# --------------------------------------------------------------------------- #

def test_build_scm_library_fits_all_links() -> None:
    """All links with sufficient data should get a fitted SCM."""
    df = _synthetic_multi_link(n_links=5, n_rows_per_link=200)
    archetypes = {1: "Commuter", 2: "Saturator", 3: "Ghost", 4: "Chronic", 5: "Chameleon"}
    scms = build_scm_library(df, archetypes)
    assert len(scms) == 5
    for link_id, scm in scms.items():
        assert scm._fitted, f"SCM for link {link_id} not fitted"


def test_build_scm_library_uses_archetype_intervention() -> None:
    """Commuter archetype uses is_am_peak as intervention column."""
    df = _synthetic_multi_link(n_links=2, n_rows_per_link=300)
    df.loc[df["LINK_ID"] == 1, "LINK_ID"] = 99
    df.loc[df["LINK_ID"] == 2, "LINK_ID"] = 99  # merge to link 99
    df["LINK_ID"] = 99
    archetypes = {99: "Commuter"}
    scms = build_scm_library(df, archetypes)
    # Should succeed with is_am_peak as intervention
    assert 99 in scms
    assert scms[99]._fitted


# --------------------------------------------------------------------------- #
# run_all_counterfactuals tests
# --------------------------------------------------------------------------- #

def test_run_all_counterfactuals_output_schema() -> None:
    """Every result dict must have the required keys."""
    df = _synthetic_multi_link(n_links=3, n_rows_per_link=150)
    archetypes = {1: "Commuter", 2: "Saturator", 3: "Ghost"}
    scms = build_scm_library(df, archetypes)
    results = run_all_counterfactuals(df, scms, archetypes)
    assert len(results) == 3
    required_keys = {"link_id", "archetype", "ate_seconds", "queue_reduction_pct"}
    for r in results:
        if "error" not in r:
            assert required_keys <= set(r.keys()), f"Missing keys in result: {r}"


def test_run_all_counterfactuals_sorted_by_reduction() -> None:
    """Results should be sorted descending by queue_reduction_s."""
    df = _synthetic_multi_link(n_links=4, n_rows_per_link=200)
    archetypes = {i: "Saturator" for i in range(1, 5)}
    scms = build_scm_library(df, archetypes)
    results = [r for r in run_all_counterfactuals(df, scms, archetypes) if "error" not in r]
    reductions = [r["queue_reduction_s"] for r in results]
    assert reductions == sorted(reductions, reverse=True)


# --------------------------------------------------------------------------- #
# july1_counterfactual tests (synthetic proxy)
# --------------------------------------------------------------------------- #

def _july1_synthetic_features() -> pd.DataFrame:
    """Synthetic frame mimicking Link 36 July 1 data structure."""
    rng = np.random.default_rng(42)
    rows = []
    for minute in range(0, 1440, 5):
        hour_val = minute // 60
        lane6 = 0.0
        total_v = rng.integers(800, 1200)
        occup = min(0.6 + (0.3 if 540 <= minute <= 600 else 0.0) + rng.normal(0, 0.03), 1.0)
        speed_div = rng.uniform(1, 10)
        queue = max(100 + 600 * occup + 20 * speed_div - 30 * lane6 + rng.normal(0, 50), 0)
        rows.append({
            "LINK_ID": JULY1_LINK,
            "day_number": JULY1_DAY,
            "minute_of_day": minute,
            "hour": hour_val,
            "total_vehs": float(total_v),
            "lane6_active": lane6,
            "is_am_peak": float(8 <= hour_val <= 9),
            "mean_occup": float(occup),
            "mean_speed_div": float(speed_div),
            "congestion_index": float(occup * 0.6),
            "road_health_score": float(max(100 - queue / 10, 0)),
            "mean_queue_s": float(queue),
        })
    return pd.DataFrame(rows)


def test_july1_counterfactual_structure(tmp_path: Path) -> None:
    """July 1 CF returns a result dict with all required fields."""
    df = _july1_synthetic_features()
    scm = SCM(link_id=JULY1_LINK, archetype="Landmine")
    scm.fit(df, intervention_col="lane6_active")

    result = july1_counterfactual(
        features=df,
        scm=scm,
        causal_graph_json=tmp_path / "causal_graph.json",  # doesn't exist → fallback
        cascade_events_csv=tmp_path / "cascade_events.csv",  # doesn't exist → fallback
    )
    required = {
        "link_id", "day_number", "intervention_minute", "peak_minute",
        "observed_queue_s", "counterfactual_queue_s", "queue_reduction_pct",
        "narrative", "causal_mechanism", "cascade_source_confirmed", "estimation_mode",
    }
    assert required <= set(result.keys()), f"Missing keys: {required - set(result.keys())}"


def test_july1_narrative_non_empty(tmp_path: Path) -> None:
    """Narrative must be a non-empty string containing key facts."""
    df = _july1_synthetic_features()
    scm = SCM(link_id=JULY1_LINK, archetype="Landmine")
    scm.fit(df, intervention_col="lane6_active")
    result = july1_counterfactual(df, scm, tmp_path / "g.json", tmp_path / "c.csv")
    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 100
    assert "Counterfactual Analysis" in result["narrative"]
    assert "Road 36" in result["narrative"]


def test_july1_counterfactual_reduction_positive(tmp_path: Path) -> None:
    """The intervention should reduce or maintain queue (non-negative reduction expected)."""
    df = _july1_synthetic_features()
    scm = SCM(link_id=JULY1_LINK, archetype="Landmine")
    scm.fit(df, intervention_col="lane6_active")
    result = july1_counterfactual(df, scm, tmp_path / "g.json", tmp_path / "c.csv")
    # The synthetic data is designed so lane6=1 reduces queue
    assert result["counterfactual_queue_s"] <= result["modeled_baseline_queue_s"] + 1.0, (
        f"CF queue {result['counterfactual_queue_s']:.1f} exceeds baseline "
        f"{result['modeled_baseline_queue_s']:.1f}"
    )


# --------------------------------------------------------------------------- #
# Integration test — requires built artifacts
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not config.FEATURES_PARQUET.exists(),
    reason="features.parquet not built -- run pipeline first",
)
def test_run_produces_artifacts() -> None:
    """Full run() must produce all four output artifacts and pass the gate."""
    from echo.counterfactual import run
    out = run()
    assert config.COUNTERFACTUAL_RESULTS_JSON.exists()
    assert (config.ECHO_REPORTS_DIR / "scm_graph.png").exists()
    assert (config.ECHO_REPORTS_DIR / "july1_counterfactual.txt").exists()
    assert (config.ECHO_REPORTS_DIR / "scm_coefficients.json").exists()
    assert out["july1_ok"], "July 1 counterfactual failed"
    assert out["narrative_ok"], "Narrative was not produced"
    assert out["n_scms_fitted"] > 0


# --------------------------------------------------------------------------- #
# Policy-simulation fallback tests
# --------------------------------------------------------------------------- #

def _zero_variance_link() -> pd.DataFrame:
    """Synthetic link where intervention has zero variance (always 0)."""
    rng = np.random.default_rng(7)
    n = 200
    total_vehs = rng.integers(600, 1200, n).astype(float)
    hour = (np.arange(n) % 24).astype(float)
    mean_occup = np.clip(0.5 + 0.1 * total_vehs / 1200 + rng.normal(0, 0.05, n), 0, 1)
    mean_speed_div = rng.uniform(1, 6, n)
    mean_queue_s = np.maximum(200 + 300 * mean_occup + rng.normal(0, 30, n), 0)
    return pd.DataFrame({
        "LINK_ID": 36,
        "day_number": 1,
        "minute_of_day": (np.arange(n) * 5) % 1440,
        "hour": hour,
        "total_vehs": total_vehs,
        "lane6_active": np.zeros(n),   # ALWAYS 0 -- zero variance
        "is_am_peak": ((hour >= 8) & (hour <= 9)).astype(float),
        "mean_occup": mean_occup,
        "mean_speed_div": mean_speed_div,
        "congestion_index": mean_occup * 0.6,
        "road_health_score": np.clip(100 - mean_queue_s / 10, 0, 100),
        "mean_queue_s": mean_queue_s,
    })


def test_policy_simulation_mode_triggers() -> None:
    """When intervention has zero variance, SCM must switch to policy_simulation."""
    df = _zero_variance_link()
    scm = SCM(link_id=36, archetype="Chronic")
    scm.fit(df, intervention_col="lane6_active")
    assert scm.estimation_mode == "policy_simulation", (
        f"Expected policy_simulation mode, got {scm.estimation_mode}"
    )


def test_policy_simulation_coef_is_negative() -> None:
    """Policy-simulation Stage-1 coefficient must be negative (intervention reduces occup)."""
    df = _zero_variance_link()
    scm = SCM(link_id=36, archetype="Chronic")
    scm.fit(df, intervention_col="lane6_active")
    assert scm.eq1.coef_ is not None
    intervention_coef = scm.eq1.coef_[0]
    assert intervention_coef < 0, (
        f"Expected negative intervention coef, got {intervention_coef:.4f}"
    )


def test_policy_simulation_do_reduces_queue() -> None:
    """Even with zero-variance intervention, do(1) should produce lower queue than do(0)."""
    df = _zero_variance_link()
    scm = SCM(link_id=36, archetype="Chronic")
    scm.fit(df, intervention_col="lane6_active")
    q0 = scm.do(df, "lane6_active", 0.0).mean()
    q1 = scm.do(df, "lane6_active", 1.0).mean()
    assert q1 < q0, f"Expected q1({q1:.1f}) < q0({q0:.1f}) in policy-sim mode"


def test_run_all_counterfactuals_has_estimation_mode() -> None:
    """All sweep results must include estimation_mode field."""
    df = _synthetic_multi_link(n_links=3, n_rows_per_link=150)
    archetypes = {1: "Commuter", 2: "Saturator", 3: "Ghost"}
    scms = build_scm_library(df, archetypes)
    results = run_all_counterfactuals(df, scms, archetypes)
    for r in results:
        if "error" not in r:
            assert "estimation_mode" in r, f"Missing estimation_mode in {r}"
            assert r["estimation_mode"] in ("ols", "policy_simulation")


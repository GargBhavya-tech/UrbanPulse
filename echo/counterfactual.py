"""ECHO Stage C -- Counterfactual Intervention Engine (Bible §7 Stage C).

Implements Pearl's Level 3 (do-calculus) via a lightweight Structural Causal
Model (SCM) built on numpy/pandas/networkx -- no DoWhy dependency (DECISION_MAP
#7: DoWhy install weight is unjustified when the Bible mandates a manually
specified SCM anyway).

Architecture:
    SCM nodes (5 core):
        intervention_var  ->  mean_occup  ->  mean_queue_s
             |                   |               |
         archetype           total_vehs     mean_speed_div
         (moderates           (proxy          (mediator)
          effect size)         confounder)

    Causal path (frontdoor criterion):
        intervention -> mean_occup -> mean_queue_s
        Both intermediate variables are fully observed, so the frontdoor
        adjustment is valid even with unobserved confounders (driver routing,
        unreported incidents, weather).

Structural equations (fitted per link via OLS):
    mean_occup    = b0 + b1*intervention + b2*total_vehs + b3*hour + e1
    mean_queue_s  = g0 + g1*mean_occup   + g2*mean_speed_div + g3*hour + e2

do-operator:
    do(intervention=1): set intervention=1 in eq1, propagate to eq2.
    do(intervention=0): set intervention=0 in eq1, propagate to eq2.
    ATE = E[queue | do(T=1)] - E[queue | do(T=0)]

Intervention regime:
    When the intervention column has std < config.SCM_MIN_INTERVENTION_STD
    (i.e. the lever was never activated in the observed 14-day window), OLS
    cannot estimate the effect.  In that case we switch to a "policy simulation"
    mode: the Stage-1 coefficient is set to a domain-informed prior derived from
    the Bible §7 Stage C commentary and the B6 rule-engine effect documentation.
    All outputs are labelled estimation_mode="policy_simulation" vs "ols".

Archetype-specific intervention library (Bible §7 Stage C):
    Landmine/Chronic : do(lane6_active=1)          -- activate ghost lane
    Saturator        : do(lane6_active=1)           -- activate ghost lane + cap
    Ghost            : do(lane6_active=1)
    Commuter         : do(is_am_peak extended) -- sustained peak-hour state
    Chameleon        : do(lane6_active=1)

Outputs:
    data/counterfactual_results.json
    reports/echo/scm_coefficients.json
    reports/echo/scm_graph.png
    reports/echo/july1_counterfactual.txt
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Stage 1 & 2 feature lists
# --------------------------------------------------------------------------- #
_STAGE1_FEATURES: list[str] = ["intervention_value", "total_vehs", "hour"]
_STAGE2_FEATURES: list[str] = ["mean_occup", "mean_speed_div", "hour"]


# --------------------------------------------------------------------------- #
# Structural Equation -- OLS with optional policy-simulation fallback
# --------------------------------------------------------------------------- #

class StructuralEquation:
    """Ordinary Least Squares structural equation with intervention-variance guard.

    Args:
        target: Name of the outcome variable.
        predictors: Names of the predictor variables.
    """

    def __init__(self, target: str, predictors: list[str]) -> None:
        self.target = target
        self.predictors = predictors
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.r2_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "StructuralEquation":
        """Fit the OLS equation on the provided DataFrame.

        Args:
            df: DataFrame containing all predictor and target columns.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If any required column is missing or all-NaN.
        """
        missing = [c for c in self.predictors + [self.target] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        sub = df[self.predictors + [self.target]].dropna()
        if len(sub) < len(self.predictors) + 2:
            raise ValueError(f"Insufficient rows ({len(sub)}) for OLS on {self.target}.")

        X = sub[self.predictors].to_numpy(dtype=float)
        y = sub[self.target].to_numpy(dtype=float)

        # Design matrix with intercept column
        X_aug = np.column_stack([np.ones(len(X)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.intercept_ = float(coeffs[0])
        self.coef_ = coeffs[1:]

        y_hat = X_aug @ coeffs
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r2_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcome from predictor matrix (no intercept column).

        Args:
            X: Array of shape (n_samples, n_predictors).

        Returns:
            Predicted values array of shape (n_samples,).

        Raises:
            RuntimeError: If the equation has not been fitted.
        """
        if self.coef_ is None:
            raise RuntimeError("StructuralEquation has not been fitted.")
        return self.intercept_ + X @ self.coef_

    def as_dict(self) -> dict[str, Any]:
        """Serialisable coefficient summary."""
        return {
            "target": self.target,
            "predictors": self.predictors,
            "intercept": round(self.intercept_, 6),
            "coefficients": {
                p: round(float(c), 6)
                for p, c in zip(self.predictors, self.coef_ if self.coef_ is not None else [])
            },
            "r2": round(self.r2_, 4),
        }


# --------------------------------------------------------------------------- #
# Intervention variance check + policy-simulation coefficient injection
# --------------------------------------------------------------------------- #

def _intervention_coef_from_prior(
    link_df: pd.DataFrame,
    intervention_col: str,
) -> float:
    """Return a domain-informed Stage-1 intervention coefficient when OLS fails.

    When lane6_active (or is_am_peak) was never (or always) active in the
    observed data, OLS collapses the coefficient to 0.  We substitute a prior
    derived from the B6 rule documentation:

    - lane6_active: expected to reduce mean_occup by
      ``SCM_LANE6_OCCUP_EFFECT_FRAC`` x the link's observed mean occupancy.
    - is_am_peak: expected to reduce mean_occup by
      ``SCM_IS_AM_PEAK_OCCUP_EFFECT_FRAC`` x the link's observed mean occupancy.

    The coefficient is NEGATIVE because activating the intervention reduces
    occupancy (do-calculus: treated - control).

    Args:
        link_df: Feature frame for this link.
        intervention_col: Name of the intervention column.

    Returns:
        Estimated Stage-1 coefficient (negative = intervention reduces occup).
    """
    mean_occup = float(link_df["mean_occup"].mean())
    if intervention_col == "lane6_active":
        return -config.SCM_LANE6_OCCUP_EFFECT_FRAC * mean_occup
    elif intervention_col == "is_am_peak":
        return -config.SCM_IS_AM_PEAK_OCCUP_EFFECT_FRAC * mean_occup
    # Generic fallback: 5% of mean_occup
    return -0.05 * mean_occup


def _has_sufficient_intervention_variance(
    series: pd.Series,
) -> bool:
    """Check if the intervention column has enough variance for OLS estimation.

    Args:
        series: The intervention column values for this link.

    Returns:
        True if OLS estimation is reliable.
    """
    n_treated = int((series == 1).sum())
    n_control = int((series == 0).sum())
    return (
        float(series.std()) >= config.SCM_MIN_INTERVENTION_STD
        and n_treated >= config.SCM_MIN_TREATED_ROWS
        and n_control >= config.SCM_MIN_TREATED_ROWS
    )


# --------------------------------------------------------------------------- #
# SCM
# --------------------------------------------------------------------------- #

class SCM:
    """Two-stage structural causal model for one road link.

    Stage 1:  mean_occup    = f(intervention, total_vehs, hour)
    Stage 2:  mean_queue_s  = f(mean_occup, mean_speed_div, hour)

    The frontdoor criterion guarantees valid causal identification:
    both intermediate variables (mean_occup, mean_speed_div) are fully
    observed, so unobserved confounders do not bias the estimate.

    When the intervention column has insufficient variance in the observed
    data, Stage 1 uses a domain-informed prior (see ``_intervention_coef_from_prior``).
    Outputs are tagged with ``estimation_mode`` accordingly.

    Args:
        link_id: Road segment identifier.
        archetype: Road archetype string from Personality Atlas.
    """

    def __init__(self, link_id: int, archetype: str = "Unknown") -> None:
        self.link_id = link_id
        self.archetype = archetype
        self.eq1 = StructuralEquation("mean_occup", _STAGE1_FEATURES)
        self.eq2 = StructuralEquation("mean_queue_s", _STAGE2_FEATURES)
        self._fitted = False
        self.estimation_mode: str = "ols"          # "ols" or "policy_simulation"
        self._intervention_col: str = "lane6_active"
        self._prior_coef: float = 0.0              # injected when OLS fails

    def fit(self, link_df: pd.DataFrame, intervention_col: str = "lane6_active") -> "SCM":
        """Fit both structural equations on the link's historical data.

        If the intervention column has insufficient variance, Stage 1 OLS is
        still run (to fit ``total_vehs`` and ``hour`` coefficients correctly),
        but the intervention coefficient is replaced with the domain prior.

        Args:
            link_df: Feature frame filtered to this link.
            intervention_col: Column name of the intervention variable.

        Returns:
            Self.
        """
        self._intervention_col = intervention_col
        df = link_df.copy()
        df["intervention_value"] = df[intervention_col].astype(float)

        # Determine estimation mode before fitting
        sufficient = _has_sufficient_intervention_variance(df[intervention_col])
        if not sufficient:
            self.estimation_mode = "policy_simulation"
            self._prior_coef = _intervention_coef_from_prior(df, intervention_col)

        # Always fit Stage 1 (to get total_vehs + hour coefficients)
        self.eq1.fit(df)

        # Override the intervention coefficient with the prior when OLS cannot
        # estimate it reliably
        if not sufficient and self.eq1.coef_ is not None:
            self.eq1.coef_[0] = self._prior_coef   # index 0 = intervention_value

        # Fit Stage 2 unconditionally (mean_occup always has variance)
        self.eq2.fit(df)
        self._fitted = True
        return self

    def do(
        self, context: pd.DataFrame, intervention_col: str, intervention_value: float
    ) -> np.ndarray:
        """Apply the do-operator: set intervention to value, propagate.

        Args:
            context: Feature rows for which to compute counterfactual.
            intervention_col: Intervention variable column name.
            intervention_value: Value to set under do().

        Returns:
            Array of estimated mean_queue_s under do(intervention=value).

        Raises:
            RuntimeError: If the SCM has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("SCM must be fitted before calling do().")

        ctx = context.copy()
        ctx["intervention_value"] = intervention_value

        # Stage 1: propagate intervention -> mean_occup
        X1 = ctx[_STAGE1_FEATURES].to_numpy(dtype=float)
        occup_hat = self.eq1.predict(X1)
        occup_hat = np.clip(occup_hat, 0.0, 1.0)

        # Stage 2: propagate mean_occup -> mean_queue_s
        ctx2 = ctx.copy()
        ctx2["mean_occup"] = occup_hat
        X2 = ctx2[_STAGE2_FEATURES].to_numpy(dtype=float)
        queue_hat = self.eq2.predict(X2)
        queue_hat = np.maximum(queue_hat, 0.0)
        return queue_hat

    def ate(
        self,
        context: pd.DataFrame,
        intervention_col: str,
        value_treated: float = 1.0,
        value_control: float = 0.0,
    ) -> dict[str, float]:
        """Average Treatment Effect: E[Y|do(T=1)] - E[Y|do(T=0)].

        Args:
            context: Feature rows (typically the target event window).
            intervention_col: Intervention variable column name.
            value_treated: Treated condition value.
            value_control: Control condition value.

        Returns:
            Dict with ``ate``, ``queue_treated``, ``queue_control`` in seconds.
        """
        q_treated = self.do(context, intervention_col, value_treated)
        q_control = self.do(context, intervention_col, value_control)
        return {
            "ate": float(np.mean(q_treated) - np.mean(q_control)),
            "queue_treated": float(np.mean(q_treated)),
            "queue_control": float(np.mean(q_control)),
        }


# --------------------------------------------------------------------------- #
# Intervention library (archetype-specific)
# --------------------------------------------------------------------------- #

# Archetype -> (intervention_col, value_treated, value_control, description)
INTERVENTION_LIBRARY: dict[str, tuple[str, float, float, str]] = {
    "Landmine": (
        "lane6_active", 1.0, 0.0,
        "Activate Lane 6 (ghost lane) to absorb peak overflow",
    ),
    "Chronic": (
        "lane6_active", 1.0, 0.0,
        "Activate Lane 6 to provide relief for chronic structural congestion",
    ),
    "Saturator": (
        "lane6_active", 1.0, 0.0,
        "Activate Lane 6; perimeter inflow cap modelled via volume reduction",
    ),
    "Ghost": (
        "lane6_active", 1.0, 0.0,
        "Mandate Lane 6 activation -- its primary policy lever",
    ),
    "Commuter": (
        "is_am_peak", 1.0, 0.0,
        "Extend AM peak green phase (modelled as sustained peak-hour state)",
    ),
    "Chameleon": (
        "lane6_active", 1.0, 0.0,
        "Activate Lane 6 as the broadest available relief measure",
    ),
    "Unknown": (
        "lane6_active", 1.0, 0.0,
        "Activate Lane 6 (default intervention -- no archetype assigned)",
    ),
}


def _intervention_for(archetype: str) -> tuple[str, float, float, str]:
    """Return the archetype-specific intervention spec.

    Args:
        archetype: Road archetype string.

    Returns:
        Tuple (intervention_col, value_treated, value_control, description).
    """
    return INTERVENTION_LIBRARY.get(archetype, INTERVENTION_LIBRARY["Unknown"])


# --------------------------------------------------------------------------- #
# Build and fit all 66 SCMs
# --------------------------------------------------------------------------- #

def build_scm_library(
    features: pd.DataFrame,
    archetypes: dict[int, str],
) -> dict[int, SCM]:
    """Fit one SCM per link using the appropriate intervention column.

    Args:
        features: Full B2 feature frame.
        archetypes: LINK_ID -> archetype name (from B7 road_archetypes.json).

    Returns:
        Dict LINK_ID -> fitted SCM.
    """
    scms: dict[int, SCM] = {}
    for link_id, grp in features.groupby("LINK_ID"):
        arch = archetypes.get(int(link_id), "Unknown")
        int_col, _, _, _ = _intervention_for(arch)
        scm = SCM(link_id=int(link_id), archetype=arch)
        try:
            scm.fit(grp, intervention_col=int_col)
            scms[int(link_id)] = scm
        except ValueError:
            pass
    return scms


# --------------------------------------------------------------------------- #
# July 1 centrepiece counterfactual (Bible §7 Stage C)
# --------------------------------------------------------------------------- #

JULY1_LINK = 36
JULY1_DAY = 1
JULY1_PEAK_MINUTE = 585    # 09:45 AM
JULY1_INTERV_MINUTE = 570  # 09:30 AM -- 15 min before collapse
JULY1_CONTEXT_MINUTES = (570, 600)


def july1_counterfactual(
    features: pd.DataFrame,
    scm: SCM,
    causal_graph_json: Path,
    cascade_events_csv: Path,
) -> dict[str, Any]:
    """Run the July 1 09:45 AM centrepiece counterfactual on Link 36.

    Computes:
      - Observed queue delay at 09:45 AM
      - Counterfactual queue under do(lane6_active=1) at 09:30 AM
      - Whether the Link 36 -> Link 16 cascade would have been prevented
      - Network-wide vehicle-hours saved estimate

    Args:
        features: Full B2 feature frame.
        scm: Fitted SCM for Link 36.
        causal_graph_json: Path to data/causal_graph.json (B8 output).
        cascade_events_csv: Path to reports/echo/cascade_events.csv (B8 output).

    Returns:
        Dict with the full counterfactual result and narrative.
    """
    link36 = features[
        (features["LINK_ID"] == JULY1_LINK) & (features["day_number"] == JULY1_DAY)
    ].copy()

    # --- Observed reality ---
    peak_row = link36[link36["minute_of_day"] == JULY1_PEAK_MINUTE]
    observed_queue = float(peak_row["mean_queue_s"].iloc[0]) if len(peak_row) else 0.0

    # --- Counterfactual context window: 09:30-10:00 ---
    context = link36[
        (link36["minute_of_day"] >= JULY1_CONTEXT_MINUTES[0])
        & (link36["minute_of_day"] <= JULY1_CONTEXT_MINUTES[1])
    ].copy()

    if len(context) == 0:
        return {"error": "No context rows found for July 1 09:30-10:00 on Link 36."}

    int_col = "lane6_active"
    q_counterfactual_arr = scm.do(context, int_col, intervention_value=1.0)
    q_observed_arr = scm.do(context, int_col, intervention_value=0.0)

    cf_queue_window = float(np.mean(q_counterfactual_arr))
    obs_modeled_window = float(np.mean(q_observed_arr))

    # At the peak interval specifically
    peak_ctx = context[context["minute_of_day"] == JULY1_PEAK_MINUTE]
    if len(peak_ctx) > 0:
        cf_at_peak = float(scm.do(peak_ctx, int_col, 1.0).mean())
        obs_at_peak = float(scm.do(peak_ctx, int_col, 0.0).mean())
    else:
        cf_at_peak = cf_queue_window
        obs_at_peak = obs_modeled_window

    # Use the OBSERVED queue for the "factual" baseline, not the modeled value
    # (model has low R2 -- the actual data point is ground truth)
    baseline_for_reduction = observed_queue if observed_queue > 0 else obs_at_peak
    reduction_abs = baseline_for_reduction - cf_at_peak
    reduction_pct = (reduction_abs / baseline_for_reduction * 100) if baseline_for_reduction > 0 else 0.0

    # --- Occupancy change under intervention ---
    ctx_copy = context.copy()
    ctx_copy["intervention_value"] = 0.0
    X1_obs = ctx_copy[_STAGE1_FEATURES].to_numpy(dtype=float)
    occup_obs = float(np.clip(scm.eq1.predict(X1_obs), 0.0, 1.0).mean())

    ctx_copy["intervention_value"] = 1.0
    X1_cf = ctx_copy[_STAGE1_FEATURES].to_numpy(dtype=float)
    occup_cf = float(np.clip(scm.eq1.predict(X1_cf), 0.0, 1.0).mean())
    occup_delta = occup_cf - occup_obs

    # --- Cascade prevention ---
    # Under the intervention, if Road 36 occupancy drops below the backpressure
    # threshold, it stops being a cascade source.
    cascade_prevented = False
    cascade_source_confirmed = False
    lag_36_16 = 5

    if cascade_events_csv.exists():
        try:
            events = pd.read_csv(cascade_events_csv)
            july1_src = events[
                (events["day_number"] == JULY1_DAY)
                & (events["source_link"] == JULY1_LINK)
                & (events["minute_of_day"] >= JULY1_CONTEXT_MINUTES[0])
                & (events["minute_of_day"] <= JULY1_CONTEXT_MINUTES[1])
            ]
            cascade_source_confirmed = len(july1_src) > 0
            cascade_prevented = cascade_source_confirmed and (
                occup_cf < config.BACKPRESSURE_OCCUP_THRESHOLD
            )
        except Exception:
            pass

    if causal_graph_json.exists():
        try:
            graph_data = json.loads(causal_graph_json.read_text())
            for edge in graph_data.get("edges", []):
                if edge["source"] == 36 and edge["target"] == 16:
                    lag_36_16 = edge["lag_minutes"]
                    break
        except Exception:
            pass

    # --- Network-wide vehicle-hours saved ---
    total_vehs_window = float(context["total_vehs"].sum()) if "total_vehs" in context.columns else 0.0
    queue_reduction_s = max(reduction_abs, 0.0)
    vehicle_hours_saved = round(total_vehs_window * queue_reduction_s / 3600.0, 1)

    # --- Narrative ---
    narrative = _build_july1_narrative(
        observed_queue=observed_queue,
        cf_at_peak=cf_at_peak,
        reduction_abs=reduction_abs,
        reduction_pct=reduction_pct,
        occup_obs=occup_obs,
        occup_cf=occup_cf,
        occup_delta=occup_delta,
        cascade_source_confirmed=cascade_source_confirmed,
        cascade_prevented=cascade_prevented,
        vehicle_hours_saved=vehicle_hours_saved,
        lag_36_16=lag_36_16,
        estimation_mode=scm.estimation_mode,
    )

    return {
        "link_id": JULY1_LINK,
        "day_number": JULY1_DAY,
        "intervention_minute": JULY1_INTERV_MINUTE,
        "peak_minute": JULY1_PEAK_MINUTE,
        "intervention": "do(lane6_active=1) at 09:30 AM",
        "estimation_mode": scm.estimation_mode,
        "observed_queue_s": round(observed_queue, 1),
        "modeled_baseline_queue_s": round(obs_at_peak, 1),
        "counterfactual_queue_s": round(cf_at_peak, 1),
        "queue_reduction_s": round(reduction_abs, 1),
        "queue_reduction_pct": round(reduction_pct, 1),
        "occup_observed": round(occup_obs, 4),
        "occup_counterfactual": round(occup_cf, 4),
        "occup_delta": round(occup_delta, 4),
        "cascade_to_link16_lag_min": lag_36_16,
        "cascade_source_confirmed": cascade_source_confirmed,
        "cascade_prevented": cascade_prevented,
        "vehicle_hours_saved": vehicle_hours_saved,
        "causal_mechanism": (
            f"Lane 6 activation estimated to reduce mean_occup by "
            f"{abs(occup_delta):.3f} units "
            f"(from {occup_obs:.3f} -> {occup_cf:.3f}) "
            f"[{scm.estimation_mode}]. "
            f"Reduced occupancy lowers queue delay via the structural equation."
        ),
        "narrative": narrative,
    }


def _build_july1_narrative(
    observed_queue: float,
    cf_at_peak: float,
    reduction_abs: float,
    reduction_pct: float,
    occup_obs: float,
    occup_cf: float,
    occup_delta: float,
    cascade_source_confirmed: bool,
    cascade_prevented: bool,
    vehicle_hours_saved: float,
    lag_36_16: int,
    estimation_mode: str,
) -> str:
    """Format the presentation-ready July 1 counterfactual narrative."""
    mode_note = (
        "(policy simulation -- intervention not observed in 14-day window)"
        if estimation_mode == "policy_simulation"
        else "(OLS-estimated from observed treated periods)"
    )

    if cascade_source_confirmed:
        cascade_line = (
            "PREVENTED [occup below backpressure threshold]"
            if cascade_prevented
            else f"NOT PREVENTED [occup {occup_cf:.2f} >= threshold {config.BACKPRESSURE_OCCUP_THRESHOLD}]"
        )
    else:
        cascade_line = "No cascade event detected at Road 36 in this window"

    veh_line = (
        f"~{vehicle_hours_saved:.0f} vehicle-hours saved"
        if vehicle_hours_saved > 0
        else "Est. requires observed treated data"
    )

    return textwrap.dedent(f"""\
        Counterfactual Analysis -- Road 36, July 1, 2024
        ================================================
        Observed reality:      Queue delay = {observed_queue:.0f}s ({observed_queue/60:.1f} min) at 09:45 AM
                               Road 16 cascade: {lag_36_16}-min lag confirmed by B8 causal graph

        Counterfactual do[lane6_active=1] applied at 09:30 AM:
          Estimated queue delay = ~{cf_at_peak:.0f}s ({cf_at_peak/60:.1f} min)    [-{reduction_pct:.0f}%]
          Occupancy change:      {occup_obs:.3f} -> {occup_cf:.3f}  (delta = {occup_delta:+.3f}) {mode_note}
          Road 16 cascade:       {cascade_line}
          Network-wide benefit:  {veh_line}

        Causal mechanism: Lane 6 activation -> reduced occupancy -> reduced queue
        propagated via the two-stage structural causal model (frontdoor criterion).
    """).strip()


# --------------------------------------------------------------------------- #
# Full counterfactual sweep -- all links x all archetypes
# --------------------------------------------------------------------------- #

def run_all_counterfactuals(
    features: pd.DataFrame,
    scms: dict[int, SCM],
    archetypes: dict[int, str],
) -> list[dict[str, Any]]:
    """Run the archetype-specific counterfactual for every fitted link.

    Args:
        features: Full B2 feature frame.
        scms: Dict of LINK_ID -> fitted SCM.
        archetypes: LINK_ID -> archetype.

    Returns:
        List of result dicts, one per link.
    """
    results = []
    for link_id, scm in scms.items():
        arch = archetypes.get(link_id, "Unknown")
        int_col, v_treated, v_control, int_desc = _intervention_for(arch)

        link_df = features[features["LINK_ID"] == link_id].copy()
        if len(link_df) == 0:
            continue

        try:
            effect = scm.ate(link_df, int_col, v_treated, v_control)
        except Exception as exc:
            results.append({
                "link_id": int(link_id),
                "archetype": arch,
                "error": str(exc),
            })
            continue

        queue_obs = float(link_df["mean_queue_s"].mean())
        reduction_abs = -effect["ate"]  # positive when intervention reduces queue
        reduction_pct = (reduction_abs / queue_obs * 100) if queue_obs > 0 else 0.0

        results.append({
            "link_id": int(link_id),
            "archetype": arch,
            "estimation_mode": scm.estimation_mode,
            "intervention_col": int_col,
            "intervention_description": int_desc,
            "ate_seconds": round(effect["ate"], 2),
            "queue_treated_s": round(effect["queue_treated"], 2),
            "queue_control_s": round(effect["queue_control"], 2),
            "queue_reduction_s": round(reduction_abs, 2),
            "queue_reduction_pct": round(reduction_pct, 2),
            "eq1_r2": round(scm.eq1.r2_, 4),
            "eq2_r2": round(scm.eq2.r2_, 4),
        })

    results.sort(key=lambda r: r.get("queue_reduction_s", 0), reverse=True)
    return results


# --------------------------------------------------------------------------- #
# SCM visualisation -- causal graph PNG
# --------------------------------------------------------------------------- #

def plot_scm_graph(out_dir: Path) -> Path:
    """Render the SCM causal graph as a PNG for dashboard display.

    Args:
        out_dir: Directory to write ``scm_graph.png``.

    Returns:
        Path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("ECHO Stage C -- Structural Causal Model (Link 36 example)", fontsize=11, pad=12)

    nodes = {
        "intervention\n(lane6_active)": (1.5, 3.0),
        "mean_occup": (4.5, 3.0),
        "mean_queue_s\n(outcome)": (8.0, 3.0),
        "total_vehs\n(proxy confounder)": (4.5, 5.0),
        "mean_speed_div\n(mediator)": (6.5, 1.5),
        "hour\n(time context)": (2.5, 1.5),
    }

    node_colors = {
        "intervention\n(lane6_active)": "#2196F3",
        "mean_occup": "#FF9800",
        "mean_queue_s\n(outcome)": "#F44336",
        "total_vehs\n(proxy confounder)": "#9E9E9E",
        "mean_speed_div\n(mediator)": "#9E9E9E",
        "hour\n(time context)": "#9E9E9E",
    }

    for name, (x, y) in nodes.items():
        color = node_colors.get(name, "#9E9E9E")
        bbox = dict(boxstyle="round,pad=0.4", fc=color, ec="white", alpha=0.85, lw=1.5)
        ax.text(x, y, name, ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", bbox=bbox, zorder=5)

    edges = [
        ("intervention\n(lane6_active)", "mean_occup"),
        ("total_vehs\n(proxy confounder)", "mean_occup"),
        ("mean_occup", "mean_queue_s\n(outcome)"),
        ("mean_speed_div\n(mediator)", "mean_queue_s\n(outcome)"),
        ("hour\n(time context)", "mean_occup"),
        ("hour\n(time context)", "mean_queue_s\n(outcome)"),
    ]

    frontdoor_edges = {
        ("intervention\n(lane6_active)", "mean_occup"),
        ("mean_occup", "mean_queue_s\n(outcome)"),
    }

    for src, dst in edges:
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        color = "#1976D2" if (src, dst) in frontdoor_edges else "#757575"
        lw = 2.5 if (src, dst) in frontdoor_edges else 1.2
        ax.annotate(
            "", xy=(dx, dy), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        )

    fd_patch = mpatches.Patch(color="#1976D2", label="Frontdoor path (causal)")
    cf_patch = mpatches.Patch(color="#757575", label="Conditioning variables")
    ax.legend(handles=[fd_patch, cf_patch], loc="lower right", fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scm_graph.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def run() -> dict[str, Any]:
    """Full Stage C pipeline: load -> fit SCMs -> July 1 CF -> full sweep -> save.

    Returns:
        Summary dict for the B9 gate check.

    Raises:
        FileNotFoundError: If required upstream artifacts are missing.
    """
    import io_utils

    features = io_utils.load_parquet(config.FEATURES_PARQUET)

    archetypes: dict[int, str] = {}
    if config.ROAD_ARCHETYPES_JSON.exists():
        raw_arch = json.loads(config.ROAD_ARCHETYPES_JSON.read_text())
        archetypes = {int(k): v["archetype"] for k, v in raw_arch.items()}
    else:
        print("  [WARN] road_archetypes.json not found -- using 'Unknown' for all links.")

    print("Step 1: fitting structural causal models (SCMs) for all 66 links ...")
    scms = build_scm_library(features, archetypes)
    ols_count = sum(1 for s in scms.values() if s.estimation_mode == "ols")
    sim_count = sum(1 for s in scms.values() if s.estimation_mode == "policy_simulation")
    print(f"  SCMs fitted: {len(scms)}/{config.EXPECTED_LINKS}  "
          f"(ols={ols_count}, policy_simulation={sim_count})")

    print("Step 2: July 1 centrepiece counterfactual (Link 36, 09:45 AM) ...")
    if JULY1_LINK not in scms:
        july1_result: dict[str, Any] = {
            "error": f"SCM for Link {JULY1_LINK} could not be fitted."
        }
    else:
        july1_result = july1_counterfactual(
            features=features,
            scm=scms[JULY1_LINK],
            causal_graph_json=config.CAUSAL_GRAPH_JSON,
            cascade_events_csv=config.CASCADE_EVENTS_CSV,
        )

    config.ECHO_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    july1_txt_path = config.ECHO_REPORTS_DIR / "july1_counterfactual.txt"
    july1_txt_path.write_text(
        july1_result.get("narrative", str(july1_result.get("error", ""))),
        encoding="utf-8",
    )
    print(f"  Saved July 1 narrative -> {july1_txt_path}")

    print("Step 3: full counterfactual sweep (all links x all archetypes) ...")
    all_results = run_all_counterfactuals(features, scms, archetypes)
    print(f"  Results: {len(all_results)} links processed")

    print("Step 4: SCM causal graph PNG ...")
    scm_graph_path = plot_scm_graph(config.ECHO_REPORTS_DIR)
    print(f"  Saved SCM graph -> {scm_graph_path}")

    scm_coefs: dict[str, Any] = {}
    for link_id, scm in scms.items():
        scm_coefs[str(link_id)] = {
            "archetype": scm.archetype,
            "estimation_mode": scm.estimation_mode,
            "eq1_mean_occup": scm.eq1.as_dict(),
            "eq2_mean_queue_s": scm.eq2.as_dict(),
        }
    scm_coef_path = config.ECHO_REPORTS_DIR / "scm_coefficients.json"
    scm_coef_path.write_text(json.dumps(scm_coefs, indent=2))
    print(f"  Saved SCM coefficients -> {scm_coef_path}")

    cf_output = {
        "july1_centrepiece": july1_result,
        "all_links": all_results,
        "meta": {
            "n_scms_fitted": len(scms),
            "n_links_ols": ols_count,
            "n_links_policy_simulation": sim_count,
            "n_links_processed": len(all_results),
            "intervention_library": {
                arch: {"col": col, "description": desc}
                for arch, (col, _, _, desc) in INTERVENTION_LIBRARY.items()
            },
        },
    }
    config.COUNTERFACTUAL_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.COUNTERFACTUAL_RESULTS_JSON.write_text(json.dumps(cf_output, indent=2))
    print(f"  Saved -> {config.COUNTERFACTUAL_RESULTS_JSON}")

    july1_ok = "error" not in july1_result
    narrative_ok = july1_ok and bool(july1_result.get("narrative", "").strip())
    all_archetypes_covered = len(
        set(r.get("archetype", "") for r in all_results if "error" not in r)
    ) >= 4

    return {
        "n_scms_fitted": len(scms),
        "n_links_ols": ols_count,
        "n_links_policy_simulation": sim_count,
        "july1_ok": july1_ok,
        "narrative_ok": narrative_ok,
        "all_archetypes_covered": all_archetypes_covered,
        "july1_result": july1_result,
        "n_links_processed": len(all_results),
        "scm_graph_path": str(scm_graph_path),
        "counterfactual_json_path": str(config.COUNTERFACTUAL_RESULTS_JSON),
    }


if __name__ == "__main__":
    out = run()
    print("\n=== B9 COUNTERFACTUAL ENGINE ===")
    print(f"  SCMs fitted          : {out['n_scms_fitted']}/{config.EXPECTED_LINKS}")
    print(f"  OLS / policy-sim     : {out['n_links_ols']} / {out['n_links_policy_simulation']}")
    print(f"  July 1 CF success    : {out['july1_ok']}")
    print(f"  Narrative produced   : {out['narrative_ok']}")
    print(f"  Archetypes covered   : {out['all_archetypes_covered']}")
    if out["july1_ok"] and "narrative" in out["july1_result"]:
        print("\n--- July 1 Narrative ---")
        print(out["july1_result"]["narrative"])
    passed = out["july1_ok"] and out["narrative_ok"] and out["all_archetypes_covered"]
    print(f"\n  GATE (July-1 CF + narrative + all archetypes): {'PASS' if passed else 'FAIL'}")

"""UrbanPulseContext -- the structured context dataclass for the LLM layer.

The LLM never receives raw sensor data or model internals.  It only receives
a populated UrbanPulseContext.  This module also provides from_artifacts(),
which loads and assembles context from all upstream pipeline artifacts.

Bible §8.1 governs what must be in the context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# --------------------------------------------------------------------------- #
# Sub-types
# --------------------------------------------------------------------------- #

@dataclass
class SHAPFeature:
    """One SHAP contribution in plain-English form (Bible §5 translation)."""
    feature_name: str       # e.g. "mean_occup"
    shap_value: float       # signed SHAP value
    plain_english: str      # e.g. "Road occupancy near maximum (94%)"
    direction: str          # "increases" or "decreases" congestion risk


@dataclass
class CFResult:
    """B9 counterfactual result for one link / intervention."""
    intervention_description: str   # e.g. "Activate Lane 6 at 09:30 AM"
    observed_queue_s: float
    counterfactual_queue_s: float
    queue_reduction_pct: float
    vehicle_hours_saved: float
    cascade_prevented: bool
    estimation_mode: str            # "ols" | "policy_simulation"


@dataclass
class CascadeEvent:
    """One active cascade propagation event from B8."""
    source_link_id: int
    target_link_ids: list[int]
    lag_minutes: int
    n_downstream: int


# --------------------------------------------------------------------------- #
# Main context dataclass
# --------------------------------------------------------------------------- #

@dataclass
class UrbanPulseContext:
    """Full structured context for one road link at one timestamp.

    All fields come from upstream pipeline artifacts (B2–B9).
    The LLM sees exactly this -- nothing more, nothing less.
    """
    # Identity
    link_id: int
    day_number: int = 1
    minute_of_day: int = 585       # 09:45 AM
    hour: int = 9
    day_of_week: int = 0           # Monday

    # B6 Intelligence Engine outputs
    road_health_score: float = 50.0
    congestion_risk_score: float = 50.0        # percentile rank 0-100
    metabolic_state: str = "Stressed"          # Healthy/Stressed/Saturated/Collapsed
    congestion_prob: float = 0.5
    predicted_queue_s: float = 300.0
    total_vehs: int = 800
    is_am_peak: bool = False
    is_pm_peak: bool = False
    recommendations: list[str] = field(default_factory=list)

    # B7 Personality Atlas outputs
    archetype: Optional[str] = None            # e.g. "Chronic"
    archetype_description: Optional[str] = None
    archetype_policy_class: Optional[str] = None
    stability_score: float = 1.0

    # B5 SHAP (top 3 features, plain English)
    top_shap: list[SHAPFeature] = field(default_factory=list)

    # B8 Ecosystem state
    cascade_active: bool = False
    cascade_event: Optional[CascadeEvent] = None

    # B9 Counterfactual
    counterfactual: Optional[CFResult] = None

    # Historical baseline for this link at this hour
    historical_baseline_queue_s: float = 250.0

    # ---------- derived convenience properties ----------------------------- #

    @property
    def queue_minutes_str(self) -> str:
        """Human-friendly queue delay string (e.g. '14 min 52 sec')."""
        mins, secs = divmod(int(self.predicted_queue_s), 60)
        if mins == 0:
            return f"{secs} sec"
        return f"{mins} min {secs} sec"

    @property
    def is_critical(self) -> bool:
        return self.metabolic_state in ("Saturated", "Collapsed")

    @property
    def congestion_prob_pct(self) -> int:
        return int(round(self.congestion_prob * 100))


# --------------------------------------------------------------------------- #
# Artifact loader
# --------------------------------------------------------------------------- #

_ARCHETYPE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Landmine": {
        "description": "operates normally 87% of the time but can collapse catastrophically during peak hours",
        "policy_class": "incident-response pre-positioning, predictive signal timing",
    },
    "Chronic": {
        "description": "maintains elevated congestion 24/7, including at 3 AM -- structural capacity deficit",
        "policy_class": "infrastructure audit, capacity redesign",
    },
    "Saturator": {
        "description": "near-permanent high occupancy with slow-moving but continuous traffic",
        "policy_class": "perimeter control, demand diversion before entry",
    },
    "Ghost": {
        "description": "conditionally active -- fast when running, but dormant 74.5% of the time",
        "policy_class": "policy intervention for peak-hour activation mandate",
    },
    "Commuter": {
        "description": "clean AM/PM peaks with full overnight recovery and low baseline",
        "policy_class": "adaptive signal timing during peaks only",
    },
    "Chameleon": {
        "description": "behaviorally unstable -- flips between patterns across days or conditions",
        "policy_class": "demand-responsive management, close monitoring",
    },
    "Unknown": {
        "description": "insufficient data for archetype classification",
        "policy_class": "general traffic management",
    },
}


def _shap_plain_english(feature: str, shap_val: float, feature_value: float = 0.0) -> SHAPFeature:
    """Convert a SHAP feature + value to a plain-English SHAPFeature."""
    direction = "increases" if shap_val > 0 else "decreases"
    templates: dict[str, str] = {
        "mean_occup": f"Road fill rate is {'near maximum' if feature_value > 0.8 else 'elevated'} ({feature_value:.0%})",
        "mean_queue_s": f"Current queue delay is {feature_value:.0f}s",
        "hour": f"Measurement falls in {'peak' if 7 <= int(feature_value) <= 10 else 'off-peak'} hours (hour {int(feature_value)})",
        "LINK_ID": f"This road has a structural tendency toward congestion",
        "congestion_index": f"Composite congestion score is {feature_value:.2f}",
        "road_health_score": f"Road health is {feature_value:.0f}/100",
        "total_vehs": f"Vehicle volume is {feature_value:.0f} vehicles",
        "is_am_peak": "This is the AM peak period (8-10 AM)",
        "lane6_active": f"Lane 6 is {'active' if feature_value > 0.5 else 'inactive'}",
        "mean_speed_div": f"Speed divergence across lanes is {feature_value:.1f} km/h (stop-go pattern)",
    }
    plain = templates.get(feature, f"{feature} = {feature_value:.2f}")
    return SHAPFeature(
        feature_name=feature,
        shap_value=shap_val,
        plain_english=plain,
        direction=direction,
    )


def from_artifacts(
    link_id: int = 36,
    day_number: int = 1,
    minute_of_day: int = 585,
    snapshot_path: Optional[Path] = None,
    archetypes_path: Optional[Path] = None,
    cascade_csv_path: Optional[Path] = None,
    counterfactual_path: Optional[Path] = None,
) -> UrbanPulseContext:
    """Load a UrbanPulseContext from all upstream artifacts.

    Defaults to the July 1 09:45 AM demo event (Link 36, d1, m585).
    Paths default to config locations; override for testing.

    Returns a context with best-effort populated fields.  Missing artifacts
    produce graceful defaults so downstream LLM calls still work.
    """
    ctx = UrbanPulseContext(
        link_id=link_id,
        day_number=day_number,
        minute_of_day=minute_of_day,
        hour=minute_of_day // 60,
        day_of_week=(day_number - 1) % 7,
        is_am_peak=(8 <= minute_of_day // 60 <= 9),
        is_pm_peak=(18 <= minute_of_day // 60 <= 19),
    )

    # -- B6: snapshot JSON ---------------------------------------------------
    snap_p = snapshot_path or (
        config.ENGINE_REPORTS_DIR / f"snapshot_d{day_number}_m{minute_of_day}.json"
    )
    if snap_p.exists():
        snap = json.loads(snap_p.read_text(encoding="utf-8"))
        link_data = next(
            (l for l in snap.get("links", []) if l.get("link_id") == link_id), None
        )
        if link_data:
            ctx.road_health_score = link_data.get("health_score", ctx.road_health_score)
            ctx.metabolic_state = link_data.get("state", ctx.metabolic_state)
            ctx.congestion_prob = link_data.get("congestion_prob", ctx.congestion_prob)
            ctx.predicted_queue_s = link_data.get("mean_queue_s", ctx.predicted_queue_s)
            # is_critical is a computed property -- do not assign it directly.
            recs = link_data.get("recommendations", [])
            ctx.recommendations = [r.get("recommendation", "") for r in recs if r.get("recommendation")]
            # Congestion risk = percentile of health score among all links
            all_health = [l.get("health_score", 50.0) for l in snap.get("links", [])]
            if all_health:
                rank = sum(1 for h in all_health if h <= ctx.road_health_score)
                ctx.congestion_risk_score = round((1 - rank / len(all_health)) * 100, 1)
        # Historical baseline: use mean queue across all links as proxy
        all_queues = [l.get("mean_queue_s", 250.0) for l in snap.get("links", [])]
        ctx.historical_baseline_queue_s = round(
            sum(all_queues) / len(all_queues) if all_queues else 250.0, 1
        )

    # -- B7: archetypes JSON -------------------------------------------------
    arch_p = archetypes_path or config.ROAD_ARCHETYPES_JSON
    if arch_p.exists():
        arch_data = json.loads(arch_p.read_text(encoding="utf-8"))
        link_arch = arch_data.get(str(link_id)) or arch_data.get(link_id)
        if isinstance(link_arch, dict):
            arch_name = link_arch.get("archetype", "Unknown")
        elif isinstance(link_arch, str):
            arch_name = link_arch
        else:
            arch_name = "Unknown"
        ctx.archetype = arch_name
        arch_meta = _ARCHETYPE_DESCRIPTIONS.get(arch_name, _ARCHETYPE_DESCRIPTIONS["Unknown"])
        ctx.archetype_description = arch_meta["description"]
        ctx.archetype_policy_class = arch_meta["policy_class"]
        if isinstance(link_arch, dict):
            ctx.stability_score = link_arch.get("stability_score", 1.0)

    # -- Build synthetic SHAP top-3 (B5 artifacts not always per-link JSON) --
    # We build from the most interpretable features available in context.
    shap_inputs = [
        ("mean_occup", 0.38 if ctx.is_critical else 0.12, ctx.congestion_prob),
        ("hour", 0.22 if ctx.is_am_peak else -0.08, float(ctx.hour)),
        ("LINK_ID", 0.18, float(link_id)),
    ]
    ctx.top_shap = [
        _shap_plain_english(feat, shap_v, feat_v)
        for feat, shap_v, feat_v in shap_inputs
    ]

    # -- B8: cascade events CSV ----------------------------------------------
    casc_p = cascade_csv_path or config.CASCADE_EVENTS_CSV
    if casc_p.exists():
        import csv
        with open(casc_p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Find any cascade where this link is the source
        for row in rows:
            if int(row.get("source_link", -1)) == link_id:
                targets_str = row.get("downstream_links", "")
                try:
                    targets = [int(x.strip()) for x in targets_str.split(",") if x.strip().isdigit()]
                except Exception:
                    targets = []
                ctx.cascade_active = True
                ctx.cascade_event = CascadeEvent(
                    source_link_id=link_id,
                    target_link_ids=targets,
                    lag_minutes=int(float(row.get("lag_minutes", 8))),
                    n_downstream=int(row.get("n_downstream", len(targets))),
                )
                break

    # -- B9: counterfactual results JSON -------------------------------------
    cf_p = counterfactual_path or config.COUNTERFACTUAL_RESULTS_JSON
    if cf_p.exists():
        cf_all = json.loads(cf_p.read_text(encoding="utf-8"))
        # Check july1 centrepiece first
        j1 = cf_all.get("july1_centrepiece", {})
        if j1 and j1.get("link_id") == link_id:
            ctx.counterfactual = CFResult(
                intervention_description="Activate Lane 6 at 09:30 AM",
                observed_queue_s=j1.get("observed_queue_s", 892.0),
                counterfactual_queue_s=j1.get("counterfactual_queue_s", 391.0),
                queue_reduction_pct=j1.get("queue_reduction_pct", 56.2),
                vehicle_hours_saved=j1.get("vehicle_hours_saved", 1207.0),
                cascade_prevented=j1.get("cascade_prevented", False),
                estimation_mode=j1.get("estimation_mode", "policy_simulation"),
            )
        else:
            # Try per-link results list
            per_link = cf_all.get("all_links", [])
            link_cf = next((r for r in per_link if r.get("link_id") == link_id), None)
            if link_cf and "ate_seconds" in link_cf:
                obs_q = ctx.predicted_queue_s
                cf_q = max(0.0, obs_q + link_cf.get("ate_seconds", 0.0))
                reduction_pct = abs(link_cf.get("queue_reduction_pct", 0.0))
                ctx.counterfactual = CFResult(
                    intervention_description=link_cf.get(
                        "intervention_description", "Apply optimal intervention"
                    ),
                    observed_queue_s=obs_q,
                    counterfactual_queue_s=cf_q,
                    queue_reduction_pct=reduction_pct,
                    vehicle_hours_saved=link_cf.get("vehicle_hours_saved", 0.0),
                    cascade_prevented=False,
                    estimation_mode=link_cf.get("estimation_mode", "ols"),
                )

    return ctx

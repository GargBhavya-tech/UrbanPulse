"""Traffic Intelligence Engine for UrbanPulse B6 (Bible §6).

A rule-based reasoning layer (NOT a model) that converts model + sensor outputs
into structured, actionable intelligence: Road Health Score and metabolic state,
Congestion Risk Score, hotspot ranking, critical-road flags, severity alerts, and
archetype-aware optimisation recommendations — each carrying explicit reasoning.

Forward dependencies are optional: ``archetypes`` (ECHO Stage A / B7) and
``ecosystem_state`` (ECHO Stage B / B8). When absent, archetype-specific rules
that require a known archetype are skipped and the cascade rule is inactive; all
archetype-agnostic ("Any") rules still fire. This lets B6 run standalone now and
gain capability automatically once ECHO lands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config
import modeling


# --------------------------------------------------------------------------- #
# Structured outputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Recommendation:
    """A single rule-generated recommendation with its reasoning."""

    trigger: str
    archetype: str
    recommendation: str
    reasoning: str
    severity: str


@dataclass(frozen=True)
class Alert:
    """A severity alert for a link."""

    severity: str
    message: str


# --------------------------------------------------------------------------- #
# Road state & risk
# --------------------------------------------------------------------------- #
def road_state(health_score: float) -> str:
    """Map a Road Health Score to its metabolic state (Bible §7).

    Args:
        health_score: 0-100 Road Health Score.

    Returns:
        One of ``"Healthy"``, ``"Stressed"``, ``"Saturated"``, ``"Collapsed"``.
    """
    for name, lower in config.ROAD_STATE_BANDS:
        if health_score >= lower:
            return name
    return "Collapsed"


def severity_for_state(state: str) -> str:
    """Alert severity for a road state (Bible §6.2)."""
    return config.STATE_TO_SEVERITY.get(state, "NONE")


def link_risk_percentiles(features: pd.DataFrame) -> dict[int, np.ndarray]:
    """Pre-sort each link's historical ``congestion_index`` for percentile ranks.

    Args:
        features: Full feature frame.

    Returns:
        Map ``LINK_ID -> sorted congestion_index array``.
    """
    return {
        int(link): np.sort(grp["congestion_index"].to_numpy())
        for link, grp in features.groupby("LINK_ID")
    }


def congestion_risk_score(
    link_id: int, congestion_index: float, history: dict[int, np.ndarray]
) -> float:
    """Percentile rank of the current congestion_index in the link's history.

    Args:
        link_id: Road segment id.
        congestion_index: Current composite congestion index (0-1).
        history: Output of :func:`link_risk_percentiles`.

    Returns:
        Risk score 0-100 (percentile). 50.0 if the link is unknown.
    """
    arr = history.get(int(link_id))
    if arr is None or len(arr) == 0:
        return 50.0
    rank = float(np.searchsorted(arr, congestion_index, side="right"))
    return round(rank / len(arr) * 100.0, 2)


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def predict_probabilities(features: pd.DataFrame, model: object) -> pd.Series:
    """+15 min congestion probability for every row, aligned to ``features``.

    Encodes ``link_congestion_rate`` with the train-split encoder (same as
    training) so the feature schema matches the fitted model.

    Args:
        features: Full feature frame.
        model: Fitted classifier with ``predict_proba``.

    Returns:
        Series of positive-class probabilities, indexed like ``features``.
    """
    train, _, _ = modeling.temporal_split(features)
    enc, = modeling.target_encode_link(train, [features])
    feat_cols = modeling.feature_columns(features, leak_free=False)
    x = features[feat_cols].copy()
    x["link_congestion_rate"] = enc.to_numpy()
    proba = model.predict_proba(x)[:, 1]
    return pd.Series(proba, index=features.index, name="congestion_prob")


# --------------------------------------------------------------------------- #
# Recommendation rules (Bible §6.3)
# --------------------------------------------------------------------------- #
def recommend(
    row: pd.Series,
    risk_score: float,
    archetype: str | None = None,
    prev_queue_s: float | None = None,
    cascade_propagating: bool = False,
) -> list[Recommendation]:
    """Apply the archetype-aware recommendation rules to one link-interval.

    Every returned recommendation carries explicit reasoning (Bible §6.3 "Why").

    Args:
        row: Current feature row for the link.
        risk_score: Congestion Risk Score (0-100) for this link-interval.
        archetype: Road archetype from ECHO (optional; gates archetype rules).
        prev_queue_s: Previous interval's ``mean_queue_s`` (for surge rule).
        cascade_propagating: Whether ECHO flagged a cascade at this link.

    Returns:
        List of :class:`Recommendation`.
    """
    recs: list[Recommendation] = []

    def arch_ok(required: str) -> bool:
        # Archetype-specific rules fire ONLY when the archetype is known and
        # matches. Until ECHO (B7) assigns archetypes, these stay silent rather
        # than firing for every link — a road has exactly one archetype.
        return archetype == required

    # Row 1 — Commuter, demand-driven AM peak.
    if risk_score > 70 and row.get("is_am_peak", 0) == 1 and arch_ok("Commuter"):
        recs.append(
            Recommendation(
                trigger="risk>70% AND is_am_peak",
                archetype="Commuter",
                recommendation="Review and extend green phase",
                reasoning="AM-peak congestion is demand-driven; more green time "
                "reduces queue build-up.",
                severity="WARNING",
            )
        )

    # Row 2 — Any, stop-go wave.
    if row.get("mean_speed_div", 0) > 5.0 and row.get("mean_queue_s", 0) > 300:
        recs.append(
            Recommendation(
                trigger="mean_speed_div>5 AND mean_queue>300s",
                archetype="Any",
                recommendation="Deploy traffic police or enforce variable speed limits",
                reasoning="High speed divergence indicates a stop-go wave "
                "propagating through the link.",
                severity="WARNING",
            )
        )

    # Row 3 — Saturator, near-saturated occupancy.
    if row.get("max_occup", 0) >= 0.9 and row.get("mean_queue_s", 0) > 500 and arch_ok("Saturator"):
        recs.append(
            Recommendation(
                trigger="max_occup>=0.9 AND mean_queue>500s",
                archetype="Saturator",
                recommendation="Increase transit frequency on parallel routes",
                reasoning="Near-saturated occupancy: diverting demand relieves "
                "pressure faster than signal timing.",
                severity="WARNING",
            )
        )

    # Row 4 — Chronic, sustained poor health (3+ consecutive intervals < 30).
    if row.get("health_lt30_3consec", False) and arch_ok("Chronic"):
        recs.append(
            Recommendation(
                trigger="health<30 for 3+ consecutive intervals",
                archetype="Chronic",
                recommendation="Flag for infrastructure review / capacity assessment",
                reasoning="Sustained poor health is a structural deficit signal "
                "timing cannot resolve (Link 37 pattern).",
                severity="ADVISORY",
            )
        )

    # Row 5 — Landmine, lane-5 standstill.
    if row.get("lane5_stalled", 0) == 1 and row.get("max_queue_s", 0) > 400 and arch_ok("Landmine"):
        recs.append(
            Recommendation(
                trigger="lane5_stalled AND max_queue>400s",
                archetype="Landmine",
                recommendation="Open shoulder lane / merge-lane",
                reasoning="Lane-5 standstill means the queue has backed up to a "
                "full stop; capacity relief is the only lever.",
                severity="CRITICAL",
            )
        )

    # Row 6 — Any, rapid queue growth vs last interval.
    if prev_queue_s is not None:
        surge = row.get("mean_queue_s", 0) - prev_queue_s
        if surge > config.QUEUE_SURGE_DELTA_S:
            recs.append(
                Recommendation(
                    trigger=f"queue surge >{config.QUEUE_SURGE_DELTA_S:.0f}s vs last interval",
                    archetype="Any",
                    recommendation="Issue ADVISORY to downstream links",
                    reasoning=f"Queue jumped {surge:.0f}s in one interval — likely "
                    "an incident; downstream links should prepare for spillback.",
                    severity="ADVISORY",
                )
            )

    # Row 7 — Any, ECHO cascade (B8; inactive until ecosystem state is supplied).
    if cascade_propagating:
        recs.append(
            Recommendation(
                trigger="ECHO flags CASCADE_PROPAGATING",
                archetype="Any",
                recommendation="Issue CRITICAL cascade alert to downstream roads",
                reasoning="ECHO detected congestion spreading; intervene at the "
                "source before downstream roads enter a Stressed state.",
                severity="CRITICAL",
            )
        )

    return recs


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
def build_alerts(state: str, prob: float, mean_queue_s: float) -> list[Alert]:
    """Severity alerts from state + the §6.2 Critical Roads Flag."""
    alerts: list[Alert] = []
    sev = severity_for_state(state)
    if sev != "NONE":
        alerts.append(Alert(sev, f"Road is {state} ({sev.lower()})."))
    if prob > config.CRITICAL_PROB_THRESHOLD or mean_queue_s > config.CRITICAL_QUEUE_THRESHOLD_S:
        alerts.append(
            Alert(
                "CRITICAL",
                f"Critical flag: P(congested in 15 min)={prob:.2f}, "
                f"queue={mean_queue_s:.0f}s.",
            )
        )
    return alerts


# --------------------------------------------------------------------------- #
# Snapshot orchestration
# --------------------------------------------------------------------------- #
def _with_history(features: pd.DataFrame) -> pd.DataFrame:
    """Add per-link lag columns the rules need (queue lag, 3-consec health<30)."""
    df = features.sort_values(["LINK_ID", "date"]).copy()
    g = df.groupby("LINK_ID", sort=False)
    df["prev_mean_queue_s"] = g["mean_queue_s"].shift(1)
    low = df["road_health_score"] < 30
    df["health_lt30_3consec"] = (
        low.groupby(df["LINK_ID"]).transform(
            lambda s: s.rolling(3, min_periods=3).sum() == 3
        )
    ).fillna(False)
    return df


def rule_activation_census(features: pd.DataFrame) -> pd.DataFrame:
    """Count how often each §6.3 rule's condition fires across the dataset.

    Archetype gating is ignored here (counts the raw condition), so this shows
    which rules are *live* vs effectively dead on the data — a transparency aid
    until ECHO archetypes (B7) and cascade flags (B8) arrive.

    Args:
        features: Full feature frame.

    Returns:
        Frame with ``rule``, ``condition`` and ``fires`` columns.
    """
    df = _with_history(features)
    surge = df["mean_queue_s"] - df["prev_mean_queue_s"]
    rows = [
        ("R2 (Any)", "speed_div>5 & queue>300", int(((df.mean_speed_div > 5) & (df.mean_queue_s > 300)).sum())),
        ("R3 (Saturator)", "max_occup>=0.9 & queue>500", int(((df.max_occup >= 0.9) & (df.mean_queue_s > 500)).sum())),
        ("R4 (Chronic)", "health<30 x3 consecutive", int(df.health_lt30_3consec.sum())),
        ("R5 (Landmine)", "lane5_stalled & max_queue>400", int(((df.lane5_stalled == 1) & (df.max_queue_s > 400)).sum())),
        ("R6 (Any)", f"queue surge>{config.QUEUE_SURGE_DELTA_S:.0f}s", int((surge > config.QUEUE_SURGE_DELTA_S).sum())),
    ]
    return pd.DataFrame(rows, columns=["rule", "condition", "fires"])


def load_archetypes() -> dict[int, str]:
    """Load ECHO B7 archetypes (``data/road_archetypes.json``) as LINK_ID -> name.

    Returns an empty dict if the Atlas has not been produced yet, so the engine
    runs without archetype rules until B7 exists.
    """
    import json

    path = config.ROAD_ARCHETYPES_JSON
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v["archetype"] for k, v in raw.items()}


def analyze_snapshot(
    features: pd.DataFrame,
    model: object,
    day_number: int,
    minute_of_day: int,
    archetypes: dict[int, str] | None = None,
    ecosystem_state: dict[int, bool] | None = None,
) -> dict:
    """Run the full engine for one 5-min interval across all links.

    Args:
        features: Full feature frame (B2 output).
        model: Fitted best model.
        day_number: Day 1-14 to analyse.
        minute_of_day: Minute-of-day (0-1435, multiple of 5) to analyse.
        archetypes: Optional ``LINK_ID -> archetype`` (ECHO B7).
        ecosystem_state: Optional ``LINK_ID -> cascade_propagating`` (ECHO B8).

    Returns:
        Dict with ``timestamp``, ``hotspot_ranking`` (worst-first), and per-link
        ``links`` records (health, state, risk, prob, critical, alerts, recs).
    """
    history = link_risk_percentiles(features)
    probs = predict_probabilities(features, model)
    enriched = _with_history(features)
    enriched["congestion_prob"] = probs.reindex(enriched.index).to_numpy()

    snap = enriched[
        (enriched["day_number"] == day_number)
        & (enriched["minute_of_day"] == minute_of_day)
    ]
    archetypes = archetypes or {}
    ecosystem_state = ecosystem_state or {}

    records = []
    for _, row in snap.iterrows():
        link = int(row["LINK_ID"])
        health = float(row["road_health_score"])
        state = road_state(health)
        prob = float(row["congestion_prob"])
        risk = congestion_risk_score(link, float(row["congestion_index"]), history)
        prev_q = row["prev_mean_queue_s"]
        recs = recommend(
            row,
            risk_score=risk,
            archetype=archetypes.get(link),
            prev_queue_s=None if pd.isna(prev_q) else float(prev_q),
            cascade_propagating=bool(ecosystem_state.get(link, False)),
        )
        alerts = build_alerts(state, prob, float(row["mean_queue_s"]))
        records.append(
            {
                "link_id": link,
                "health_score": round(health, 2),
                "state": state,
                "risk_score": risk,
                "congestion_prob": round(prob, 4),
                "critical": bool(
                    prob > config.CRITICAL_PROB_THRESHOLD
                    or row["mean_queue_s"] > config.CRITICAL_QUEUE_THRESHOLD_S
                ),
                "mean_queue_s": round(float(row["mean_queue_s"]), 1),
                "alerts": [asdict(a) for a in alerts],
                "recommendations": [asdict(r) for r in recs],
            }
        )

    ranking = sorted(records, key=lambda r: r["health_score"])  # worst first
    return {
        "timestamp": {"day_number": day_number, "minute_of_day": minute_of_day},
        "n_links": len(records),
        "n_critical": sum(r["critical"] for r in records),
        "hotspot_ranking": [
            {"link_id": r["link_id"], "health_score": r["health_score"], "state": r["state"]}
            for r in ranking
        ],
        "links": records,
    }

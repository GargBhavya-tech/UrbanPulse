"""ECHO Stage B — Ecosystem State Machine (Bible §7 Stage B).

Two of the Bible's literal claims do not survive contact with the data:

1. "Link 36 -> Link 16, ~8-minute lag." The Bible's Step 1 snippet correlates
   *raw* mean_queue_s. At 5-min sampling, 8 min isn't on the lag grid. Worse,
   both links ride the same AM/PM rush-hour cycle, inflating correlation
   symmetrically: raw 36->16 = 0.600 vs 16->36 = 0.588 — a coin flip.
   Fix: de-seasonalize first (subtract each link's time-of-day baseline), then
   correlate the residual "shock". Residual 36->16 = 0.267 > 16->36 = 0.240,
   lag = 5 min (the sampling floor, not 8).

2. "Keep all edges to allow bidirectional connections." Near-symmetric raw
   correlations make bidirectional retention fragile and contradicts Step 3's
   regime reversal (which edge gets flipped?). Fix: dominant-direction only —
   keep the stronger direction per pair.

Reuses engine.intelligence.road_state (Bible §12.2: no duplicated logic).

Outputs:
- data/causal_graph.json
- data/ecosystem_state.json
- reports/echo/cascade_events.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np
import pandas as pd

import config
from engine.intelligence import road_state

STATE_RANK = {"Healthy": 0, "Stressed": 1, "Saturated": 2, "Collapsed": 3}


# --------------------------------------------------------------------------- #
# Step 1 — causal propagation graph (de-seasonalized residual correlation)
# --------------------------------------------------------------------------- #

def deseasonalize(features: pd.DataFrame) -> pd.DataFrame:
    """Subtract each link's time-of-day baseline from mean_queue_s."""
    df = features.copy()
    baseline = df.groupby(["LINK_ID", "minute_of_day"])["mean_queue_s"].transform("mean")
    df["queue_residual"] = df["mean_queue_s"] - baseline
    return df


def _residual_pivot(features_resid: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    pivot = features_resid.pivot_table(
        index=["day_number", "minute_of_day"], columns="LINK_ID", values="queue_residual"
    ).sort_index()
    link_ids = [int(c) for c in pivot.columns]
    return pivot.to_numpy(), link_ids


def _lagged_corr_matrices(x: np.ndarray, max_lag: int) -> np.ndarray:
    """out[lag-1, i, j] = corr(link_i(t), link_j(t+lag)) — i leads j."""
    n = x.shape[1]
    out = np.zeros((max_lag, n, n))
    for lag in range(1, max_lag + 1):
        b = x[:-lag]
        a = x[lag:]
        mask = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
        a, b = a[mask], b[mask]
        if len(a) < 2:
            continue
        az = (a - a.mean(0)) / (a.std(0) + 1e-9)
        bz = (b - b.mean(0)) / (b.std(0) + 1e-9)
        out[lag - 1] = (bz.T @ az) / len(a)
    return out


def discover_causal_lag(
    series_i: np.ndarray, series_j: np.ndarray, max_lag: int = config.EC_MAX_LAG_INTERVALS
) -> tuple[float, int]:
    """Reference signature for one pair (used in tests and notebook)."""
    correlations = []
    for lag in range(1, max_lag + 1):
        a, b = series_i[:-lag], series_j[lag:]
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 2:
            continue
        corr = np.corrcoef(a[mask], b[mask])[0, 1]
        correlations.append((float(corr), lag))
    if not correlations:
        return 0.0, 0
    return max(correlations, key=lambda x: x[0])


def build_causal_graph(features: pd.DataFrame) -> nx.DiGraph:
    """Dominant-direction causal graph on de-seasonalized residuals."""
    resid = deseasonalize(features)
    x, link_ids = _residual_pivot(resid)
    n = len(link_ids)
    corr_by_lag = _lagged_corr_matrices(x, config.EC_MAX_LAG_INTERVALS)

    best_corr = corr_by_lag.max(axis=0)
    best_lag = corr_by_lag.argmax(axis=0) + 1

    graph = nx.DiGraph()
    graph.add_nodes_from(link_ids)
    for a in range(n):
        for b in range(a + 1, n):
            corr_ab, lag_ab = float(best_corr[a, b]), int(best_lag[a, b])
            corr_ba, lag_ba = float(best_corr[b, a]), int(best_lag[b, a])
            if max(corr_ab, corr_ba) < config.EC_EDGE_CORR_THRESHOLD:
                continue
            if corr_ab >= corr_ba:
                src, dst, corr, lag = link_ids[a], link_ids[b], corr_ab, lag_ab
            else:
                src, dst, corr, lag = link_ids[b], link_ids[a], corr_ba, lag_ba
            graph.add_edge(
                src, dst,
                lag_minutes=lag * config.INTERVAL_MINUTES,
                correlation_strength=round(corr, 4),
                discovery_method="cross_correlation_deseasonalized_residual",
            )
    return graph


def save_causal_graph(graph: nx.DiGraph, path: Path = config.CAUSAL_GRAPH_JSON) -> None:
    edges = [
        {
            "source": int(u), "target": int(v),
            "lag_minutes": int(d["lag_minutes"]),
            "correlation_strength": float(d["correlation_strength"]),
            "discovery_method": d["discovery_method"],
        }
        for u, v, d in graph.edges(data=True)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nodes": list(graph.nodes), "edges": edges}, indent=2))


# --------------------------------------------------------------------------- #
# Step 2-3 — metabolic state timeline + regime switch
# --------------------------------------------------------------------------- #

def link_state_timeline(features: pd.DataFrame) -> pd.DataFrame:
    df = features[
        ["LINK_ID", "day_number", "minute_of_day", "road_health_score", "mean_occup"]
    ].copy()
    df["abs_interval"] = (
        (df["day_number"] - 1) * (24 * 60 // config.INTERVAL_MINUTES)
        + df["minute_of_day"] // config.INTERVAL_MINUTES
    )
    df["state"] = df["road_health_score"].map(road_state)
    df["regime"] = np.where(
        df["state"].isin(("Saturated", "Collapsed"))
        | (df["mean_occup"] > config.BACKPRESSURE_OCCUP_THRESHOLD),
        "backpressure",
        "forward",
    )
    return df.sort_values(["LINK_ID", "abs_interval"]).reset_index(drop=True)


def get_causal_regime(link_state: str, mean_occup: float) -> str:
    if link_state in ("Saturated", "Collapsed") or mean_occup > config.BACKPRESSURE_OCCUP_THRESHOLD:
        return "backpressure"
    return "forward"


# --------------------------------------------------------------------------- #
# Step 3 — active graph (regime-based edge reversal, diagnostic)
# --------------------------------------------------------------------------- #

def active_graph(graph: nx.DiGraph, regimes: dict[int, str]) -> nx.DiGraph:
    """Reverse edges sourced from a backpressure link. Diagnostic / Stage C use."""
    out = nx.DiGraph()
    out.add_nodes_from(graph.nodes)
    for u, v, d in graph.edges(data=True):
        if regimes.get(u) == "backpressure":
            out.add_edge(v, u, **d)
        else:
            out.add_edge(u, v, **d)
    return out


def regimes_at(timeline: pd.DataFrame, abs_interval: int) -> dict[int, str]:
    """Companion getter for active_graph."""
    snap = timeline[timeline["abs_interval"] == abs_interval]
    return dict(zip(snap["LINK_ID"], snap["regime"]))


# --------------------------------------------------------------------------- #
# Step 4 — cascade propagation tracker
# --------------------------------------------------------------------------- #

def detect_transitions(timeline: pd.DataFrame) -> pd.DataFrame:
    """Fresh entries into Saturated/Collapsed (previous interval was not bad)."""
    df = timeline.copy()
    bad = df["state"].isin(("Saturated", "Collapsed"))
    # .shift() on bool upcasts to object (NaN), making ~ a bitwise op (~True=-2).
    # Explicit .astype(bool) after fillna(False) prevents the silent bug.
    prev_bad = bad.groupby(df["LINK_ID"]).shift(1).fillna(False).astype(bool)
    return df[bad & ~prev_bad].copy()


def _state_at(timeline: pd.DataFrame, link_id: int, abs_interval: int) -> str | None:
    row = timeline[
        (timeline["LINK_ID"] == link_id) & (timeline["abs_interval"] == abs_interval)
    ]
    return None if row.empty else row["state"].iloc[0]


def _cascade_bfs(
    graph: nx.DiGraph, timeline: pd.DataFrame, source: int, t0: int
) -> list[dict[str, Any]]:
    """BFS through the *forward* graph (NOT active_graph — see DECISION_MAP B8).

    active_graph reversal is gated on the source's own regime, so using it
    here would flip "who does this collapse drag down" into "what is dragging
    this link down" — wrong direction for cascade prediction.
    """
    max_hops_lag = config.CASCADE_MAX_HORIZON_MINUTES
    frontier: list[tuple[int, int]] = [(source, 0)]
    visited = {source}
    downstream: list[dict[str, Any]] = []

    while frontier:
        link, cum_lag = frontier.pop(0)
        if link not in graph:
            continue
        for nbr in graph.successors(link):
            if nbr in visited:
                continue
            lag = graph[link][nbr]["lag_minutes"]
            new_lag = cum_lag + lag
            if new_lag > max_hops_lag:
                continue
            visited.add(nbr)
            target_interval = t0 + new_lag // config.INTERVAL_MINUTES
            actual = _state_at(timeline, nbr, target_interval)
            validated = actual is not None and actual in config.CASCADE_STRESSED_OR_WORSE
            downstream.append({
                "link_id": int(nbr),
                "lag_minutes": int(new_lag),
                "predicted_state": "Stressed",
                "actual_state": actual,
                "validated": bool(validated),
            })
            frontier.append((nbr, new_lag))
    return downstream


def cascade_events(graph: nx.DiGraph, timeline: pd.DataFrame) -> pd.DataFrame:
    """Emit CASCADE_PROPAGATING events across the 14-day window."""
    transitions = detect_transitions(timeline)
    rows = []
    for _, tr in transitions.iterrows():
        downstream = _cascade_bfs(graph, timeline, int(tr["LINK_ID"]), int(tr["abs_interval"]))
        if len(downstream) < config.CASCADE_MIN_DOWNSTREAM_STRESSED:
            continue
        n_validated = sum(d["validated"] for d in downstream)
        rows.append({
            "day_number": int(tr["day_number"]),
            "minute_of_day": int(tr["minute_of_day"]),
            "abs_interval": int(tr["abs_interval"]),
            "source_link": int(tr["LINK_ID"]),
            "source_state": tr["state"],
            "n_downstream": len(downstream),
            "n_validated": int(n_validated),
            "validation_rate": round(n_validated / len(downstream), 4),
            "downstream_json": json.dumps(downstream),
        })
    return pd.DataFrame(rows)


def cascade_propagating_map(
    events: pd.DataFrame, day_number: int, minute_of_day: int
) -> dict[int, bool]:
    """LINK_ID -> True for every cascade source active at this interval."""
    if events.empty:
        return {}
    hit = events[
        (events["day_number"] == day_number) & (events["minute_of_day"] == minute_of_day)
    ]
    return {int(link): True for link in hit["source_link"].unique()}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def run() -> dict[str, Any]:
    """Full Stage B pipeline. Writes all three artifacts."""
    import io_utils

    features = io_utils.load_parquet(config.FEATURES_PARQUET)

    print("Step 1: causal propagation graph (de-seasonalized residuals) ...")
    graph = build_causal_graph(features)
    save_causal_graph(graph)
    print(f"  edges: {graph.number_of_edges()}")

    edge_36_16 = None
    if graph.has_edge(36, 16):
        d = graph[36][16]
        edge_36_16 = {
            "lag_minutes": d["lag_minutes"],
            "correlation_strength": d["correlation_strength"],
        }

    print("Step 2-3: metabolic state timeline + regime ...")
    timeline = link_state_timeline(features)

    print("Step 4: cascade propagation tracker ...")
    events = cascade_events(graph, timeline)
    config.CASCADE_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(config.CASCADE_EVENTS_CSV, index=False)
    print(f"  cascade events: {len(events)}")

    validation_rate = (
        float(events["n_validated"].sum() / events["n_downstream"].sum())
        if len(events) else 0.0
    )

    demo_day, demo_minute = None, None
    config.ECOSYSTEM_STATE_JSON.parent.mkdir(parents=True, exist_ok=True)

    if len(events):
        demo = events.sort_values("n_downstream", ascending=False).iloc[0]
        demo_day, demo_minute = int(demo["day_number"]), int(demo["minute_of_day"])
        snap = timeline[
            (timeline["day_number"] == demo_day) & (timeline["minute_of_day"] == demo_minute)
        ]
        ecosystem_state = {
            int(r.LINK_ID): {
                "state": r.state,
                "regime": r.regime,
                "road_health_score": round(float(r.road_health_score), 2),
            }
            for r in snap.itertuples()
        }
        config.ECOSYSTEM_STATE_JSON.write_text(
            json.dumps(
                {"day_number": demo_day, "minute_of_day": demo_minute, "links": ecosystem_state},
                indent=2,
            )
        )
    else:
        # Edge Case E: no cascade events (zero graph edges or low-correlation dataset).
        # Write a safe sentinel so B10 from_artifacts() never raises FileNotFoundError.
        # Use the most congested observed interval (lowest mean road_health_score) as the
        # representative snapshot; this gives the LLM layer real metabolic state data.
        worst_interval = (
            timeline.groupby(["day_number", "minute_of_day"])["road_health_score"]
            .mean()
            .idxmin()
        )
        demo_day, demo_minute = int(worst_interval[0]), int(worst_interval[1])
        snap = timeline[
            (timeline["day_number"] == demo_day) & (timeline["minute_of_day"] == demo_minute)
        ]
        ecosystem_state = {
            int(r.LINK_ID): {
                "state": r.state,
                "regime": r.regime,
                "road_health_score": round(float(r.road_health_score), 2),
            }
            for r in snap.itertuples()
        }
        config.ECOSYSTEM_STATE_JSON.write_text(
            json.dumps(
                {
                    "day_number": demo_day,
                    "minute_of_day": demo_minute,
                    "links": ecosystem_state,
                    "_sentinel": True,   # flag: no cascade events found
                    "_note": "No cascade events detected; showing most congested interval.",
                },
                indent=2,
            )
        )
        print(
            f"  [WARN] No cascade events detected. "
            f"Wrote sentinel ecosystem_state.json at day={demo_day} min={demo_minute}."
        )


    return {
        "n_edges": graph.number_of_edges(),
        "edge_36_16": edge_36_16,
        "n_cascade_events": int(len(events)),
        "n_cascade_events_day1": int((events["day_number"] == 1).sum()) if len(events) else 0,
        "validation_rate": validation_rate,
        "demo_event": {"day_number": demo_day, "minute_of_day": demo_minute},
        "events": events,
    }


if __name__ == "__main__":
    out = run()
    print("\n=== B8 ECOSYSTEM STATE MACHINE ===")
    print(f"  causal edges       : {out['n_edges']}")
    if out["edge_36_16"]:
        print(
            f"  36 -> 16 edge      : corr={out['edge_36_16']['correlation_strength']:.3f}"
            f"  lag={out['edge_36_16']['lag_minutes']}min"
        )
    else:
        print("  36 -> 16 edge      : NOT FOUND")
    print(f"  cascade events     : {out['n_cascade_events']}  (Day 1: {out['n_cascade_events_day1']})")
    print(f"  validation rate    : {out['validation_rate']:.1%}")
    passed = out["edge_36_16"] is not None and out["n_cascade_events_day1"] > 0
    print(f"\n  GATE (36->16 edge + July-1 cascade): {'PASS' if passed else 'CHECK'}")

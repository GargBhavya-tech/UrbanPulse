"""Tests for ECHO Stage B — Ecosystem State Machine (Bible §7 Stage B)."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from echo import ecosystem as eco


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def _synthetic_features(n_days: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    intervals_per_day = 48
    minute_step = 1440 // intervals_per_day
    shock = rng.normal(0, 1, n_days * intervals_per_day)
    rows = []
    for link, lag in [(1, 0), (2, 2), (3, 0)]:
        for day in range(1, n_days + 1):
            for k in range(intervals_per_day):
                t = (day - 1) * intervals_per_day + k
                minute = k * minute_step
                diurnal = 200 + 150 * np.sin(2 * np.pi * minute / 1440)
                src_t = t - lag
                shock_term = 80 * shock[src_t] if 0 <= src_t < len(shock) and link != 3 else 0.0
                noise = rng.normal(0, 5)
                queue = max(diurnal + shock_term + noise, 0)
                rows.append({
                    "LINK_ID": link, "day_number": day, "minute_of_day": minute,
                    "mean_queue_s": queue, "mean_occup": min(queue / 600, 1.0),
                    "road_health_score": max(100 - queue / 6, 0),
                })
    return pd.DataFrame(rows)


def _toy_timeline() -> pd.DataFrame:
    rows = []
    states_prev = {30: "Stressed", 36: "Healthy", 16: "Healthy", 5: "Healthy"}
    states_t0 = {30: "Collapsed", 36: "Stressed", 16: "Healthy", 5: "Healthy"}
    states_t1 = {30: "Collapsed", 36: "Stressed", 16: "Stressed", 5: "Healthy"}
    health = {"Healthy": 90, "Stressed": 50, "Saturated": 30, "Collapsed": 10}
    for t, (interval, minute, states) in enumerate([
        (-1, 0, states_prev), (0, 5, states_t0), (1, 10, states_t1)
    ]):
        for link, state in states.items():
            rows.append({
                "LINK_ID": link, "day_number": 1, "minute_of_day": minute,
                "abs_interval": interval, "state": state,
                "regime": eco.get_causal_regime(state, 0.3),
                "road_health_score": health[state], "mean_occup": 0.3,
            })
    return pd.DataFrame(rows)


def _toy_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edge(30, 36, lag_minutes=5, correlation_strength=0.5, discovery_method="x")
    g.add_edge(30, 16, lag_minutes=5, correlation_strength=0.4, discovery_method="x")
    g.add_edge(30, 5,  lag_minutes=5, correlation_strength=0.3, discovery_method="x")
    return g


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_deseasonalize_removes_diurnal_mean() -> None:
    feats = _synthetic_features()
    resid = eco.deseasonalize(feats)
    means = resid.groupby(["LINK_ID", "minute_of_day"])["queue_residual"].mean()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-6)


def test_residual_recovers_direction_raw_confounds() -> None:
    feats = _synthetic_features()
    pivot_raw = feats.pivot_table(
        index=["day_number", "minute_of_day"], columns="LINK_ID", values="mean_queue_s"
    ).sort_index()
    s1, s2 = pivot_raw[1].to_numpy(), pivot_raw[2].to_numpy()
    raw_12, _ = eco.discover_causal_lag(s1, s2, max_lag=6)
    raw_21, _ = eco.discover_causal_lag(s2, s1, max_lag=6)
    # Both directions inflated by shared diurnal (confound exists)
    assert raw_12 > 0.5 and raw_21 > 0.5
    # But residual graph still recovers the correct direction
    graph = eco.build_causal_graph(feats)
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(2, 1)
    assert graph[1][2]["lag_minutes"] > 0


def test_unrelated_link_gets_no_edge() -> None:
    feats = _synthetic_features()
    graph = eco.build_causal_graph(feats)
    assert not graph.has_edge(3, 1)
    assert not graph.has_edge(1, 3)
    assert not graph.has_edge(3, 2)


def test_save_causal_graph_schema(tmp_path) -> None:
    g = nx.DiGraph()
    g.add_edge(36, 16, lag_minutes=5, correlation_strength=0.267, discovery_method="x")
    out = tmp_path / "graph.json"
    eco.save_causal_graph(g, out)
    d = __import__("json").loads(out.read_text())
    assert d["edges"][0].keys() >= {"source", "target", "lag_minutes", "correlation_strength", "discovery_method"}


def test_get_causal_regime() -> None:
    assert eco.get_causal_regime("Healthy", 0.3) == "forward"
    assert eco.get_causal_regime("Stressed", 0.5) == "forward"
    assert eco.get_causal_regime("Saturated", 0.3) == "backpressure"
    assert eco.get_causal_regime("Healthy", 0.85) == "backpressure"


def test_active_graph_reverses_source_in_backpressure() -> None:
    g = nx.DiGraph()
    g.add_edge(36, 16, lag_minutes=5, correlation_strength=0.27, discovery_method="x")
    fwd = eco.active_graph(g, {36: "forward", 16: "forward"})
    assert fwd.has_edge(36, 16)
    bp = eco.active_graph(g, {36: "backpressure", 16: "forward"})
    assert bp.has_edge(16, 36)
    assert not bp.has_edge(36, 16)


def test_detect_transitions_finds_fresh_collapse() -> None:
    tl = _toy_timeline()
    tr = eco.detect_transitions(tl)
    assert (tr["LINK_ID"] == 30).any()
    assert not ((tr["LINK_ID"] == 36) & (tr["abs_interval"] == 0)).any()


def test_cascade_bfs_predicts_and_validates() -> None:
    tl = _toy_timeline()
    g = _toy_graph()
    downstream = eco._cascade_bfs(g, tl, source=30, t0=0)
    ids = {d["link_id"] for d in downstream}
    assert ids == {36, 16, 5}
    by_id = {d["link_id"]: d for d in downstream}
    assert by_id[16]["validated"] is True
    assert by_id[5]["validated"] is False


def test_cascade_events_emits_when_threshold_met() -> None:
    tl = _toy_timeline()
    g = _toy_graph()
    events = eco.cascade_events(g, tl)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["source_link"] == 30
    assert row["n_downstream"] == 3
    assert row["n_validated"] == 2  # link 16 and 36 both reach Stressed at t1


def test_cascade_propagating_map() -> None:
    tl = _toy_timeline()
    g = _toy_graph()
    events = eco.cascade_events(g, tl)
    m = eco.cascade_propagating_map(events, day_number=1, minute_of_day=5)
    assert m == {30: True}
    assert eco.cascade_propagating_map(events, day_number=1, minute_of_day=999) == {}


@pytest.mark.skipif(not config.FEATURES_PARQUET.exists(), reason="features.parquet not built")
def test_run_produces_artifacts_on_real_data() -> None:
    out = eco.run()
    assert out["n_edges"] > 0
    assert config.CAUSAL_GRAPH_JSON.exists()
    assert config.CASCADE_EVENTS_CSV.exists()

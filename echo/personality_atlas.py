"""ECHO Stage A — Personality Atlas (Bible §7 Stage A).

Assigns every road link a behavioral archetype from its 14-day temporal
fingerprint, using spatial-DEC-inspired clustering: an 8-dimensional fingerprint
per link plus a spatial penalty derived from a lag-correlation adjacency matrix
(the two-pass resolution of ticket #9 — the cheap adjacency is computed here and
the full causal graph is built later in Stage B).

MVB clustering (ticket #8): weighted k-means where spatial structure is folded in
as a graph (Laplacian-eigenmap) embedding scaled by alpha, then concatenated with
the standardized fingerprint. This realizes ``L_cluster + alpha * L_spatial`` in a
deterministic, CPU-cheap, Euclidean-k-means-friendly form. A DEC autoencoder is
the documented upgrade path.

Outputs:
- ``data/road_archetypes.json`` — per link: archetype, confidence, stability
- ``reports/echo/personality_atlas.png`` — 2D projection coloured by archetype
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# echo/ is a subpackage; ensure the repo root (where config.py lives) is importable
# whether this module is imported via a script or run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

import config

_RNG = 42
FINGERPRINT_COLS = [
    "peak_severity",
    "off_peak_floor",
    "spike_frequency",
    "recovery_speed",
    "lane6_utilization",
    "stall_rate",
    "speed_divergence_mean",
    "occupancy_variance",
]


# --------------------------------------------------------------------------- #
# Step 1 — fingerprints
# --------------------------------------------------------------------------- #
def extract_fingerprints(features: pd.DataFrame) -> pd.DataFrame:
    """Compute the 8-dim temporal fingerprint per link (Bible §7 Step 1).

    Args:
        features: Full feature frame (B2 output).

    Returns:
        Frame indexed by ``LINK_ID`` with the eight fingerprint columns.
    """
    df = features.sort_values(["LINK_ID", "date"]).copy()
    df["queue_delta"] = df.groupby("LINK_ID")["mean_queue_s"].diff()
    df["stalled"] = ((df["lane4_stalled"] == 1) | (df["lane5_stalled"] == 1)).astype(int)

    rows = []
    for link, g in df.groupby("LINK_ID"):
        peak = g.loc[g["hour"].isin([8, 9, 10]), "mean_queue_s"]
        off = g.loc[g["hour"].isin([1, 2, 3, 4, 5]), "mean_queue_s"]
        neg = g.loc[g["queue_delta"] < 0, "queue_delta"]
        rows.append(
            {
                "LINK_ID": int(link),
                "peak_severity": float(peak.max()) if len(peak) else 0.0,
                "off_peak_floor": float(off.mean()) if len(off) else 0.0,
                "spike_frequency": float((g["mean_queue_s"] > 400).mean()),
                "recovery_speed": float(neg.mean()) if len(neg) else 0.0,
                "lane6_utilization": float(g["lane6_active"].mean()),
                "stall_rate": float(g["stalled"].mean()),
                "speed_divergence_mean": float(g["mean_speed_div"].mean()),
                "occupancy_variance": float(g["mean_occup"].std()),
            }
        )
    return pd.DataFrame(rows).set_index("LINK_ID").sort_index()


# --------------------------------------------------------------------------- #
# Step 2 — lag-correlation adjacency (two-pass seed for Stage B)
# --------------------------------------------------------------------------- #
def build_adjacency(features: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Binary spatial adjacency from max lagged queue correlation (Bible §7 Step 2).

    For each link pair, take the max absolute cross-correlation of ``mean_queue_s``
    over lags 0..``ATLAS_LAG_MAX`` and threshold it.

    Args:
        features: Full feature frame.

    Returns:
        Tuple ``(A, link_ids)`` — A is the 66x66 binary adjacency, link_ids the
        row/column order.
    """
    pivot = features.pivot_table(
        index=["day_number", "minute_of_day"], columns="LINK_ID", values="mean_queue_s"
    ).sort_index()
    link_ids = [int(c) for c in pivot.columns]
    x = pivot.to_numpy()
    n = x.shape[1]

    best = np.zeros((n, n))
    for lag in range(config.ATLAS_LAG_MAX + 1):
        if lag == 0:
            a, b = x, x
        else:
            a, b = x[lag:], x[:-lag]
        mask = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
        a, b = a[mask], b[mask]
        az = (a - a.mean(0)) / (a.std(0) + 1e-9)
        bz = (b - b.mean(0)) / (b.std(0) + 1e-9)
        corr = np.abs(az.T @ bz) / len(a)  # cross-correlation matrix
        best = np.maximum(best, corr)

    np.fill_diagonal(best, 0.0)
    adj = (best >= config.ATLAS_ADJ_THRESHOLD).astype(float)
    adj = np.maximum(adj, adj.T)  # symmetrise
    return adj, link_ids


# --------------------------------------------------------------------------- #
# Step 3 — spatial-penalty k-means (MVB)
# --------------------------------------------------------------------------- #
def cluster(
    fingerprints: pd.DataFrame, adjacency: np.ndarray
) -> dict[str, Any]:
    """Weighted k-means with a spatial-graph penalty (Bible §7 Step 3, MVB).

    Args:
        fingerprints: 66x8 fingerprint frame (index = LINK_ID).
        adjacency: 66x66 binary adjacency aligned to ``fingerprints.index``.

    Returns:
        Dict with ``labels`` (Series), ``centroids``, ``silhouette``, ``confidence``
        (Series), and the standardized fingerprint matrix ``fz``.
    """
    # RobustScaler (median/IQR) guards against outlier links that inflate
    # the variance of queue-based fingerprint dimensions and pull centroids.
    fz = RobustScaler().fit_transform(fingerprints.to_numpy())
    # Additionally standardize so all 8 dimensions have unit variance after
    # the robust centering — this gives k-means equal weight per dimension.
    fz = StandardScaler(with_mean=False).fit_transform(fz)

    # Graph embedding folds spatial connectivity into the Euclidean space.
    # Disabled when ATLAS_ALPHA<=0 (B7 decision: fingerprint-only clustering).
    if adjacency.sum() > 0 and config.ATLAS_ALPHA > 0:
        emb = SpectralEmbedding(
            n_components=config.ATLAS_SPECTRAL_DIMS,
            affinity="precomputed",
            random_state=_RNG,
        ).fit_transform(adjacency)
        emb = StandardScaler().fit_transform(emb)
        combined = np.hstack([fz, np.sqrt(config.ATLAS_ALPHA) * emb])
    else:
        combined = fz

    km = KMeans(n_clusters=config.ATLAS_K, n_init=10, random_state=_RNG)
    labels = km.fit_predict(combined)

    sil = float(silhouette_score(combined, labels)) if len(set(labels)) > 1 else 0.0
    sample_sil = silhouette_samples(combined, labels)
    confidence = np.clip((sample_sil + 1) / 2, 0, 1)  # map [-1,1] -> [0,1]

    return {
        "labels": pd.Series(labels, index=fingerprints.index, name="cluster"),
        "centroids": km.cluster_centers_,
        "silhouette": sil,
        "confidence": pd.Series(confidence, index=fingerprints.index, name="confidence"),
        "fz": fz,
        "combined": combined,
    }


# --------------------------------------------------------------------------- #
# Step 4 — archetype assignment (data-driven, signature matching)
# --------------------------------------------------------------------------- #
def assign_archetypes(
    fingerprints: pd.DataFrame, labels: pd.Series
) -> dict[int, str]:
    """Name each cluster (Bible §7 Step 4): anchor on documented exemplars, then
    signature-assign the remaining clusters.

    The clustering is unsupervised; only the *naming* uses the Bible's documented
    exemplar links (``config.ARCHETYPE_ANCHORS``: Link 37=Chronic, 36=Landmine,
    5=Saturator). Remaining clusters are named by their centroid signature:
    highest ghost-lane usage -> Ghost, largest clean/low-baseline -> Commuter, the
    rest -> Chameleon. This avoids brittle pure-signature naming (which mislabeled
    the textbook Chronic road).

    Args:
        fingerprints: 66x8 fingerprint frame.
        labels: Cluster label per link.

    Returns:
        Map ``cluster_id -> archetype name``.
    """
    fz = pd.DataFrame(
        StandardScaler().fit_transform(fingerprints),
        index=fingerprints.index,
        columns=fingerprints.columns,
    )
    profiles = fz.groupby(labels).mean()
    all_clusters = list(profiles.index)

    assignment: dict[int, str] = {}
    used: set[int] = set()

    # 1) Anchor clusters that contain a documented exemplar link.
    for link, arch in config.ARCHETYPE_ANCHORS.items():
        if link in labels.index:
            cid = int(labels.loc[link])
            if cid not in used and arch not in assignment.values():
                assignment[cid] = arch
                used.add(cid)

    remaining = [c for c in all_clusters if c not in used]

    # 2) Ghost — highest ghost-lane utilisation among remaining.
    if remaining and "Ghost" not in assignment.values():
        ghost = max(remaining, key=lambda c: profiles.loc[c, "lane6_utilization"])
        assignment[ghost] = "Ghost"
        used.add(ghost)
        remaining = [c for c in remaining if c != ghost]

    # 3) Commuter — largest remaining cluster with a clean low baseline.
    sizes = labels.value_counts()
    if remaining and "Commuter" not in assignment.values():
        commuter = max(
            remaining,
            key=lambda c: sizes.get(c, 0) - profiles.loc[c, "off_peak_floor"],
        )
        assignment[commuter] = "Commuter"
        used.add(commuter)
        remaining = [c for c in remaining if c != commuter]

    # 4) Anything still unnamed -> Chameleon.
    for cid in all_clusters:
        assignment.setdefault(int(cid), "Chameleon")
    return {int(k): v for k, v in assignment.items()}


# --------------------------------------------------------------------------- #
# Step 5 — stability across days
# --------------------------------------------------------------------------- #
def stability_scores(
    features: pd.DataFrame, centroids: np.ndarray, labels: pd.Series, adjacency: np.ndarray
) -> pd.Series:
    """Fraction of days each link stays in its assigned archetype (Bible §7 Step 5).

    For each day, a daily fingerprint is computed and assigned to the nearest
    overall centroid (fingerprint block only); stability is the share of the 14
    days that match the link's primary cluster.

    Args:
        features: Full feature frame.
        centroids: Cluster centroids from :func:`cluster` (combined space).
        labels: Primary cluster per link.
        adjacency: Adjacency (unused for daily assignment; kept for signature).

    Returns:
        Series of stability scores in [0, 1] indexed by LINK_ID.
    """
    # Use only the fingerprint block of the centroids for daily nearest-centroid.
    fp_dim = len(FINGERPRINT_COLS)
    cent_fp = centroids[:, :fp_dim]
    scaler = StandardScaler().fit(extract_fingerprints(features).to_numpy())

    match_counts = {link: 0 for link in labels.index}
    day_count = 0
    for day, g in features.groupby("day_number"):
        day_count += 1
        fp_day = extract_fingerprints(g)
        fp_day = fp_day.reindex(labels.index)  # align all links
        if fp_day.isna().all(axis=1).any():
            # Some links have no rows for this day (sensor outage / complete stall).
            # col_means can itself be NaN if the entire column is absent for this day.
            # Fall back to 0.0 for those columns so scaler.transform never receives NaN.
            col_means = fp_day.mean()
            col_means = col_means.fillna(0.0)
            fp_day = fp_day.fillna(col_means)
        fz_day = scaler.transform(fp_day.to_numpy())
        # nearest centroid by fingerprint block
        d = np.linalg.norm(fz_day[:, None, :] - cent_fp[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        for link, nc in zip(labels.index, nearest):
            if nc == labels.loc[link]:
                match_counts[link] += 1

    return pd.Series(
        {link: match_counts[link] / max(day_count, 1) for link in labels.index},
        name="stability_score",
    ).sort_index()


# --------------------------------------------------------------------------- #
# Plot + orchestrator
# --------------------------------------------------------------------------- #
def plot_atlas(combined: np.ndarray, archetypes: pd.Series, out_dir) -> "Path":
    """2D PCA projection of links coloured by archetype (UMAP-style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = PCA(n_components=2, random_state=_RNG).fit_transform(combined)
    fig, ax = plt.subplots(figsize=(8, 6))
    for arch in sorted(archetypes.unique()):
        m = (archetypes == arch).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], label=arch, s=60, alpha=0.8)
    for i, link in enumerate(archetypes.index):
        ax.annotate(str(link), (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.6)
    ax.set_title("Personality Atlas — 66 links by archetype (2D PCA)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "personality_atlas.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run() -> dict[str, Any]:
    """Full Stage A: fingerprints -> adjacency -> cluster -> archetypes -> stability.

    Writes ``data/road_archetypes.json`` and the atlas plot.

    Returns:
        Summary dict for the B7 gate (silhouette, n_archetypes, counts).
    """
    import io_utils

    features = io_utils.load_parquet(config.FEATURES_PARQUET)

    print("Step 1: fingerprints ...")
    fp = extract_fingerprints(features)

    print("Step 2: lag-correlation adjacency ...")
    adj, link_ids = build_adjacency(features)
    fp = fp.reindex(link_ids)  # align order to adjacency
    print(f"  adjacency edges: {int(adj.sum() / 2)}")

    print("Step 3: fingerprint k-means (spatial penalty disabled, B7 decision) ...")
    cl = cluster(fp, adj)
    print(f"  silhouette: {cl['silhouette']:.3f}")

    print("Step 4: archetype assignment ...")
    cluster_to_arch = assign_archetypes(fp, cl["labels"])
    archetypes = cl["labels"].map(cluster_to_arch).rename("archetype")

    print("Step 5: stability ...")
    stab = stability_scores(features, cl["centroids"], cl["labels"], adj)

    atlas_png = plot_atlas(cl["combined"], archetypes, config.ECHO_REPORTS_DIR)

    # Assemble road_archetypes.json
    record = {}
    for link in fp.index:
        record[str(int(link))] = {
            "archetype": archetypes.loc[link],
            "confidence": round(float(cl["confidence"].loc[link]), 4),
            "stability_score": round(float(stab.loc[link]), 4),
            "stable": bool(stab.loc[link] >= config.STABILITY_THRESHOLD),
        }
    config.ROAD_ARCHETYPES_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.ROAD_ARCHETYPES_JSON.write_text(json.dumps(record, indent=2))

    counts = archetypes.value_counts().to_dict()
    return {
        "silhouette": cl["silhouette"],
        "n_archetypes": int(archetypes.nunique()),
        "archetype_counts": counts,
        "n_stable": int((stab >= config.STABILITY_THRESHOLD).sum()),
        "atlas_plot": str(atlas_png),
        "archetypes": archetypes,
        "fingerprints": fp,
    }


if __name__ == "__main__":
    out = run()
    print("\n=== B7 PERSONALITY ATLAS ===")
    print(f"  silhouette       : {out['silhouette']:.3f}")
    print(f"  archetypes       : {out['n_archetypes']}")
    print(f"  counts           : {out['archetype_counts']}")
    print(f"  stable links     : {out['n_stable']}/{config.EXPECTED_LINKS}")
    passed = 5 <= out["n_archetypes"] <= 7 and out["silhouette"] > 0.5
    print(f"\n  GATE (5-7 archetypes, silhouette>0.5): {'PASS' if passed else 'CHECK'}")

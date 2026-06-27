"""Exploratory data analysis outputs for UrbanPulse (Bible §5 Notebook 01).

Pure analytical functions return DataFrames; plotting functions write PNGs to
``reports/eda/``. The road-level aggregate helpers here are intentionally small
and local — the canonical feature definitions live in ``features.py`` (B2).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# Local road-level aggregates (minimal; full feature set is B2)
# --------------------------------------------------------------------------- #
def _road_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the few road-level aggregates the EDA tables need.

    Args:
        df: Cleaned frame.

    Returns:
        Frame with ``mean_occup`` (driving lanes), ``mean_queue_s`` (all lanes),
        ``mean_speed_kmh`` (driving lanes), and a row-level ``congested`` flag.
    """
    occ = df[config.metric_cols("occup", config.DRIVING_LANES)].mean(axis=1)
    queue = df[config.metric_cols("queue")].mean(axis=1)
    speed = df[config.metric_cols("arith", config.DRIVING_LANES)].mean(axis=1)
    congested = (
        (occ > config.CONGESTION_OCCUP_THRESHOLD)
        & (queue > config.CONGESTION_QUEUE_THRESHOLD_S)
    ).astype("int64")
    return pd.DataFrame(
        {
            "LINK_ID": df["LINK_ID"].to_numpy(),
            "hour": df["hour"].to_numpy(),
            "day_number": df["day_number"].to_numpy(),
            "total_vehs": df[config.metric_cols("vehs")].sum(axis=1).to_numpy(),
            "mean_occup": occ.to_numpy(),
            "mean_queue_s": queue.to_numpy(),
            "mean_speed_kmh": speed.to_numpy(),
            "congested": congested.to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype and missing-value counts.

    Args:
        df: Any frame.

    Returns:
        Frame indexed by column with ``dtype`` and ``missing`` columns.
    """
    return pd.DataFrame(
        {"dtype": df.dtypes.astype(str), "missing": df.isna().sum()}
    ).reset_index(names="column")


def occupancy_exceedance(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Count rows with occupancy >1.0 per lane, on the RAW (pre-cap) data.

    Args:
        raw_df: The raw frame before ``cap_occupancy`` was applied.

    Returns:
        Frame with ``lane``, ``rows_over_1`` and ``max_value`` per lane.
    """
    rows = []
    for lane in config.LANES:
        col = f"{config.METRIC_PREFIXES['occup']}_{lane}"
        series = raw_df[col]
        rows.append(
            {
                "lane": lane,
                "rows_over_1": int((series > config.OCCUPANCY_CAP).sum()),
                "max_value": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def per_link_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Per-link congestion table, ranked by mean queue delay (all 66 links).

    Args:
        df: Cleaned frame.

    Returns:
        Frame with mean queue, mean occupancy, mean speed and congested-period
        share for each link, sorted by mean queue descending.
    """
    agg = _road_aggregates(df)
    table = (
        agg.groupby("LINK_ID")
        .agg(
            mean_queue_s=("mean_queue_s", "mean"),
            mean_occup=("mean_occup", "mean"),
            mean_speed_kmh=("mean_speed_kmh", "mean"),
            congested_pct=("congested", "mean"),
        )
        .reset_index()
    )
    table["congested_pct"] = (table["congested_pct"] * 100).round(2)
    return table.sort_values("mean_queue_s", ascending=False).reset_index(drop=True)


def lane6_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Lane 6 activity: zero-rate and mean speed when active vs. inactive.

    Args:
        df: Cleaned frame (speeds already in km/h).

    Returns:
        One-row frame summarising Lane 6 behaviour.
    """
    speed6 = df[f"{config.METRIC_PREFIXES['arith']}_6"]
    active = df["lane6_active"] == 1
    return pd.DataFrame(
        [
            {
                "active_rate": float(active.mean()),
                "zero_rate": float((~active).mean()),
                "mean_speed_active_kmh": float(speed6[active].mean()),
                "mean_speed_inactive_kmh": float(speed6[~active].mean()),
            }
        ]
    )


def hourly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Mean vehicles, queue, and occupancy by hour of day.

    Args:
        df: Cleaned frame.

    Returns:
        Frame indexed 0-23 with mean metrics per hour.
    """
    agg = _road_aggregates(df)
    return (
        agg.groupby("hour")
        .agg(
            mean_vehs=("total_vehs", "mean"),
            mean_queue_s=("mean_queue_s", "mean"),
            mean_occup=("mean_occup", "mean"),
        )
        .reset_index()
    )


def daily_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Mean volume and queue by day number (1-14).

    Args:
        df: Cleaned frame.

    Returns:
        Frame indexed by ``day_number`` with mean metrics per day.
    """
    agg = _road_aggregates(df)
    return (
        agg.groupby("day_number")
        .agg(
            mean_vehs=("total_vehs", "mean"),
            mean_queue_s=("mean_queue_s", "mean"),
        )
        .reset_index()
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of road-level numeric aggregates.

    Computed on aggregates rather than all 30 raw sensor columns to keep the
    matrix interpretable (the raw 30-col matrix is dominated by lane copies).

    Args:
        df: Cleaned frame.

    Returns:
        Square correlation DataFrame.
    """
    agg = _road_aggregates(df).drop(columns=["LINK_ID", "day_number"])
    return agg.corr(numeric_only=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_metric_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    """Histogram grid: each metric across the six lanes.

    Args:
        df: Cleaned frame.
        out_dir: Directory to write the PNG into.

    Returns:
        Path to the saved figure.
    """
    metrics = list(config.METRIC_PREFIXES.keys())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
    for ax, metric in zip(axes, metrics):
        cols = config.metric_cols(metric)
        df[cols].plot.hist(bins=50, alpha=0.4, ax=ax, legend=False)
        ax.set_title(f"{config.METRIC_PREFIXES[metric]} — lanes 1-6")
    fig.tight_layout()
    path = out_dir / "metric_distributions.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_hourly(df: pd.DataFrame, out_dir: Path) -> Path:
    """Line plot of mean queue and vehicles by hour.

    Args:
        df: Cleaned frame.
        out_dir: Directory to write the PNG into.

    Returns:
        Path to the saved figure.
    """
    hourly = hourly_patterns(df)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(hourly["hour"], hourly["mean_queue_s"], "r-o", label="mean queue (s)")
    ax2.plot(hourly["hour"], hourly["mean_vehs"], "b-s", label="mean vehicles")
    ax1.set_xlabel("hour of day")
    ax1.set_ylabel("mean queue delay (s)", color="r")
    ax2.set_ylabel("mean vehicles", color="b")
    ax1.set_title("Hourly traffic pattern")
    fig.tight_layout()
    path = out_dir / "hourly_pattern.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of the road-level correlation matrix.

    Args:
        df: Cleaned frame.
        out_dir: Directory to write the PNG into.

    Returns:
        Path to the saved figure.
    """
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Road-level correlation matrix")
    fig.tight_layout()
    path = out_dir / "correlation_matrix.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

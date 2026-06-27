"""Central configuration for UrbanPulse: paths, column groups, and constants.

All magic numbers from the Project Bible live here so every module references a
single source of truth. No business logic in this file.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the repo root, i.e. this file's parent)
# --------------------------------------------------------------------------- #
ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "data"
MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ROOT / "reports"
EDA_REPORTS_DIR: Path = REPORTS_DIR / "eda"

RAW_CSV: Path = DATA_DIR / "raw.csv"
CLEANED_PARQUET: Path = DATA_DIR / "cleaned.parquet"
FEATURES_PARQUET: Path = DATA_DIR / "features.parquet"

# --------------------------------------------------------------------------- #
# Dataset structure (Bible §1.3)
# --------------------------------------------------------------------------- #
LANES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
DRIVING_LANES: tuple[int, ...] = (1, 2, 3, 4, 5)  # Lane 6 excluded from aggregates

METRIC_PREFIXES: dict[str, str] = {
    "vehs": "VEHS(ALL)",
    "arith": "SPEEDAVGARITH(ALL)",
    "harm": "SPEEDAVGHARM(ALL)",
    "queue": "QUEUEDELAY(ALL)",
    "occup": "OCCUPRATE(ALL)",
}

ID_COLUMNS: tuple[str, ...] = ("TIMEINT", "date", "LINK_ID", "DAY")
DROP_COLUMNS: tuple[str, ...] = ("TIMEINT",)  # Bible: fully redundant

# --------------------------------------------------------------------------- #
# Cleaning constants (Bible §2)
# --------------------------------------------------------------------------- #
SPEED_SCALE: float = 10.0          # raw speed is 0.1 km/h -> divide by 10
OCCUPANCY_CAP: float = 1.0         # cap sensor-saturated occupancy

# --------------------------------------------------------------------------- #
# Congestion definitions
# --------------------------------------------------------------------------- #
# Descriptive per-link congestion analysis (Bible §2 Finding 3): keeps the
# literal 400 s threshold so the EDA reproduces the Bible's per-link figures.
CONGESTION_OCCUP_THRESHOLD: float = 0.5
CONGESTION_QUEUE_THRESHOLD_S: float = 400.0

# ML binary target (Bible §5 Option A, recalibrated — decision #11). The Bible's
# literal "queue > 400" yields only 0.55% positive dataset-wide (the "13%" in
# §5 was Link 36's per-link rate, mislabeled). We keep occupancy > 0.5 and lower
# the queue cut to 238 s so the dataset-wide positive rate lands at ~13.1%,
# matching the §12.3 B2 gate ("~13% positive") and making ROC-AUC > 0.85
# meaningful rather than trivially inflated.
TARGET_OCCUP_THRESHOLD: float = 0.5
TARGET_QUEUE_THRESHOLD_S: float = 238.0

# --------------------------------------------------------------------------- #
# Forecasting / modeling (Bible §5 Notebook 03; decision #12)
# --------------------------------------------------------------------------- #
INTERVAL_MINUTES: int = 5
HORIZON_INTERVALS: int = 3          # predict 15 min ahead (3 x 5 min)
MODEL_METRICS_CSV: Path = REPORTS_DIR / "model_metrics.csv"
BEST_MODEL_PKL: Path = MODELS_DIR / "best_model.pkl"

# --------------------------------------------------------------------------- #
# Feature engineering (Bible §5 Notebook 02)
# --------------------------------------------------------------------------- #
AM_PEAK_HOURS: tuple[int, ...] = (8, 9)
PM_PEAK_HOURS: tuple[int, ...] = (18, 19)
WEEKEND_DOW_START: int = 5  # 0=Mon ... 5=Sat, 6=Sun

# congestion_index = w_occup*norm_occup + w_queue*norm_queue + w_invspeed*norm_inv_speed
CONGESTION_INDEX_WEIGHTS: dict[str, float] = {
    "occup": 0.40,
    "queue": 0.35,
    "inv_speed": 0.25,
}
FEATURE_NORMS_JSON: Path = DATA_DIR / "feature_norms.json"

# Expected ground-truth shape, used as a post-clean integrity check.
EXPECTED_ROWS: int = 266_112
EXPECTED_LINKS: int = 66


def metric_cols(metric: str, lanes: tuple[int, ...] = LANES) -> list[str]:
    """Return the column names for a metric across the given lanes.

    Args:
        metric: One of the keys in ``METRIC_PREFIXES`` (e.g. ``"occup"``).
        lanes: Lane numbers to include. Defaults to all six.

    Returns:
        Column names like ``["OCCUPRATE(ALL)_1", ...]``.

    Raises:
        KeyError: If ``metric`` is not a known metric key.
    """
    prefix = METRIC_PREFIXES[metric]
    return [f"{prefix}_{n}" for n in lanes]

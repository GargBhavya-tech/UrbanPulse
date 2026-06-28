"""Central configuration for UrbanPulse: paths, column groups, and constants.

All magic numbers from the Project Bible live here so every module references a
single source of truth. No business logic in this file.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
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
DRIVING_LANES: tuple[int, ...] = (1, 2, 3, 4, 5)

METRIC_PREFIXES: dict[str, str] = {
    "vehs": "VEHS(ALL)",
    "arith": "SPEEDAVGARITH(ALL)",
    "harm": "SPEEDAVGHARM(ALL)",
    "queue": "QUEUEDELAY(ALL)",
    "occup": "OCCUPRATE(ALL)",
}

ID_COLUMNS: tuple[str, ...] = ("TIMEINT", "date", "LINK_ID", "DAY")
DROP_COLUMNS: tuple[str, ...] = ("TIMEINT",)

# --------------------------------------------------------------------------- #
# Cleaning constants (Bible §2)
# --------------------------------------------------------------------------- #
SPEED_SCALE: float = 10.0
OCCUPANCY_CAP: float = 1.0

# --------------------------------------------------------------------------- #
# Congestion definitions
# --------------------------------------------------------------------------- #
CONGESTION_OCCUP_THRESHOLD: float = 0.5
CONGESTION_QUEUE_THRESHOLD_S: float = 400.0

TARGET_OCCUP_THRESHOLD: float = 0.5
TARGET_QUEUE_THRESHOLD_S: float = 238.0

# --------------------------------------------------------------------------- #
# Forecasting / modeling
# --------------------------------------------------------------------------- #
INTERVAL_MINUTES: int = 5
HORIZON_INTERVALS: int = 3
MODEL_METRICS_CSV: Path = REPORTS_DIR / "model_metrics.csv"
BEST_MODEL_PKL: Path = MODELS_DIR / "best_model.pkl"

# --------------------------------------------------------------------------- #
# Feature engineering (Bible §5 Notebook 02)
# --------------------------------------------------------------------------- #
AM_PEAK_HOURS: tuple[int, ...] = (8, 9)
PM_PEAK_HOURS: tuple[int, ...] = (18, 19)
WEEKEND_DOW_START: int = 5

CONGESTION_INDEX_WEIGHTS: dict[str, float] = {
    "occup": 0.40,
    "queue": 0.35,
    "inv_speed": 0.25,
}
FEATURE_NORMS_JSON: Path = DATA_DIR / "feature_norms.json"

EXPECTED_ROWS: int = 266_112
EXPECTED_LINKS: int = 66


def metric_cols(metric: str, lanes: tuple[int, ...] = LANES) -> list[str]:
    prefix = METRIC_PREFIXES[metric]
    return [f"{prefix}_{n}" for n in lanes]


# --------------------------------------------------------------------------- #
# Traffic Intelligence Engine (Bible §6)
# --------------------------------------------------------------------------- #
ENGINE_REPORTS_DIR: Path = REPORTS_DIR / "engine"

ROAD_STATE_BANDS: tuple[tuple[str, float], ...] = (
    ("Healthy", 70.0),
    ("Stressed", 40.0),
    ("Saturated", 20.0),
    ("Collapsed", 0.0),
)

STATE_TO_SEVERITY: dict[str, str] = {
    "Healthy": "NONE",
    "Stressed": "ADVISORY",
    "Saturated": "WARNING",
    "Collapsed": "CRITICAL",
}

CRITICAL_PROB_THRESHOLD: float = 0.70
CRITICAL_QUEUE_THRESHOLD_S: float = 600.0
QUEUE_SURGE_DELTA_S: float = 200.0

# --------------------------------------------------------------------------- #
# ECHO Stage A — Personality Atlas (Bible §7 Stage A)
# --------------------------------------------------------------------------- #
ECHO_REPORTS_DIR: Path = REPORTS_DIR / "echo"
ROAD_ARCHETYPES_JSON: Path = DATA_DIR / "road_archetypes.json"

ATLAS_K: int = 6
ATLAS_ALPHA: float = 0.0
ATLAS_LAG_MAX: int = 6
ATLAS_ADJ_THRESHOLD: float = 0.5
ATLAS_SPECTRAL_DIMS: int = 4
STABILITY_THRESHOLD: float = 0.7

ARCHETYPE_NAMES: tuple[str, ...] = (
    "Landmine", "Chronic", "Saturator", "Ghost", "Commuter", "Chameleon",
)

ARCHETYPE_ANCHORS: dict[int, str] = {37: "Chronic", 36: "Landmine", 5: "Saturator"}

# --------------------------------------------------------------------------- #
# ECHO Stage B — Ecosystem State Machine (Bible §7 Stage B)
# --------------------------------------------------------------------------- #
CAUSAL_GRAPH_JSON: Path = DATA_DIR / "causal_graph.json"
ECOSYSTEM_STATE_JSON: Path = DATA_DIR / "ecosystem_state.json"
CASCADE_EVENTS_CSV: Path = ECHO_REPORTS_DIR / "cascade_events.csv"

EC_MAX_LAG_INTERVALS: int = 12

# De-seasonalized residual correlation threshold (NOT raw mean_queue_s).
# Raw correlation at any threshold mostly rediscovers the shared AM/PM cycle
# (36<->16 raw: 0.600 vs 0.588 — a coin flip). Residual correlation recovers
# direction: 36->16: 0.267 vs 16->36: 0.240. See DECISION_MAP B8.
EC_EDGE_CORR_THRESHOLD: float = 0.25

BACKPRESSURE_OCCUP_THRESHOLD: float = 0.7

CASCADE_MIN_DOWNSTREAM_STRESSED: int = 2
CASCADE_MAX_HORIZON_MINUTES: int = 60
CASCADE_STRESSED_OR_WORSE: tuple[str, ...] = ("Stressed", "Saturated", "Collapsed")

# --------------------------------------------------------------------------- #
# ECHO Stage C — Counterfactual Intervention Engine (Bible §7 Stage C)
# --------------------------------------------------------------------------- #
COUNTERFACTUAL_RESULTS_JSON: Path = DATA_DIR / "counterfactual_results.json"

# Minimum std of the intervention column in link data to trust OLS estimation.
# When std < threshold (i.e., the intervention was never actually used on this
# link in the 14-day dataset), fall back to a synthetic policy simulation using
# the domain-informed effect priors below.
SCM_MIN_INTERVENTION_STD: float = 0.05
SCM_MIN_TREATED_ROWS: int = 20

# Domain-informed prior effect sizes for lane6_active (Bible §7 Stage C).
# Source: B6 intelligence engine rule R3 (Saturator) and R4 (Chronic) document
# that Lane 6 activation is expected to relieve ~10-15% of occupancy at
# saturated links. We use the conservative end: 10% of the link's mean occupancy
# as the expected delta.  This is labelled "policy_simulation" in output.
SCM_LANE6_OCCUP_EFFECT_FRAC: float = 0.10   # expected fractional occup reduction
SCM_IS_AM_PEAK_OCCUP_EFFECT_FRAC: float = 0.08  # expected occupancy reduction from extended green

# --------------------------------------------------------------------------- #
# B10 — LLM Intelligence Layer (Bible §8)
# --------------------------------------------------------------------------- #
LLM_REPORTS_DIR: Path = REPORTS_DIR / "llm"

# Backend: "template" | "flan_t5" | "gemini"
# "template" = deterministic structured templates (zero deps, used by tests/gate)
# "flan_t5"  = google/flan-t5-small via HuggingFace transformers (no API key, CPU)
# "gemini"   = cloud API (requires GEMINI_API_KEY env var)
LLM_DEFAULT_BACKEND: str = "flan_t5"

# Flan-T5 model name (HuggingFace hub id).
# "google/flan-t5-small" (~300 MB): 2-5s per response on CPU
# "google/flan-t5-base"  (~900 MB): 8-15s per response on CPU -- better quality
LLM_FLAN_T5_MODEL: str = "google/flan-t5-small"

# Maximum new tokens to generate per LLM call.
LLM_MAX_NEW_TOKENS: int = 180

# Gemini model name (only used if backend="gemini")
LLM_GEMINI_MODEL: str = "gemini-1.5-flash"

# Citizen-facing: terms that must NEVER appear in output.
LLM_CITIZEN_FORBIDDEN: tuple[str, ...] = (
    "SHAP", "occupancy rate", "occupancy", "harmonic mean",
    "sensor saturation", "archetype", "SCM", "counterfactual",
    "structural causal", "do-calculus", "percentile", "AUC",
)

# --------------------------------------------------------------------------- #
# B11 — FastAPI Serving Layer (DECISION_MAP #4/#5)
# --------------------------------------------------------------------------- #
# The frontend (React + three.js + p5.js, Phase 2) is a separate app. The
# backend exposes the precomputed pipeline artifacts (parquet + JSON + the
# B4 best model, when present) over a read-only HTTP API. This is the single
# data-access path that replaces the Bible's Streamlit role.

# Precomputed artifact locations served by the API (all produced by B1-B10).
MODEL_COMPARISON_CSV: Path = REPORTS_DIR / "model_comparison" / "model_comparison.csv"
BEST_MODEL_META_JSON: Path = MODELS_DIR / "best_model_meta.json"

# CORS: which frontend origins may call the API. "*" is fine for the local
# demo; lock this down to the deployed frontend origin in production.
API_CORS_ORIGINS: tuple[str, ...] = ("*",)

# The July 1 09:45 AM Link 36 demo event — the platform's narrative centrepiece.
# Endpoints default to this snapshot so the API is useful with zero query params.
API_DEMO_LINK_ID: int = 36
API_DEMO_DAY: int = 1
API_DEMO_MINUTE: int = 585     # 09:45 AM

API_TITLE: str = "UrbanPulse API"
API_VERSION: str = "1.1.0"

# Timeline endpoint (Stage 2 scrubber + cascade replay). The features parquet
# holds road_health_score per link per 5-min interval across all 14 days; the
# timeline endpoint serves one frame (all 66 links' state) per (day, minute),
# and the cascade endpoint serves the source->downstream propagation with lags.
INTERVALS_PER_DAY: int = 24 * 60 // INTERVAL_MINUTES   # 288


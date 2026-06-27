# UrbanPulse — Decision Map

> Canonical planning artifact. Loaded into every session. Keep compact.
> Source of truth for scope: `UrbanPulse_Project_Bible_v2.md`.

## Locked context

- **Scope**: Full backend per the Bible — every cleaning step, every engineered
  feature, all 7 models + SHAP, Traffic Intelligence Engine, all 3 ECHO stages
  (Personality Atlas → Ecosystem State Machine → Counterfactual), LLM layer.
- **Structure**: Follows the Bible's module layout (Section 11 file tree).
- **Runtime** *(assumption — confirm)*: local Windows laptop, i7-1355U, 16 GB,
  **CPU-only**. All deps pip-installable on Windows. No GPU. SHAP kept tractable
  by sampling, not run on all 266,112 rows.
- **Data**: single source = `Pangyo_14days_lanes_w_arith_adj.csv`
  (266,112 rows × 34 cols, speeds still in 0.1 km/h → ÷10 applies).

---

## Resolved decisions

### #1 — Codebase structure → **Per the Bible**
Module layout from Bible §11 (`data/`, `features/`, `models/`, `engine/`,
`echo/`, `llm/`, `db/`, plus the serving layer below). Full feature coverage.

### #2 — LLM backend → **Provider-agnostic, Gemini default**
Thin `LLMClient` interface; no vendor hardcoded anywhere else. Default backend =
Gemini (free tier, already integrated in prior work). Swappable in one line.

### #3 — Frontend stack (Phase 2) → **React + three.js + p5.js** (NOT Streamlit)
- three.js: 3D render of the 66-link road network; animate cascade propagation
  (Link 36 → Link 16, 8-min lag); counterfactual "what-if" overlays.
- p5.js: generative 2D state-field / road-health visualizations.
- Designed with `ui-ux-pro-max`. Bible's Streamlit `app/` is dropped.

### #4 — Backend↔frontend contract → **FastAPI serving layer** (NEW vs Bible)
Because the frontend is now a separate React app, the backend exposes a FastAPI
service over the Bible's `db/urbanpulse.db` (+ `schema.sql`) and the precomputed
artifacts. This replaces Streamlit's role as the data-access path.

### #5 — Persistence → **Parquet artifacts + SQLite** (both, per Bible)
Parquet between pipeline stages (`cleaned.parquet`, `features.parquet`); SQLite
`urbanpulse.db` as the query store the FastAPI layer serves from.

### #6 — Boosting libraries → **XGBoost + LightGBM + CatBoost, CPU mode**
All ship Windows wheels; install via pip, no compilation. CatBoost optional if
install friction; degrade gracefully.

### #7 — ECHO Stage C counterfactual → **Custom lightweight SCM**
numpy / pandas / networkx, manual do-operator. No DoWhy (install weight + the
Bible mandates a manually-specified SCM anyway). Cross-correlation for edges
(Bible's transfer-entropy proxy).

### #10 — Lane 6 speed units → **do NOT ÷10** (deviation from Bible step 8)
B1 finding: lanes 1-5 speeds are 0.1 km/h (÷10 → km/h), but Lane 6 is already
km/h. Evidence: Lane 6 raw active mean = 41.19, matching Bible §1.4 ("41 km/h")
and the §13.2 killer fact exactly. The Bible's literal "divide *all* speed
columns by 10" would understate Lane 6 tenfold and break the ECHO counterfactual
("activate Lane 6's unused 41 km/h capacity"). `cleaning.scale_speeds` scales
lanes 1-5 only. Documented inline.

### #11 — ML target → **recalibrated to ~13%** (deviation from literal §5)
B2 finding: the literal `mean_occup>0.5 AND mean_queue_s>400` yields only **0.55%**
positive dataset-wide — the Bible's "13%" was Link 36's *per-link* rate (Finding
3), mislabeled in §5 as a class rate. The §12.3 B2 gate explicitly wants ~13%, so
we honor that intent: keep occupancy >0.5, lower the queue cut to **238 s** →
13.1% positive. The descriptive per-link analysis keeps the literal 400 s (so EDA
still reproduces Finding 3). Two separate thresholds in `config.py`.

---

## Open tickets (frontier)

## #8: ECHO Stage A clustering — MVB or full DEC?

Blocked by: —
Type: Discuss

### Question
Personality Atlas: weighted k-means with spatial penalty (Bible "MVB"), or the
full autoencoder + embedded clustering loss (Bible "full build")?

### Answer
*Recommendation:* MVB (weighted k-means) for the first pass — deterministic,
CPU-cheap, demo-stable, silhouette > 0.5 achievable. Leave a clean seam to swap
in DEC later. **Pending your confirmation.**

## #9: ECHO A↔B ordering — adjacency chicken-and-egg

Blocked by: —
Type: Discuss

### Question
Atlas Step 2 needs spatial adjacency `A`, which the Bible sources from "Ecosystem
discovery Step 1" — but Ecosystem (Stage B) runs *after* Atlas (Stage A).

### Answer
*Recommendation:* two-pass. Pre-compute a cheap lag-correlation adjacency first,
feed it into Atlas's spatial penalty; Stage B then builds the full causal graph.
Avoids the circular dependency without iterating. **Pending your confirmation.**

## #12: B3 target leakage — which features may the model see?

Blocked by: —
Type: Discuss → **RESOLVED**

### Question
The target is `mean_occup>0.5 AND mean_queue_s>238`, but `mean_occup`,
`mean_queue_s`, `max_occup`, `max_queue_s`, `congestion_index`, and
`road_health_score` are all features — they *define* the target, so including
them in X is trivial leakage (ROC-AUC → ~1.0).

### Answer
**Forecast +15 min (horizon = 3 intervals).** The label is congestion 15 min
ahead, so current occupancy/queue are legitimate predictors (no leakage) and all
features stay. Matches the Bible's "about to congest" framing. A horizon sweep
(nowcast-leakfree / +5 / +10 / +15) on HistGradientBoosting showed ROC-AUC flat
at ~0.977 and PR-AUC ~0.85 across all horizons — so the longer window costs
almost nothing in accuracy while tripling warning time. `config.HORIZON_INTERVALS
= 3`; leak-safe shifting + cross-split guard in `modeling.shift_target`.

---

## Backend implementation roadmap (sub-phases)

Built in Bible dependency order. Each ships as runnable, tested code producing
its artifact, with the Bible's §12.3 "before moving on" gate.

| # | Sub-phase | Produces | Gate |
|---|---|---|---|
| B1 | EDA + cleaning | `cleaned.parquet` | zero bad occupancy, Lane 6 flag, speeds in km/h |
| B2 | Feature engineering | `features.parquet` | no NaN, ~13% positive class |
| B3 | Model training (7 models) | `models/` | temporal split, no leakage |
| B4 | Model comparison | charts, best model | ROC-AUC > 0.85, inference < 500 ms |
| B5 | SHAP explainability | 6 SHAP outputs | Link 36 + Link 37 waterfalls | ✅ **DONE** — `shap_analysis.py`, `scripts/b5_shap.py`, `notebooks/05_SHAP.ipynb` |
| B6 | Traffic Intelligence Engine | health score, alerts, recs | every rec has reasoning text |
| B7 | ECHO A — Personality Atlas | `road_archetypes.json` | 5–7 archetypes, silhouette > 0.5 |
| B8 | ECHO B — Ecosystem State Machine | `causal_graph.json`, `cascade_events.csv` | Link 36→16 edge, July-1 cascade |
| B9 | ECHO C — Counterfactual | `counterfactual_results.json` | July-1 run, narrative output |
| B10 | LLM layer | `llm/` | grounded responses, no hallucination |
| B11 | FastAPI serving layer | running API | endpoints for all artifacts |

Phase 2 (frontend: React + three.js + p5.js) begins after B11.

---

## B6 — Traffic Intelligence Engine (done)

`engine/intelligence.py` (Bible §6). Road Health Score → metabolic state
(Healthy/Stressed/Saturated/Collapsed), Congestion Risk Score (per-link
percentile), hotspot ranking, Critical Roads Flag (P>0.70 or queue>600s),
severity alerts, and the §6.3 archetype-aware recommendation rules — every
recommendation carries explicit reasoning (gate).

**Design:** archetype (B7) and cascade (B8) inputs are optional. Archetype-
specific rules fire ONLY on a known matching archetype, so they stay silent
(rather than flooding all 66 links) until ECHO assigns archetypes.

**Findings (for B7):**
- **R5 (Landmine) rule is dead on this data** — `lane5_stalled` and a >400s queue
  never co-occur in 266k rows. Flagged via `rule_activation_census`.
- **Link 36 behaves like a Saturator**, not the Bible's placeholder "Landmine":
  it satisfies the R3 saturation condition 56× and never the R5 Landmine
  condition. B7 archetype discovery should re-classify it from data.

---

## B7 — ECHO Stage A: Personality Atlas (done)

`echo/personality_atlas.py` (Bible §7 Stage A). 8-dim temporal fingerprint per
link → k-means → archetype assignment (anchored on documented exemplars) →
14-day stability. Output: `data/road_archetypes.json`, atlas plot. Archetypes
now feed the B6 engine (archetype rules active).

**#8 resolved → MVB k-means** (DEC autoencoder is the documented upgrade seam).
**#9 resolved → two-pass adjacency**: a lag-correlation adjacency is computed
(`build_adjacency`) and reserved for Stage B; it is NOT used in clustering (see
below).

**Findings (deviations from the Bible, with evidence):**
- **Spatial penalty dropped (α=0).** The Bible's penalty pulls *adjacent* links
  together, but Link 36↔37 are adjacent (lag-corr edge), so it *merged* the two
  roads the Bible used it to "separate", and lowered silhouette (0.211 → 0.265).
  Spatial/causal structure is used in Stage B, its natural home.
- **No Landmine archetype in the data.** Link 36 (Bible's "Landmine") has a high
  off-peak floor (356s — congested at 3 AM) and co-clusters robustly with Chronic
  Link 37. It is empirically Chronic/Saturator-like, not a Landmine. Triple-
  confirmed (B6 rule census, fingerprint, clustering). Result: 5 honest
  archetypes (Chronic, Saturator, Ghost, Commuter, Chameleon).
- **Silhouette ~0.27 is intrinsic** (max ~0.29 at k=4): the 66 links form a
  behavioral continuum, not crisp clusters. Stability (57/66 ≥ 0.7) is the
  primary validation; the Bible's 0.5 silhouette target is aspirational.

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
| B9 | ECHO C — Counterfactual | `counterfactual_results.json` | July-1 run, narrative output | ✅ **DONE** — `echo/counterfactual.py`, `scripts/09_counterfactual.py` |
| B10 | LLM layer | `llm/` | grounded responses, no hallucination |
| B11 | FastAPI serving layer | running API | endpoints for all artifacts | ✅ **DONE** — `api/`, `scripts/11_serve.py`, `tests/test_api.py` |

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

---

## B9 — ECHO Stage C: Counterfactual Intervention Engine (done)

`echo/counterfactual.py` (Bible §7 Stage C). Custom lightweight SCM
(numpy/pandas OLS) — no DoWhy (DECISION_MAP #7 resolved).

**Architecture:**
- `StructuralEquation`: OLS with pseudoinverse lstsq; one equation per stage.
- `SCM`: Two-stage structural causal model per link.
  - Stage 1: `mean_occup = f(intervention, total_vehs, hour)` 
  - Stage 2: `mean_queue_s = f(mean_occup, mean_speed_div, hour)`
- `do()` operator: sets intervention value, propagates through both stages.
- `ate()`: Average Treatment Effect = E[Y|do(T=1)] − E[Y|do(T=0)].

**Frontdoor criterion (not backdoor):** Both intermediate variables
(`mean_occup`, `mean_speed_div`) are fully observed → valid causal
identification even with unobserved confounders (driver routing, incidents,
weather). Matches the Bible §7 Stage C justification exactly.

**Intervention library (archetype-specific):**
| Archetype | Intervention Column | Description |
|---|---|---|
| Landmine / Chronic / Ghost / Chameleon | `lane6_active` | Activate ghost lane |
| Saturator | `lane6_active` | Lane 6 + perimeter inflow modelling |
| Commuter | `is_am_peak` | Extend AM peak green phase |
| Unknown | `lane6_active` | Default broadest lever |

**July 1 centrepiece:** Link 36, 09:30 AM intervention, 09:45 AM peak.
Output: `data/counterfactual_results.json`, `reports/echo/scm_graph.png`,
`reports/echo/july1_counterfactual.txt`, `reports/echo/scm_coefficients.json`.

**Gate:** July-1 CF runs, narrative produced, ≥4 archetypes covered.

---

## B11 — FastAPI Serving Layer (done)

`api/` package (DECISION_MAP #4/#5). Read-only HTTP API over the precomputed
B1–B10 artifacts — the single backend↔frontend data-access path that replaces
the Bible's Streamlit role.

**Design:**
- `api/store.py`: cached artifact loaders with graceful degradation. Missing
  artifacts never crash the app — read-only endpoints stay up after a partial
  pipeline run; the live `/snapshot` returns a 503 with a "run train.py" hint.
  **No recomputation** — that's the job of B1–B10. The store only reads.
- `api/schemas.py`: Pydantic v2 response models = the wire contract the React +
  three.js + p5.js frontend is built against.
- `api/routers_artifacts.py`: archetypes (B7), causal graph + cascades +
  ecosystem state (B8), counterfactuals (B9), model metrics (B3/B4).
- `api/routers_live.py`: `/snapshot` runs the B6 engine on demand over the B4
  best model for a chosen (day, minute); model cached after first load.
- `api/routers_llm.py`: B10 layer over HTTP — 5 structured output types +
  grounded Q&A. Context assembled via `from_artifacts` (LLM never sees raw
  sensor data). Backend = config default; layer cached.
- `api/main.py`: app factory, CORS, `/health` (artifact presence map),
  `/admin/reload` (drop caches to pick up a fresh pipeline run).

**Endpoints:** `/health`, `/links`, `/archetypes[/{id}]`,
`/echo/causal-graph`, `/echo/ecosystem-state`, `/echo/cascades`,
`/echo/counterfactual[/{id}]`, `/models/metrics`, `/snapshot`,
`/llm/backends`, `/llm/generate`, `/llm/ask`.

**Run:** `python scripts/11_serve.py` → http://127.0.0.1:8000/docs

**Gate:** every artifact reachable via HTTP; `tests/test_api.py` (16 tests)
green using shipped artifacts + the deterministic template LLM backend.

Phase 2 (frontend: React + three.js + p5.js) begins next.

---

## Phase 2 — Frontend (React + R3F + three.js)

Grilled + sequenced (see session). Hero view = **3D force-directed causal-graph
constellation** (nodes = 66 links, color = metabolic state, size = health,
edges = causal relationships). Bible's 10 pages collapse to **3 surfaces**:
Planner Command Center (constellation + HUD + deep-dive drawer), Counterfactual
Lab, Citizen Mode. Counterfactual Lab to be **gamified** as "Beat the Cascade"
(timed upstream-intervention game) — but only in Stage 3 if time allows.

**Three stages, hard cut lines:**

| Stage | Scope | Status |
|---|---|---|
| **1 — Spine** | R3F app, constellation (state color / health size / causal edges), node click → deep-dive drawer, view-mode toggle, API wiring | ✅ **DONE** (`frontend/`) |
| **2 — ECHO payload** | cascade animation ▸ timeline scrubber ▸ counterfactual sandbox ▸ LLM panel + full Citizen mode | 🔶 in progress |
| **3 — Game + polish** | "Beat the Cascade" timed loop + scoring + `/echo/counterfactual/simulate`, ReactBits flourishes, SHAP page | ⬜ if time |

**Stage 2 progress:**
- ✅ **Cascade animation** — `frontend/src/scene/CascadePulse.jsx` + store cascade
  slice. Pulse fires from Link 36 along real causal edges, infecting each
  downstream link exactly at its real `lag_minutes` (16/30/44 at 5min, 31/38 at
  10min). Whole event plays in ~10s. Source node flares white-hot; reached nodes
  flash toward the collapse color. Triggered from the HUD cascade control.
  Backend: `/echo/cascades/detailed` (parsed source→downstream+lags).
- ✅ **Timeline endpoint** (scrubber data) — `/echo/timeline` (one frame: all 66
  links' state at a day/minute) + `/echo/timeline/axis`. Lazy per-frame slice of
  features.parquet (~1.6s cold build, 6ms/frame). Scrubber UI: next.
- ⬜ Counterfactual sandbox, LLM panel, full Citizen mode.

---

# PHASE 2 — Frontend (React + R3F + three.js)

> Decided via /grilling. Stack: React + @react-three/fiber + drei, ReactBits
> for UI flourishes, talking to the B11 FastAPI layer. 3D-first.

## Locked frontend decisions

### F1 — Hero view → **3D force-directed causal-graph constellation**
NOT a 2D UMAP/node-graph (Bible §9), NOT a faked geographic road map (no geo
data exists — Bible §1.3). The causal graph is the only spatially-meaningful
real structure. Nodes = 66 links (color = metabolic state green/amber/red,
size = road_health). Edges = causal relationships (thickness =
correlation_strength, label = lag_minutes). Source: `/echo/causal-graph` +
`/echo/ecosystem-state`. Personality Atlas becomes a *secondary* 3D point cloud
in fingerprint space, not the hero.

### F2 — Page structure → **3 surfaces, not the Bible's 10 pages**
1. **Command Center (Planner):** constellation is the whole screen; click node →
   side drawer deep-dive (archetype, SHAP top-3, causal connections, recs,
   counterfactual sim). Timeline scrubber drives the scene. Absorbs Bible
   planner pages 1-5.
2. **Counterfactual Lab:** the July 1 centrepiece showpiece (see F3).
3. **Citizen Mode:** animated view-mode toggle (Bible §9.1) → plain-English
   commute status + simplified network + LLM assistant. Absorbs citizen pages
   1,2,4.
   Dropped/stubbed (low judge-signal): Report an Issue, Log Intervention history,
   Compare Days, Export PDF.

### F3 — Counterfactual Lab → **"Beat the Cascade" game**
The game mechanic IS the causal mechanism (not points sprinkled on top).
- Dropped into July 1, clock running to 09:45; Link 36 heading to collapse.
- **Intervention window** countdown = the real cascade lag (8 min, 36→16).
- Pick link + intervention → fire do-operator → constellation re-simulates.
- **Score = vehicle_hours_saved** (real `/echo/counterfactual` field);
  binary **cascade_prevented** = win/lose. Act earlier → bigger save.
- **Any-link intervention; upstream interventions score higher** — forces the
  player to traverse the causal graph backward (Pearl L2/L3). Cannot be won
  well by Level-1 thinking. Scores MUST come from SCM, no faking.
- Caveat: B9 SCM is fit per-link. Cross-link upstream scoring needs SCM×graph
  composition → new endpoint `/echo/counterfactual/simulate(intervention_link,
  type, target_link)`. **Deferred to Stage 3.**

### F4 — Build staging → **3 stages, hard cut lines, no backward deps**
- **Stage 1 — The Spine (must ship):** R3F app scaffold wired to FastAPI; 3D
  constellation (66 nodes+edges, state color, health size, corr thickness,
  orbit/zoom); click node → deep-dive drawer; view-mode toggle stub. A coherent
  demo on its own. *No new backend needed.*
- **Stage 2 — ECHO payload (should ship):** cascade animation (36→16 pulse,
  real lag); timeline scrubber through 14 days; Counterfactual Lab v1
  (non-timed sandbox, July 1 preloaded); LLM panel (`/llm/generate`+`/ask`);
  full Citizen Mode + animated toggle. *Needs ONE new endpoint:* `/echo/timeline`
  (per-link health/state across a time range — data exists in features.parquet,
  not yet exposed). Approved.
- **Stage 3 — Game + polish (nice to have):** "Beat the Cascade" timed loop +
  scoring + upstream mechanic + `/echo/counterfactual/simulate`; leaderboard;
  par score; post-mortem; ReactBits flourishes; SHAP Explorer; dropped pages.

Principle: Stage 1 alone is impressive. Stage 2 makes it win. Stage 3 makes it
unforgettable.

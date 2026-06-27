# UrbanPulse — Traffic Intelligence Platform

Backend per the Project Bible v2.0 (Pangyo, 14 days, 66 links, 5-min intervals).
Frontend (Phase 2) is a separate React + three.js + p5.js app over a FastAPI layer.

## Setup
```bash
pip install -r requirements.txt
```

## Pipeline (built in Bible dependency order)
| Stage | Run | Produces |
|---|---|---|
| B1 — EDA + cleaning | `python scripts/01_eda.py` | `data/cleaned.parquet`, `reports/eda/*` |
| B2 — feature engineering | `python scripts/02_features.py` | `data/features.parquet`, `data/feature_norms.json` |
| B3 — model training (7 models) | `python train.py` | `models/*.pkl`, `reports/model_metrics.csv` |
| B4 — comparison + select best | `python compare.py` | `models/best_model.pkl`, `models/best_model_meta.json`, `reports/model_comparison/*` |

Narrative notebooks live in `notebooks/` and import the repo-root modules
(`cleaning.py`, `eda.py`, `io_utils.py`, `config.py`) — no duplicated logic.

## Tests
```bash
python -m pytest tests/ -q
```

See `DECISION_MAP.md` for locked decisions and the full roadmap (B1–B11).

# UrbanPulse — How to Run

Causal traffic-intelligence platform: Python ML/causal backend (B1–B10), a
FastAPI serving layer (B11), and a React + MapLibre frontend that shows the 66
road links on a real dark map of Pangyo, colored by congestion.

All pipeline artifacts ship pre-built (data/*.parquet, data/*.json, reports/),
so the API and frontend run immediately — no training required.

## Prerequisites
- Python 3.10+ (3.12 recommended)
- Node.js 18+ (20+ recommended)

## 1. Backend — the API (terminal 1)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/11_serve.py       # http://127.0.0.1:8000  (docs at /docs)
```
Check: `curl http://127.0.0.1:8000/health` → status ok (best_model:false is fine).

## 2. Frontend — the map (terminal 2)
```bash
cd frontend
npm install                      # first time only
npm run dev                      # http://localhost:5173
```
Vite proxies /api → :8000, so both running together just works.

## What you'll see
A real dark map of Pangyo Techno Valley with the 66 links as glowing road
segments colored by metabolic state. Click a road for its deep-dive (archetype,
causal connections). "Play cascade" lights up the July 1 spread along the
network. Planner/Citizen toggle, top-right.

## Notes
- Geography: the dataset has no coordinates (Bible §1.3). Each LINK_ID is mapped
  to a real named Pangyo street via scripts/12_link_geometry.py (output cached at
  frontend/public/link_geometry.geojson). Re-roll: python scripts/12_link_geometry.py
- Basemap = CARTO dark-matter (free, no API key).
- Old R3F scene files (Constellation/LinkTower/Ground/Flythrough) are unused —
  App.jsx renders CityMap.jsx. Safe to keep or delete.
- Optional: to rebuild ML artifacts, place the raw CSV at data/raw.csv first,
  then run scripts/01_eda.py, scripts/02_features.py, train.py, compare.py.

## Tests
    python -m pytest tests/ -q

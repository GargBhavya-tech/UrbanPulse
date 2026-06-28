"""B11/frontend asset — map each of the 66 LINK_IDs to a real Pangyo road segment.

The dataset has NO geography (Bible §1.3: LINK_IDs are categorical, no coords).
To render the network on a real Google-Maps-style basemap, we assign each
LINK_ID to a genuine, named road segment in Pangyo Techno Valley (pulled from
OpenStreetMap via Overpass). The road *geometry* is real Pangyo; only the
LINK_ID -> segment assignment is a (deterministic) convention.

Output: frontend/public/link_geometry.geojson — a FeatureCollection of 66
LineString features, each with properties { link_id, name, highway }.

Run once (needs network):
    python scripts/12_link_geometry.py

Deterministic: a fixed seed picks the same 66 segments every run, so Link 36 is
always the same street across reloads, demos, and the cascade animation.
"""
from __future__ import annotations

import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Pangyo Techno Valley bounding box (south, west, north, east) ~2km
BBOX = "37.388,127.092,37.416,127.124"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
N_LINKS = config.EXPECTED_LINKS  # 66
SEED = 36  # deterministic selection; nod to the centrepiece link
OUT_PATH = config.ROOT / "frontend" / "public" / "link_geometry.geojson"


CACHE_PATH = config.ROOT / "scripts" / "_pangyo_roads_cache.json"


def fetch_roads() -> list[dict]:
    payload = None
    # Prefer a local cache (lets the script run offline / behind a proxy).
    if CACHE_PATH.exists():
        print(f"  using cached OSM response: {CACHE_PATH.name}")
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    else:
        query = (
            f"[out:json][timeout:30];"
            f'(way["highway"~"^(primary|secondary|tertiary|residential|trunk)$"]["name"]({BBOX}););'
            f"out geom;"
        )
        data = urllib.parse.urlencode({"data": query}).encode()
        req = urllib.request.Request(
            OVERPASS_URL,
            data=data,
            headers={"User-Agent": "UrbanPulse/1.0 (hackathon project)", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
        CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")  # cache for reruns
    ways = [
        e for e in payload.get("elements", [])
        if e.get("type") == "way" and len(e.get("geometry", [])) >= 3
    ]
    return ways


def segment_midpoint(way: dict) -> tuple[float, float]:
    g = way["geometry"]
    mid = g[len(g) // 2]
    return (mid["lat"], mid["lon"])


def select_segments(ways: list[dict], n: int) -> list[dict]:
    """Pick n well-distributed, distinct segments deterministically.

    Distinct by (name, rounded-midpoint) so we don't pick two slivers of the
    same street in the same place; then spatially thinned so links spread across
    the district rather than clumping on the densest street.
    """
    seen = set()
    distinct = []
    for w in ways:
        name = w["tags"].get("name", "")
        lat, lon = segment_midpoint(w)
        key = (name, round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        seen.add(key)
        distinct.append(w)

    rng = random.Random(SEED)
    rng.shuffle(distinct)
    if len(distinct) < n:
        raise SystemExit(
            f"Only {len(distinct)} distinct segments found; need {n}. "
            "Widen the BBOX or relax the highway filter."
        )
    return distinct[:n]


def to_geojson(segments: list[dict]) -> dict:
    features = []
    for link_id, way in enumerate(segments, start=1):
        coords = [[p["lon"], p["lat"]] for p in way["geometry"]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "link_id": link_id,
                "name": way["tags"].get("name", f"Link {link_id}"),
                "highway": way["tags"].get("highway", "road"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    print("Fetching Pangyo road geometry from OpenStreetMap (Overpass)...")
    ways = fetch_roads()
    print(f"  {len(ways)} candidate road ways")
    segments = select_segments(ways, N_LINKS)
    gj = to_geojson(segments)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(gj), encoding="utf-8")
    print(f"Wrote {len(gj['features'])} link segments -> {OUT_PATH}")
    # show the centrepiece
    l36 = next(f for f in gj["features"] if f["properties"]["link_id"] == 36)
    print(f"  Link 36 = {l36['properties']['name']} ({l36['properties']['highway']})")


if __name__ == "__main__":
    main()

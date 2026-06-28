"""B11 — launch the UrbanPulse FastAPI serving layer.

Usage::

    python scripts/11_serve.py                 # 127.0.0.1:8000
    python scripts/11_serve.py --host 0.0.0.0 --port 8080 --reload

Open http://127.0.0.1:8000/docs for the interactive API.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the UrbanPulse API (B11).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on edits")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

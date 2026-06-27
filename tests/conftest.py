"""Canonical pytest configuration for the UrbanPulse test suite.

Inserts the repository root into ``sys.path`` **once** so that every test
module can import the top-level packages (``cleaning``, ``config``, etc.)
without each file needing its own ``sys.path.insert`` hack.

NOTE: The existing ``sys.path.insert`` lines in each test file are now
redundant (but harmless). They can be removed in a follow-up cleanup.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root = this file's parent's parent.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

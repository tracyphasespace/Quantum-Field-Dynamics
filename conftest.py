"""Pytest configuration helpers."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.join(ROOT_DIR, "redshift-analysis")

for path in (ROOT_DIR, PACKAGE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


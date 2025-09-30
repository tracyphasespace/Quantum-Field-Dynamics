"""Test environment configuration for the kata.

Python executed in the evaluation container starts with an empty search path
for user code, which prevents importing local helper packages such as the
lightweight ``numpy`` compatibility layer that ships with the repository.

The standard library looks for a module named :mod:`sitecustomize` during
startup, so we take advantage of that hook to ensure the repository root is on
``sys.path``.  This mirrors the behaviour of ``python -m`` and normal
interpreter sessions where ``''`` (the current working directory) is present by
default.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(__file__)

if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


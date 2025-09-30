"""
QFD Redshift Analysis Package
============================

Quantum Field Dynamics approach to cosmological redshift analysis.
Provides alternative to dark energy using experimentally validated physics.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

__version__ = "1.0.0"
__author__ = "PhaseSpace Research"
__email__ = "research@phasespace.tech"

from .cosmology import QFDCosmology
from .physics import QFDPhysics

try:  # pragma: no cover - optional dependencies
    from .redshift_analyzer import RedshiftAnalyzer
except ModuleNotFoundError:  # pragma: no cover - gracefully degrade
    RedshiftAnalyzer = None

try:  # pragma: no cover - optional dependencies
    from .visualization import RedshiftPlotter
except ModuleNotFoundError:  # pragma: no cover - gracefully degrade
    RedshiftPlotter = None

__all__ = [name for name in ("RedshiftAnalyzer", "QFDCosmology", "QFDPhysics", "RedshiftPlotter") if locals().get(name) is not None]

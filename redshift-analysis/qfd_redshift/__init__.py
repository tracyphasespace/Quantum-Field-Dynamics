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

from .redshift_analyzer import RedshiftAnalyzer
from .cosmology import QFDCosmology
from .physics import QFDPhysics
from .visualization import RedshiftPlotter

__all__ = ["RedshiftAnalyzer", "QFDCosmology", "QFDPhysics", "RedshiftPlotter"]

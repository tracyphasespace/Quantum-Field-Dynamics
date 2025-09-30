"""
QFD Supernova Analysis Package
=============================

Quantum Field Dynamics approach to supernova dimming analysis.
Provides wavelength-dependent alternative to dark energy using 
experimentally validated physics.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

__version__ = "1.0.0"
__author__ = "PhaseSpace Research"
__email__ = "research@phasespace.tech"

from .supernova_analyzer import SupernovaAnalyzer
from .plasma_physics import SupernovaPlasma
from .qvd_scattering import QVDScattering
from .visualization import SupernovaPlotter

__all__ = [
    "SupernovaAnalyzer",
    "SupernovaPlasma",
    "QVDScattering", 
    "SupernovaPlotter"
]
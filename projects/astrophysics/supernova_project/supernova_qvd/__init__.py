"""
Supernova QVD Scattering Model Package
======================================

A package for analyzing supernova light curves using a physically-motivated
phenomenological model based on Quantum Vacuum Dynamics (QVD).
"""

__version__ = "1.0.0"
__author__ = "PhaseSpace"

# Import key components to the top-level namespace
from .parameters import E144ExperimentalData, SupernovaParameters
from .model import E144ScaledQVDModel
from .analysis import create_supernova_analysis_plots, demonstrate_e144_supernova_model
from . import numerical_safety
from . import physical_bounds

__all__ = [
    'E144ExperimentalData',
    'SupernovaParameters',
    'E144ScaledQVDModel',
    'create_supernova_analysis_plots',
    'demonstrate_e144_supernova_model',
    'numerical_safety',
    'physical_bounds'
]

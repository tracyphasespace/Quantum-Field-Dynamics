"""
RedShift QVD Cosmological Model Package
=======================================

A package for analyzing cosmological redshift using a physically-motivated
phenomenological model based on Quantum Vacuum Dynamics (QVD).
"""

__version__ = "1.1.0" # Version updated to reflect significant refactoring
__author__ = "PhaseSpace"

# Import main classes for easy access
from .analyzer import EnhancedRedshiftAnalyzer
from .physics import EnhancedQVDPhysics
from .cosmology import EnhancedQVDCosmology
from . import numerical_safety
from . import physical_bounds
from . import error_handling

__all__ = [
    'EnhancedRedshiftAnalyzer',
    'EnhancedQVDPhysics',
    'EnhancedQVDCosmology',
    'numerical_safety',
    'physical_bounds',
    'error_handling'
]

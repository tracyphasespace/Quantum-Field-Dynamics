"""
QFD Lepton Mass Spectrum

A computational model for charged lepton masses based on
Quantum Fluid Dynamics (QFD) and Hill's spherical vortex.
"""

__version__ = "1.0.0"
__author__ = "QFD Research Group"

from . import functionals
from . import solvers

__all__ = ['functionals', 'solvers']

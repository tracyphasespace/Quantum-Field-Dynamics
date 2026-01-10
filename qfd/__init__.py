"""
QFD (Quantum Field Dynamics) Package

Observable adapters and physics implementations for the Grand Solver.

Core constants derived from fine structure constant α:
    from qfd.shared_constants import ALPHA, BETA, C1_SURFACE, C2_VOLUME
"""

__version__ = "0.3.0"

# Expose key derived constants at package level
from qfd.shared_constants import (
    ALPHA, ALPHA_INV,
    BETA, BETA_STANDARDIZED,
    C1_SURFACE, C2_VOLUME,
    V4_QED,
    fundamental_soliton_equation,
)

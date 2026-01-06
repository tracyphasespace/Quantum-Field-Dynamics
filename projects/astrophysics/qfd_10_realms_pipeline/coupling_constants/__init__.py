"""
Coupling Constants Mapping System for QFD Framework

This module provides comprehensive tracking, validation, and analysis
of coupling constants across all physical realms in the QFD framework.
"""

__version__ = "0.1.0"

from .registry.parameter_registry import ParameterRegistry, ParameterState, Constraint

__all__ = [
    "ParameterRegistry",
    "ParameterState", 
    "Constraint"
]
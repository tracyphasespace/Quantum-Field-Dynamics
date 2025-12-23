"""
Lean 4 Formal Constraint Validation

This module provides Python interfaces to Lean 4 formally-proven
parameter constraints for QFD cosmological parameters.

The constraints are derived from mathematical proofs in Lean 4 that guarantee:
    - Vacuum stability (energy density ≥ 0)
    - Physical scattering (opacity, not gain)
    - Bounded interactions (no divergences)

Source Proofs:
    Lean4_Schema/Proofs/AdjointStability_Complete.lean
    Lean4_Schema/Proofs/PhysicalScattering.lean
    Lean4_Schema/Proofs/BoundedInteractions.lean

Usage:
    >>> from qfd_sn.lean_validation import validate_parameters
    >>> passed, results = validate_parameters(
    ...     k_J=121.34, eta_prime=-0.04, xi=-6.45, sigma_ln_A=1.64
    ... )
    >>> print(f"All constraints passed: {passed}")
"""

from .constraints import (
    LeanConstraints,
    validate_parameters,
    validate_k_J,
    validate_eta_prime,
    validate_xi,
    validate_sigma_ln_A,
)

from .schema_interface import (
    QFDParameters,
    validate_schema_compliance,
)

from .report_generator import (
    generate_validation_report,
    create_constraint_visualization,
)

__all__ = [
    "LeanConstraints",
    "validate_parameters",
    "validate_k_J",
    "validate_eta_prime",
    "validate_xi",
    "validate_sigma_ln_A",
    "QFDParameters",
    "validate_schema_compliance",
    "generate_validation_report",
    "create_constraint_visualization",
]

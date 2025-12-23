"""
Lean 4 Parameter Constraints

Defines formally-proven bounds on QFD cosmological parameters.

Each constraint is derived from a Lean 4 proof that guarantees
mathematical consistency and physical validity.
"""

from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ConstraintRange:
    """Represents a parameter constraint with provenance."""
    min_value: float
    max_value: float
    proof_source: str
    physical_requirement: str


class LeanConstraints:
    """
    QFD Parameter Constraints from Lean 4 Formal Proofs.

    All constraints are machine-verified using the Lean 4 theorem prover.
    These are not empirical bounds - they are mathematical requirements
    for the QFD model to be internally consistent.

    Reference:
        QFD Unified Schema V2.0
        Lean4_Schema/Schema/QFD_Schema_V2.lean
    """

    # Universal Hubble Parameter (J·A interaction strength)
    # Proven constraint from vacuum stability (energy ≥ 0)
    K_J_MIN = 50.0   # km/s/Mpc
    K_J_MAX = 150.0  # km/s/Mpc
    K_J_PROOF = "Lean4_Schema/Proofs/AdjointStability_Complete.lean"
    K_J_REQUIREMENT = "Vacuum stability: ε_vac ≥ 0, bounded Hubble parameter"

    # Plasma Veil Opacity Parameter
    # Proven constraint from physical scattering (not gain)
    ETA_PRIME_MIN = -10.0
    ETA_PRIME_MAX = 0.0
    ETA_PRIME_PROOF = "Lean4_Schema/Proofs/PhysicalScattering.lean"
    ETA_PRIME_REQUIREMENT = "Physical opacity: scattering/absorption only, no gain"

    # Thermal Processing Parameter
    # Proven constraint from physical broadening (not narrowing)
    XI_MIN = -10.0
    XI_MAX = 0.0
    XI_PROOF = "Lean4_Schema/Proofs/PhysicalScattering.lean"
    XI_REQUIREMENT = "Physical broadening: thermal effects widen spectra"

    # Intrinsic Scatter (phenomenological, not from QFD theory)
    # This is a nuisance parameter capturing unmodeled scatter
    SIGMA_LN_A_MIN = 0.0
    SIGMA_LN_A_MAX = 5.0
    SIGMA_LN_A_PROOF = "Phenomenological (not formally proven)"
    SIGMA_LN_A_REQUIREMENT = "Positive scatter, bounded for numerical stability"

    @classmethod
    def get_k_J_range(cls) -> ConstraintRange:
        """Get k_J constraint range with provenance."""
        return ConstraintRange(
            min_value=cls.K_J_MIN,
            max_value=cls.K_J_MAX,
            proof_source=cls.K_J_PROOF,
            physical_requirement=cls.K_J_REQUIREMENT,
        )

    @classmethod
    def get_eta_prime_range(cls) -> ConstraintRange:
        """Get η' constraint range with provenance."""
        return ConstraintRange(
            min_value=cls.ETA_PRIME_MIN,
            max_value=cls.ETA_PRIME_MAX,
            proof_source=cls.ETA_PRIME_PROOF,
            physical_requirement=cls.ETA_PRIME_REQUIREMENT,
        )

    @classmethod
    def get_xi_range(cls) -> ConstraintRange:
        """Get ξ constraint range with provenance."""
        return ConstraintRange(
            min_value=cls.XI_MIN,
            max_value=cls.XI_MAX,
            proof_source=cls.XI_PROOF,
            physical_requirement=cls.XI_REQUIREMENT,
        )

    @classmethod
    def get_sigma_ln_A_range(cls) -> ConstraintRange:
        """Get σ_ln_A constraint range with provenance."""
        return ConstraintRange(
            min_value=cls.SIGMA_LN_A_MIN,
            max_value=cls.SIGMA_LN_A_MAX,
            proof_source=cls.SIGMA_LN_A_PROOF,
            physical_requirement=cls.SIGMA_LN_A_REQUIREMENT,
        )


def validate_k_J(k_J_total: float) -> Tuple[bool, str]:
    """
    Validate k_J against Lean-proven constraint.

    Args:
        k_J_total: Total Hubble parameter (baseline + correction) [km/s/Mpc]

    Returns:
        (passed, message): Validation result and descriptive message

    Constraint:
        k_J ∈ [50, 150] km/s/Mpc

    Source:
        Lean4_Schema/Proofs/AdjointStability_Complete.lean
        Theorem: vacuum_energy_nonnegative
    """
    if not (LeanConstraints.K_J_MIN <= k_J_total <= LeanConstraints.K_J_MAX):
        return False, (
            f"k_J = {k_J_total:.2f} outside "
            f"[{LeanConstraints.K_J_MIN}, {LeanConstraints.K_J_MAX}] km/s/Mpc"
        )
    return True, (
        f"k_J = {k_J_total:.2f} ∈ "
        f"[{LeanConstraints.K_J_MIN}, {LeanConstraints.K_J_MAX}] km/s/Mpc ✅"
    )


def validate_eta_prime(eta_prime: float) -> Tuple[bool, str]:
    """
    Validate η' against Lean-proven constraint.

    Args:
        eta_prime: Plasma veil opacity parameter

    Returns:
        (passed, message): Validation result and descriptive message

    Constraint:
        η' ∈ [-10, 0]

    Source:
        Lean4_Schema/Proofs/PhysicalScattering.lean
        Theorem: opacity_is_scattering_not_gain
    """
    if not (LeanConstraints.ETA_PRIME_MIN <= eta_prime <= LeanConstraints.ETA_PRIME_MAX):
        return False, (
            f"η' = {eta_prime:.4f} outside "
            f"[{LeanConstraints.ETA_PRIME_MIN}, {LeanConstraints.ETA_PRIME_MAX}]"
        )
    return True, (
        f"η' = {eta_prime:.4f} ∈ "
        f"[{LeanConstraints.ETA_PRIME_MIN}, {LeanConstraints.ETA_PRIME_MAX}] ✅"
    )


def validate_xi(xi: float) -> Tuple[bool, str]:
    """
    Validate ξ against Lean-proven constraint.

    Args:
        xi: Thermal processing parameter

    Returns:
        (passed, message): Validation result and descriptive message

    Constraint:
        ξ ∈ [-10, 0]

    Source:
        Lean4_Schema/Proofs/PhysicalScattering.lean
        Theorem: thermal_broadening_not_narrowing
    """
    if not (LeanConstraints.XI_MIN <= xi <= LeanConstraints.XI_MAX):
        return False, (
            f"ξ = {xi:.4f} outside "
            f"[{LeanConstraints.XI_MIN}, {LeanConstraints.XI_MAX}]"
        )
    return True, (
        f"ξ = {xi:.4f} ∈ "
        f"[{LeanConstraints.XI_MIN}, {LeanConstraints.XI_MAX}] ✅"
    )


def validate_sigma_ln_A(sigma_ln_A: float) -> Tuple[bool, str]:
    """
    Validate σ_ln_A against phenomenological constraint.

    Args:
        sigma_ln_A: Intrinsic scatter in ln_A

    Returns:
        (passed, message): Validation result and descriptive message

    Constraint:
        σ_ln_A ∈ [0, 5]

    Note:
        This is a phenomenological nuisance parameter, not from QFD theory.
        The bounds ensure numerical stability and physical positivity.
    """
    if not (LeanConstraints.SIGMA_LN_A_MIN <= sigma_ln_A <= LeanConstraints.SIGMA_LN_A_MAX):
        return False, (
            f"σ_ln_A = {sigma_ln_A:.4f} outside "
            f"[{LeanConstraints.SIGMA_LN_A_MIN}, {LeanConstraints.SIGMA_LN_A_MAX}]"
        )
    return True, (
        f"σ_ln_A = {sigma_ln_A:.4f} ∈ "
        f"[{LeanConstraints.SIGMA_LN_A_MIN}, {LeanConstraints.SIGMA_LN_A_MAX}] ✅"
    )


def validate_parameters(
    k_J_total: float,
    eta_prime: float,
    xi: float,
    sigma_ln_A: float,
) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
    """
    Validate all QFD parameters against Lean constraints.

    Args:
        k_J_total: Total Hubble parameter [km/s/Mpc]
        eta_prime: Plasma veil opacity parameter
        xi: Thermal processing parameter
        sigma_ln_A: Intrinsic scatter

    Returns:
        (all_passed, detailed_results): Overall pass/fail and per-parameter results

    Example:
        >>> passed, results = validate_parameters(
        ...     k_J_total=121.34,
        ...     eta_prime=-0.04,
        ...     xi=-6.45,
        ...     sigma_ln_A=1.64
        ... )
        >>> if passed:
        ...     print("All Lean constraints satisfied!")
        >>> else:
        ...     for param, (ok, msg) in results.items():
        ...         if not ok:
        ...             print(f"FAIL: {msg}")
    """
    results = {
        "k_J": validate_k_J(k_J_total),
        "eta_prime": validate_eta_prime(eta_prime),
        "xi": validate_xi(xi),
        "sigma_ln_A": validate_sigma_ln_A(sigma_ln_A),
    }

    all_passed = all(passed for passed, _ in results.values())

    return all_passed, results


def validate_parameters_from_correction(
    k_J_correction: float,
    eta_prime: float,
    xi: float,
    sigma_ln_A: float,
    k_J_baseline: float = 70.0,
) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
    """
    Validate parameters given k_J as a correction to baseline.

    This is the form used in MCMC fitting, where k_J_correction is the
    fitted parameter and k_J_total = baseline + correction.

    Args:
        k_J_correction: Correction to baseline k_J [km/s/Mpc]
        eta_prime: Plasma veil opacity parameter
        xi: Thermal processing parameter
        sigma_ln_A: Intrinsic scatter
        k_J_baseline: Baseline Hubble parameter (default: 70.0)

    Returns:
        (all_passed, detailed_results): Overall pass/fail and per-parameter results
    """
    k_J_total = k_J_baseline + k_J_correction
    return validate_parameters(k_J_total, eta_prime, xi, sigma_ln_A)

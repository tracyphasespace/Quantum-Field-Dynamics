"""
QFD Unified Schema V2.0 Interface

Provides Python interface to the QFD Unified Schema V2.0 formal definitions.
Ensures consistency between Python code and Lean 4 specifications.

Reference:
    Lean4_Schema/Schema/QFD_Schema_V2.lean
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class QFDParameters:
    """
    QFD cosmological parameters matching Lean 4 schema definition.

    This dataclass mirrors the parameter structure defined in:
        Lean4_Schema/Schema/QFD_Schema_V2.lean

    Attributes:
        k_J_correction: Correction to baseline k_J (baseline = 70 km/s/Mpc)
        eta_prime: Plasma veil opacity parameter
        xi: Thermal processing parameter
        sigma_ln_A: Intrinsic scatter in ln(amplitude)
    """

    k_J_correction: float
    eta_prime: float
    xi: float
    sigma_ln_A: float

    def k_J_total(self, baseline: float = 70.0) -> float:
        """Compute total k_J from baseline + correction."""
        return baseline + self.k_J_correction

    def to_dict(self, baseline: float = 70.0) -> Dict[str, Any]:
        """
        Convert to dictionary with both correction and total k_J.

        Args:
            baseline: Baseline k_J value (default: 70.0 km/s/Mpc)

        Returns:
            Dictionary with all parameter values
        """
        return {
            "k_J_correction": self.k_J_correction,
            "k_J_total": self.k_J_total(baseline),
            "eta_prime": self.eta_prime,
            "xi": self.xi,
            "sigma_ln_A": self.sigma_ln_A,
        }

    def to_json(self, baseline: float = 70.0) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(baseline), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QFDParameters":
        """
        Create from dictionary.

        Accepts either 'k_J_correction' or computes it from 'k_J_total'.
        """
        if "k_J_correction" in data:
            k_J_correction = data["k_J_correction"]
        elif "k_J_total" in data:
            baseline = data.get("k_J_baseline", 70.0)
            k_J_correction = data["k_J_total"] - baseline
        else:
            raise ValueError("Must provide either k_J_correction or k_J_total")

        return cls(
            k_J_correction=k_J_correction,
            eta_prime=data["eta_prime"],
            xi=data["xi"],
            sigma_ln_A=data["sigma_ln_A"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "QFDParameters":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"QFDParameters(\n"
            f"  k_J_total = {self.k_J_total():.4f} km/s/Mpc\n"
            f"  η' = {self.eta_prime:.4f}\n"
            f"  ξ  = {self.xi:.4f}\n"
            f"  σ_ln_A = {self.sigma_ln_A:.4f}\n"
            f")"
        )


def validate_schema_compliance(params: QFDParameters) -> bool:
    """
    Validate that parameters comply with QFD Unified Schema V2.0.

    This checks:
        1. All required fields present
        2. All fields have correct types
        3. No NaN or infinite values

    Args:
        params: QFDParameters instance

    Returns:
        True if schema-compliant, False otherwise

    Note:
        This checks *schema compliance* (data structure).
        For *Lean constraint validation* (physical bounds),
        use validate_parameters() from constraints module.
    """
    import math

    # Check all fields are present (dataclass guarantees this)
    required_fields = ["k_J_correction", "eta_prime", "xi", "sigma_ln_A"]
    for field in required_fields:
        if not hasattr(params, field):
            return False

    # Check all fields are float
    for field in required_fields:
        value = getattr(params, field)
        if not isinstance(value, (int, float)):
            return False

    # Check no NaN or infinite values
    for field in required_fields:
        value = float(getattr(params, field))
        if math.isnan(value) or math.isinf(value):
            return False

    return True

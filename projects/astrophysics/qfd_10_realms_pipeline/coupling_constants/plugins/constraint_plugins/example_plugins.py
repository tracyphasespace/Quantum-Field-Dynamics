"""
Example constraint plugins demonstrating the plugin system.

These plugins show how to implement custom physics constraints
that can be registered with the PluginManager.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from ..plugin_manager import ConstraintPlugin, PluginInfo, PluginPriority
from ...registry.parameter_registry import ParameterRegistry
from ...validation.base_validator import ValidationResult, ValidationStatus, ValidationViolation


class PhotonMassConstraintPlugin(ConstraintPlugin):
    """
    Plugin that enforces photon mass constraints from experimental limits.
    
    This plugin demonstrates how to implement observational constraints
    that limit parameter values based on experimental measurements.
    """
    
    @property
    def plugin_info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="photon_mass_constraint",
            version="1.0.0",
            author="QFD Framework",
            description="Enforces experimental limits on photon mass",
            priority=PluginPriority.HIGH,
            dependencies=[],
            active=True
        )
    
    def get_parameter_dependencies(self) -> List[str]:
        """Return parameters this constraint depends on."""
        return ["m_gamma", "k_J", "xi"]  # Photon mass and related parameters
    
    def validate_constraint(self, registry: ParameterRegistry) -> ValidationResult:
        """
        Validate photon mass constraint.
        
        Experimental limit: m_gamma < 1e-18 eV (from various experiments)
        """
        # Get relevant parameters
        m_gamma = registry.get_parameter("m_gamma")
        k_J = registry.get_parameter("k_J") 
        xi = registry.get_parameter("xi")
        
        # Check if required parameters exist and have values
        if not m_gamma or m_gamma.value is None:
            result = ValidationResult(
                validator_name="photon_mass_constraint",
                status=ValidationStatus.INVALID
            )
            result.add_violation(ValidationViolation(
                parameter_name="m_gamma",
                constraint_realm="photon_mass_constraint",
                violation_type="missing_parameter",
                message="Photon mass parameter m_gamma not found or has no value"
            ))
            return result
        
        # Experimental limit in eV (converted to natural units if needed)
        experimental_limit = 1e-18  # eV
        
        # Check direct photon mass constraint
        if m_gamma.value > experimental_limit:
            result = ValidationResult(
                validator_name="photon_mass_constraint",
                status=ValidationStatus.INVALID
            )
            result.add_violation(ValidationViolation(
                parameter_name="m_gamma",
                constraint_realm="photon_mass_constraint",
                violation_type="exceeds_limit",
                expected_value=experimental_limit,
                actual_value=m_gamma.value,
                message=f"Photon mass {m_gamma.value:.2e} exceeds experimental limit {experimental_limit:.2e}"
            ))
            result.metadata = {
                "current_value": m_gamma.value,
                "limit": experimental_limit,
                "violation_factor": m_gamma.value / experimental_limit
            }
            return result
        
        # Additional constraint: if k_J and xi affect effective photon mass
        if k_J and k_J.value is not None and xi and xi.value is not None:
            # Example: effective photon mass could be modified by QFD parameters
            effective_mass = m_gamma.value * (1 + k_J.value * xi.value)
            
            if effective_mass > experimental_limit:
                result = ValidationResult(
                    validator_name="photon_mass_constraint",
                    status=ValidationStatus.INVALID
                )
                result.add_violation(ValidationViolation(
                    parameter_name="m_gamma",
                    constraint_realm="photon_mass_constraint",
                    violation_type="effective_mass_exceeds_limit",
                    expected_value=experimental_limit,
                    actual_value=effective_mass,
                    message=f"Effective photon mass {effective_mass:.2e} exceeds experimental limit"
                ))
                result.metadata = {
                    "base_mass": m_gamma.value,
                    "effective_mass": effective_mass,
                    "k_J_contribution": k_J.value,
                    "xi_contribution": xi.value,
                    "limit": experimental_limit
                }
                return result
        
        result = ValidationResult(
            validator_name="photon_mass_constraint",
            status=ValidationStatus.VALID
        )
        result.add_info("Photon mass within experimental limits")
        result.metadata = {
            "current_value": m_gamma.value,
            "limit": experimental_limit,
            "margin": experimental_limit - m_gamma.value
        }
        return result
    
    def get_constraint_description(self) -> str:
        """Return human-readable constraint description."""
        return "Enforces experimental upper limit on photon mass (m_γ < 1×10⁻¹⁸ eV)"


class VacuumStabilityPlugin(ConstraintPlugin):
    """
    Plugin that enforces vacuum stability constraints.
    
    This plugin demonstrates how to implement theoretical constraints
    that ensure the vacuum state remains stable under QFD modifications.
    """
    
    def __init__(self):
        """Initialize vacuum stability plugin."""
        self.stability_threshold = 1e-6  # Configurable threshold
    
    @property
    def plugin_info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="vacuum_stability",
            version="1.0.0", 
            author="QFD Framework",
            description="Ensures vacuum stability under QFD parameter modifications",
            priority=PluginPriority.CRITICAL,
            dependencies=[],
            active=True
        )
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with configuration."""
        if config:
            self.stability_threshold = config.get("stability_threshold", self.stability_threshold)
    
    def get_parameter_dependencies(self) -> List[str]:
        """Return parameters this constraint depends on."""
        return ["n_vac", "k_J", "psi_s0", "xi"]
    
    def validate_constraint(self, registry: ParameterRegistry) -> ValidationResult:
        """
        Validate vacuum stability constraint.
        
        Ensures that QFD modifications don't destabilize the vacuum.
        """
        # Get vacuum-related parameters
        n_vac = registry.get_parameter("n_vac")
        k_J = registry.get_parameter("k_J")
        psi_s0 = registry.get_parameter("psi_s0")
        xi = registry.get_parameter("xi")
        
        # Check vacuum refractive index
        if not n_vac or n_vac.value is None:
            result = ValidationResult(
                validator_name="vacuum_stability",
                status=ValidationStatus.INVALID
            )
            result.add_violation(ValidationViolation(
                parameter_name="n_vac",
                constraint_realm="vacuum_stability",
                violation_type="missing_parameter",
                message="Vacuum refractive index n_vac not found"
            ))
            return result
        
        # Vacuum must have n = 1 exactly for stability
        if abs(n_vac.value - 1.0) > self.stability_threshold:
            result = ValidationResult(
                validator_name="vacuum_stability",
                status=ValidationStatus.INVALID
            )
            result.add_violation(ValidationViolation(
                parameter_name="n_vac",
                constraint_realm="vacuum_stability",
                violation_type="deviation_from_unity",
                expected_value=1.0,
                actual_value=n_vac.value,
                tolerance=self.stability_threshold,
                message=f"Vacuum refractive index {n_vac.value} deviates from unity"
            ))
            result.metadata = {
                "current_value": n_vac.value,
                "target_value": 1.0,
                "deviation": abs(n_vac.value - 1.0),
                "threshold": self.stability_threshold
            }
            return result
        
        # Check for destabilizing parameter combinations
        instability_factors = []
        
        if k_J and k_J.value is not None:
            if abs(k_J.value) > 1e-10:  # k_J should be very small
                instability_factors.append(f"k_J = {k_J.value:.2e} may be too large")
        
        if psi_s0 and psi_s0.value is not None:
            if abs(psi_s0.value) > 10:  # psi_s0 should be reasonable
                instability_factors.append(f"psi_s0 = {psi_s0.value:.2e} may destabilize vacuum")
        
        if xi and xi.value is not None:
            if xi.value < 0 or xi.value > 100:  # xi should be positive and reasonable
                instability_factors.append(f"xi = {xi.value:.2e} outside stable range")
        
        if instability_factors:
            result = ValidationResult(
                validator_name="vacuum_stability",
                status=ValidationStatus.INVALID
            )
            result.add_violation(ValidationViolation(
                parameter_name="vacuum_parameters",
                constraint_realm="vacuum_stability",
                violation_type="instability_risk",
                message="Parameter values may destabilize vacuum"
            ))
            result.metadata = {
                "instability_factors": instability_factors,
                "stability_threshold": self.stability_threshold
            }
            return result
        
        result = ValidationResult(
            validator_name="vacuum_stability",
            status=ValidationStatus.VALID
        )
        result.add_info("Vacuum stability maintained")
        result.metadata = {
            "n_vac": n_vac.value,
            "stability_margin": self.stability_threshold - abs(n_vac.value - 1.0)
        }
        return result
    
    def get_constraint_description(self) -> str:
        """Return human-readable constraint description."""
        return "Ensures vacuum stability by maintaining n_vac = 1 and checking parameter combinations"


class CosmologicalConstantPlugin(ConstraintPlugin):
    """
    Plugin that enforces cosmological constant constraints.
    
    This plugin demonstrates how to implement cosmological constraints
    that relate QFD parameters to observed cosmic acceleration.
    """
    
    def __init__(self):
        """Initialize cosmological constant plugin."""
        self.lambda_observed = 1.1e-52  # m^-2, observed cosmological constant
        self.tolerance = 0.1  # 10% tolerance
    
    @property
    def plugin_info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="cosmological_constant",
            version="1.0.0",
            author="QFD Framework", 
            description="Enforces consistency with observed cosmological constant",
            priority=PluginPriority.NORMAL,
            dependencies=[],
            active=True
        )
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with configuration."""
        if config:
            self.lambda_observed = config.get("lambda_observed", self.lambda_observed)
            self.tolerance = config.get("tolerance", self.tolerance)
    
    def get_parameter_dependencies(self) -> List[str]:
        """Return parameters this constraint depends on."""
        return ["Lambda", "psi_s0", "xi", "H0"]  # Cosmological parameters
    
    def validate_constraint(self, registry: ParameterRegistry) -> ValidationResult:
        """
        Validate cosmological constant constraint.
        
        Checks that QFD parameters are consistent with observed cosmic acceleration.
        """
        # Get cosmological parameters
        Lambda = registry.get_parameter("Lambda")
        psi_s0 = registry.get_parameter("psi_s0")
        xi = registry.get_parameter("xi")
        H0 = registry.get_parameter("H0")
        
        # Check if Lambda is directly specified
        if Lambda and Lambda.value is not None:
            relative_error = abs(Lambda.value - self.lambda_observed) / self.lambda_observed
            
            if relative_error > self.tolerance:
                result = ValidationResult(
                    validator_name="cosmological_constant",
                    status=ValidationStatus.INVALID
                )
                result.add_violation(ValidationViolation(
                    parameter_name="Lambda",
                    constraint_realm="cosmological_constant",
                    violation_type="inconsistent_with_observations",
                    expected_value=self.lambda_observed,
                    actual_value=Lambda.value,
                    tolerance=self.tolerance,
                    message=f"Cosmological constant {Lambda.value:.2e} inconsistent with observations"
                ))
                result.metadata = {
                    "current_value": Lambda.value,
                    "observed_value": self.lambda_observed,
                    "relative_error": relative_error,
                    "tolerance": self.tolerance
                }
                return result
        
        # Check derived cosmological constant from QFD parameters
        if psi_s0 and psi_s0.value is not None and xi and xi.value is not None:
            # Example: effective Lambda from QFD modifications
            # This is a simplified model - real calculation would be more complex
            effective_lambda = self.lambda_observed * (1 + psi_s0.value / xi.value * 1e-3)
            
            relative_error = abs(effective_lambda - self.lambda_observed) / self.lambda_observed
            
            if relative_error > self.tolerance:
                result = ValidationResult(
                    validator_name="cosmological_constant",
                    status=ValidationStatus.INVALID
                )
                result.add_violation(ValidationViolation(
                    parameter_name="effective_Lambda",
                    constraint_realm="cosmological_constant",
                    violation_type="qfd_modified_inconsistent",
                    expected_value=self.lambda_observed,
                    actual_value=effective_lambda,
                    tolerance=self.tolerance,
                    message=f"QFD-modified cosmological constant {effective_lambda:.2e} inconsistent with observations"
                ))
                result.metadata = {
                    "effective_lambda": effective_lambda,
                    "observed_lambda": self.lambda_observed,
                    "psi_s0_contribution": psi_s0.value,
                    "xi_contribution": xi.value,
                    "relative_error": relative_error,
                    "tolerance": self.tolerance
                }
                return result
        
        # Check Hubble constant consistency (if available)
        if H0 and H0.value is not None:
            # Hubble constant should be around 70 km/s/Mpc
            H0_expected = 70.0  # km/s/Mpc
            H0_tolerance = 0.15  # 15% tolerance due to measurement uncertainties
            
            relative_error = abs(H0.value - H0_expected) / H0_expected
            
            if relative_error > H0_tolerance:
                result = ValidationResult(
                    validator_name="cosmological_constant",
                    status=ValidationStatus.INVALID
                )
                result.add_violation(ValidationViolation(
                    parameter_name="H0",
                    constraint_realm="cosmological_constant",
                    violation_type="hubble_inconsistent",
                    expected_value=H0_expected,
                    actual_value=H0.value,
                    tolerance=H0_tolerance,
                    message=f"Hubble constant {H0.value:.2f} inconsistent with observations"
                ))
                result.metadata = {
                    "current_H0": H0.value,
                    "expected_H0": H0_expected,
                    "relative_error": relative_error,
                    "tolerance": H0_tolerance
                }
                return result
        
        result = ValidationResult(
            validator_name="cosmological_constant",
            status=ValidationStatus.VALID
        )
        result.add_info("Cosmological parameters consistent with observations")
        result.metadata = {
            "lambda_observed": self.lambda_observed,
            "tolerance": self.tolerance
        }
        return result
    
    def get_constraint_description(self) -> str:
        """Return human-readable constraint description."""
        return f"Enforces consistency with observed cosmological constant (Λ = {self.lambda_observed:.2e} m⁻²)"
"""
CMB (Cosmic Microwave Background) constraint validator.

This module validates that coupling constants are consistent with
CMB observations and constraints, including thermalization temperature,
polarization constraints, and spectral distortion limits.
"""

import time
from typing import Optional, Dict, Any
from .base_validator import BaseValidator, ValidationResult, ValidationViolation, ValidationStatus
from .derived_validator import DerivedConstraintValidator, EvaluatorFunction
from ..registry.parameter_registry import ParameterRegistry, ConstraintType


def create_cmb_temperature_evaluator(target_temp: float = 2.725, tolerance: float = 1e-6) -> EvaluatorFunction:
    """
    Create an evaluator for CMB temperature consistency.
    
    Args:
        target_temp: Target CMB temperature in Kelvin (FIRAS/Planck: 2.725 K)
        tolerance: Tolerance for temperature validation
        
    Returns:
        Evaluator function that checks CMB temperature consistency
    """
    def evaluate_cmb_temperature(registry: ParameterRegistry) -> tuple[float, bool, str]:
        # Check if T_CMB_K parameter is set
        t_cmb_param = registry.get_parameter("T_CMB_K")
        if t_cmb_param is None or t_cmb_param.value is None:
            return 0.0, False, "T_CMB_K parameter not set"
        
        temp_value = t_cmb_param.value
        deviation = abs(temp_value - target_temp)
        is_valid = deviation <= tolerance
        
        if is_valid:
            message = f"CMB temperature = {temp_value:.6f} K within tolerance of {target_temp} ± {tolerance}"
        else:
            message = f"CMB temperature = {temp_value:.6f} K deviates from {target_temp} by {deviation:.6f} K (tolerance: {tolerance})"
        
        return temp_value, is_valid, message
    
    return evaluate_cmb_temperature


def create_vacuum_drag_evaluator(max_drag: float = 1e-10) -> EvaluatorFunction:
    """
    Create an evaluator for vacuum drag constraints from CMB spectral distortions.
    
    Args:
        max_drag: Maximum allowed vacuum drag to avoid μ/y spectral distortions
        
    Returns:
        Evaluator function that checks vacuum drag constraints
    """
    def evaluate_vacuum_drag(registry: ParameterRegistry) -> tuple[float, bool, str]:
        # Check k_J parameter (incoherent photon drag)
        k_j_param = registry.get_parameter("k_J")
        if k_j_param is None or k_j_param.value is None:
            return 0.0, False, "k_J parameter not set"
        
        drag_value = abs(k_j_param.value)  # Take absolute value
        is_valid = drag_value <= max_drag
        
        if is_valid:
            message = f"Vacuum drag |k_J| = {drag_value:.2e} ≤ {max_drag:.2e} (no CMB spectral distortions)"
        else:
            message = f"Vacuum drag |k_J| = {drag_value:.2e} > {max_drag:.2e} (would cause μ/y spectral distortions)"
        
        return drag_value, is_valid, message
    
    return evaluate_vacuum_drag


def create_thermalization_anchor_evaluator() -> EvaluatorFunction:
    """
    Create an evaluator for thermalization zeropoint consistency.
    
    Returns:
        Evaluator function that checks thermalization anchor
    """
    def evaluate_thermalization_anchor(registry: ParameterRegistry) -> tuple[float, bool, str]:
        # Check psi_s0 parameter (thermalization zeropoint)
        psi_s0_param = registry.get_parameter("psi_s0")
        if psi_s0_param is None or psi_s0_param.value is None:
            return 0.0, False, "psi_s0 parameter not set for thermalization anchor"
        
        psi_s0_value = psi_s0_param.value
        
        # In a full implementation, this would check that the thermalization
        # engine with this psi_s0 value produces T_CMB
        # For now, we just check that it's been set
        is_valid = True  # Placeholder - would compute actual thermalization
        message = f"Thermalization anchor psi_s0 = {psi_s0_value:.6f} set (thermalization check not yet implemented)"
        
        return psi_s0_value, is_valid, message
    
    return evaluate_thermalization_anchor


class CMBValidator(BaseValidator):
    """
    Validator for CMB (Cosmic Microwave Background) constraints.
    
    Validates that coupling constants are consistent with CMB observations:
    - Temperature thermalization (T_CMB ≈ 2.725 K)
    - No vacuum birefringence (polarization constraints)
    - No spectral distortions (vacuum drag constraints)
    - Parity conservation in vacuum (no TB/EB mixing)
    """
    
    def __init__(self, 
                 target_temp: float = 2.725,
                 temp_tolerance: float = 1e-6,
                 max_vacuum_drag: float = 1e-10,
                 allow_vacuum_birefringence: bool = False,
                 allow_parity_violation: bool = False):
        """
        Initialize CMB validator.
        
        Args:
            target_temp: Target CMB temperature in Kelvin
            temp_tolerance: Tolerance for temperature validation
            max_vacuum_drag: Maximum allowed vacuum drag
            allow_vacuum_birefringence: Whether to allow vacuum birefringence
            allow_parity_violation: Whether to allow parity violation in vacuum
        """
        super().__init__("CMB Validator")
        self.target_temp = target_temp
        self.temp_tolerance = temp_tolerance
        self.max_vacuum_drag = max_vacuum_drag
        self.allow_vacuum_birefringence = allow_vacuum_birefringence
        self.allow_parity_violation = allow_parity_violation
        
        # Set up derived constraint validator for CMB physics
        self.derived_validator = DerivedConstraintValidator("CMB Derived Constraints")
        self.derived_validator.add_evaluator("T_CMB_K", create_cmb_temperature_evaluator(target_temp, temp_tolerance))
        self.derived_validator.add_evaluator("k_J_vacuum_drag", create_vacuum_drag_evaluator(max_vacuum_drag))
        self.derived_validator.add_evaluator("psi_s0_anchor", create_thermalization_anchor_evaluator())
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """Validate CMB constraints."""
        start_time = time.time()
        
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        # Check CMB temperature constraint
        t_cmb_param = registry.get_parameter("T_CMB_K")
        if t_cmb_param is not None:
            result.parameters_checked += 1
            
            # Check for FIXED constraint on T_CMB_K
            fixed_constraints = [c for c in t_cmb_param.get_active_constraints() 
                               if c.constraint_type == ConstraintType.FIXED]
            result.constraints_checked += len(fixed_constraints)
            
            if not fixed_constraints:
                result.add_warning("T_CMB_K should have a FIXED constraint for CMB thermalization")
            else:
                # Validate fixed temperature value
                for constraint in fixed_constraints:
                    if abs(constraint.target_value - self.target_temp) > self.temp_tolerance:
                        violation = ValidationViolation(
                            parameter_name="T_CMB_K",
                            constraint_realm=constraint.realm,
                            violation_type="cmb_temperature_mismatch",
                            expected_value=self.target_temp,
                            actual_value=constraint.target_value,
                            message=f"CMB temperature constraint {constraint.target_value} K != expected {self.target_temp} K"
                        )
                        result.add_violation(violation)
        else:
            result.add_warning("T_CMB_K parameter not found - CMB thermalization cannot be validated")
        
        # Check vacuum drag constraints
        k_j_param = registry.get_parameter("k_J")
        if k_j_param is not None:
            result.parameters_checked += 1
            
            # Check for bounds that enforce low vacuum drag
            bounded_constraints = [c for c in k_j_param.get_active_constraints() 
                                 if c.constraint_type == ConstraintType.BOUNDED]
            result.constraints_checked += len(bounded_constraints)
            
            # Validate that bounds are tight enough to prevent spectral distortions
            for constraint in bounded_constraints:
                if constraint.max_value is not None and constraint.max_value > self.max_vacuum_drag:
                    result.add_warning(f"k_J upper bound {constraint.max_value:.2e} may allow spectral distortions (recommend ≤ {self.max_vacuum_drag:.2e})")
                if constraint.min_value is not None and constraint.min_value < -self.max_vacuum_drag:
                    result.add_warning(f"k_J lower bound {constraint.min_value:.2e} may allow spectral distortions (recommend ≥ {-self.max_vacuum_drag:.2e})")
        
        # Check polarization constraints
        self._validate_polarization_constraints(registry, result)
        
        # Run derived constraint validation
        derived_result = self.derived_validator.validate(registry)
        
        # Merge derived validation results
        result.violations.extend(derived_result.violations)
        result.warnings.extend(derived_result.warnings)
        result.info_messages.extend(derived_result.info_messages)
        
        if derived_result.status == ValidationStatus.INVALID:
            result.status = ValidationStatus.INVALID
        elif derived_result.status == ValidationStatus.WARNING and result.status == ValidationStatus.VALID:
            result.status = ValidationStatus.WARNING
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _validate_polarization_constraints(self, registry: ParameterRegistry, result: ValidationResult) -> None:
        """
        Validate polarization constraints (vacuum birefringence and parity).
        
        This is a placeholder for the full implementation that would check
        coupling constants for terms that induce vacuum birefringence or
        parity violation.
        """
        # Check configuration flags
        if self.allow_vacuum_birefringence:
            result.add_warning("Vacuum birefringence allowed - may conflict with CMB polarization constraints")
        else:
            result.add_info("Vacuum birefringence disallowed ✓")
        
        if self.allow_parity_violation:
            result.add_warning("Vacuum parity violation allowed - may conflict with CMB TB/EB constraints")
        else:
            result.add_info("Vacuum parity violation disallowed ✓")
        
        # In the full implementation, this would:
        # 1. Check coupling constants that could induce vacuum birefringence
        # 2. Validate that no TB/EB mixing terms are present in vacuum
        # 3. Check eta', xi combinations that act in vacuum
        
        # For now, add placeholder info
        result.add_info("Detailed polarization coupling validation not yet implemented")
    
    def get_required_parameters(self) -> list:
        """Get parameters required for CMB validation."""
        return ["T_CMB_K", "k_J", "psi_s0"]
    
    def is_applicable(self, registry: ParameterRegistry) -> bool:
        """Check if CMB validation is applicable."""
        if not self.enabled:
            return False
        
        # Applicable if any CMB-related parameters exist
        required_params = self.get_required_parameters()
        for param_name in required_params:
            if registry.get_parameter(param_name) is not None:
                return True
        
        return False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CMBValidator':
        """
        Create CMBValidator from configuration dictionary.
        
        Args:
            config: Configuration dictionary (e.g., from YAML)
            
        Returns:
            Configured CMBValidator instance
        """
        cmb_targets = config.get('cmb_targets', {})
        polarization = cmb_targets.get('polarization', {})
        
        return cls(
            target_temp=cmb_targets.get('T_CMB_K', 2.725),
            temp_tolerance=1e-6,  # Could be configurable
            max_vacuum_drag=1e-10,  # Could be configurable
            allow_vacuum_birefringence=polarization.get('allow_vacuum_birefringence', False),
            allow_parity_violation=polarization.get('allow_parity_violation', False)
        )
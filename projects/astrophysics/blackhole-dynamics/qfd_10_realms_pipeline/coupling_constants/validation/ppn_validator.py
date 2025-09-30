"""
PPN (Parametrized Post-Newtonian) constraint validator.

This module validates that coupling constants produce PPN parameters
consistent with General Relativity and observational constraints.
"""

import time
from typing import Optional
from .base_validator import BaseValidator, ValidationResult, ValidationViolation, ValidationStatus
from .derived_validator import DerivedConstraintValidator, create_ppn_gamma_evaluator, create_ppn_beta_evaluator
from ..registry.parameter_registry import ParameterRegistry, ConstraintType


class PPNValidator(BaseValidator):
    """
    Validator for PPN parameter consistency.
    
    Checks that PPN gamma and beta parameters are consistent with
    General Relativity predictions and observational constraints.
    """
    
    def __init__(self, 
                 gamma_target: float = 1.0, 
                 beta_target: float = 1.0,
                 gamma_tolerance: float = 1e-5,
                 beta_tolerance: float = 1e-4):
        """
        Initialize PPN validator.
        
        Args:
            gamma_target: Expected value of PPN gamma (GR: 1.0)
            beta_target: Expected value of PPN beta (GR: 1.0)
            gamma_tolerance: Tolerance for gamma (solar system: ~1e-5)
            beta_tolerance: Tolerance for beta (perihelion/LLR: ~1e-4)
        """
        super().__init__("PPN Validator")
        self.gamma_target = gamma_target
        self.beta_target = beta_target
        self.gamma_tolerance = gamma_tolerance
        self.beta_tolerance = beta_tolerance
        
        # Set up derived constraint validator for PPN relations
        self.derived_validator = DerivedConstraintValidator("PPN Derived Constraints")
        self.derived_validator.add_evaluator("PPN_gamma", create_ppn_gamma_evaluator(gamma_target, gamma_tolerance))
        self.derived_validator.add_evaluator("PPN_beta", create_ppn_beta_evaluator(beta_target, beta_tolerance))
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """Validate PPN parameter consistency."""
        start_time = time.time()
        
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        # Check if PPN parameters exist and have constraints
        ppn_gamma = registry.get_parameter("PPN_gamma")
        ppn_beta = registry.get_parameter("PPN_beta")
        
        if ppn_gamma is None:
            result.add_warning("PPN_gamma parameter not found in registry")
        if ppn_beta is None:
            result.add_warning("PPN_beta parameter not found in registry")
        
        # Check for PPN target constraints
        gamma_has_target = False
        beta_has_target = False
        
        if ppn_gamma:
            result.parameters_checked += 1
            target_constraints = [c for c in ppn_gamma.get_active_constraints() 
                                if c.constraint_type == ConstraintType.TARGET]
            result.constraints_checked += len(target_constraints)
            
            if target_constraints:
                gamma_has_target = True
                # Validate target constraint values
                for constraint in target_constraints:
                    if constraint.target_value != self.gamma_target:
                        violation = ValidationViolation(
                            parameter_name="PPN_gamma",
                            constraint_realm=constraint.realm,
                            violation_type="ppn_target_mismatch",
                            expected_value=self.gamma_target,
                            actual_value=constraint.target_value,
                            message=f"PPN gamma target {constraint.target_value} != expected GR value {self.gamma_target}"
                        )
                        result.add_violation(violation)
                    
                    if constraint.tolerance > self.gamma_tolerance:
                        result.add_warning(f"PPN gamma tolerance {constraint.tolerance} > recommended {self.gamma_tolerance}")
            else:
                result.add_warning("PPN_gamma has no target constraint")
        
        if ppn_beta:
            result.parameters_checked += 1
            target_constraints = [c for c in ppn_beta.get_active_constraints() 
                                if c.constraint_type == ConstraintType.TARGET]
            result.constraints_checked += len(target_constraints)
            
            if target_constraints:
                beta_has_target = True
                # Validate target constraint values
                for constraint in target_constraints:
                    if constraint.target_value != self.beta_target:
                        violation = ValidationViolation(
                            parameter_name="PPN_beta",
                            constraint_realm=constraint.realm,
                            violation_type="ppn_target_mismatch",
                            expected_value=self.beta_target,
                            actual_value=constraint.target_value,
                            message=f"PPN beta target {constraint.target_value} != expected GR value {self.beta_target}"
                        )
                        result.add_violation(violation)
                    
                    if constraint.tolerance > self.beta_tolerance:
                        result.add_warning(f"PPN beta tolerance {constraint.tolerance} > recommended {self.beta_tolerance}")
            else:
                result.add_warning("PPN_beta has no target constraint")
        
        # Run derived constraint validation if parameters have values
        if (ppn_gamma and ppn_gamma.value is not None) or (ppn_beta and ppn_beta.value is not None):
            derived_result = self.derived_validator.validate(registry)
            
            # Merge derived validation results
            result.violations.extend(derived_result.violations)
            result.warnings.extend(derived_result.warnings)
            result.info_messages.extend(derived_result.info_messages)
            
            if derived_result.status == ValidationStatus.INVALID:
                result.status = ValidationStatus.INVALID
            elif derived_result.status == ValidationStatus.WARNING and result.status == ValidationStatus.VALID:
                result.status = ValidationStatus.WARNING
        
        # Check for coupling constant consistency (placeholder for future implementation)
        self._check_coupling_consistency(registry, result)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _check_coupling_consistency(self, registry: ParameterRegistry, result: ValidationResult) -> None:
        """
        Check that coupling constants are consistent with PPN predictions.
        
        This is a placeholder for the actual physics implementation that would
        compute PPN parameters from the coupling constants and validate consistency.
        """
        # In the full implementation, this would:
        # 1. Get relevant coupling constants (a1, a2 from refractive index)
        # 2. Use the mapping from common/ppn.py to compute gamma, beta
        # 3. Compare with target values
        # 4. Report violations if inconsistent
        
        # For now, just add an info message
        result.add_info("Coupling constant -> PPN mapping validation not yet implemented")
    
    def get_required_parameters(self) -> list:
        """Get parameters required for PPN validation."""
        return ["PPN_gamma", "PPN_beta"]
    
    def is_applicable(self, registry: ParameterRegistry) -> bool:
        """Check if PPN validation is applicable."""
        if not self.enabled:
            return False
        
        # Only applicable if at least one PPN parameter exists
        ppn_gamma = registry.get_parameter("PPN_gamma")
        ppn_beta = registry.get_parameter("PPN_beta")
        
        return ppn_gamma is not None or ppn_beta is not None
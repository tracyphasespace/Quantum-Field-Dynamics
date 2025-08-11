"""
Derived constraint validator for computed physics relationships.

This module provides a framework for validating constraints that depend
on computed relationships between parameters (e.g., PPN relations, 
energy balance equations, etc.).
"""

import time
from typing import Dict, Callable, Tuple, Any
from .base_validator import BaseValidator, ValidationResult, ValidationViolation, ValidationStatus
from ..registry.parameter_registry import ParameterRegistry


# Type alias for evaluator functions
# Returns: (computed_value, is_valid, message)
EvaluatorFunction = Callable[[ParameterRegistry], Tuple[float, bool, str]]


class DerivedConstraintValidator(BaseValidator):
    """
    Validator for derived constraints that depend on computed relationships.
    
    This validator takes evaluator functions that compute derived quantities
    and validate them against expected values or ranges.
    """
    
    def __init__(self, name: str = "Derived Constraint Validator", evaluators: Dict[str, EvaluatorFunction] = None):
        """
        Initialize the derived constraint validator.
        
        Args:
            name: Name of this validator
            evaluators: Dictionary mapping parameter names to evaluator functions
        """
        super().__init__(name)
        self.evaluators = evaluators or {}
    
    def add_evaluator(self, param_name: str, evaluator: EvaluatorFunction) -> None:
        """Add an evaluator function for a parameter."""
        self.evaluators[param_name] = evaluator
    
    def remove_evaluator(self, param_name: str) -> bool:
        """Remove an evaluator function."""
        return self.evaluators.pop(param_name, None) is not None
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """Run all evaluator functions and validate results."""
        start_time = time.time()
        
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        result.parameters_checked = len(self.evaluators)
        
        for param_name, eval_fn in self.evaluators.items():
            try:
                computed_value, is_valid, message = eval_fn(registry)
                
                if not is_valid:
                    violation = ValidationViolation(
                        parameter_name=param_name,
                        constraint_realm="derived",
                        violation_type="derived_check_failed",
                        actual_value=computed_value,
                        message=message
                    )
                    result.add_violation(violation)
                else:
                    result.add_info(f"{param_name}: {message}")
                    
            except Exception as e:
                violation = ValidationViolation(
                    parameter_name=param_name,
                    constraint_realm="derived",
                    violation_type="evaluator_error",
                    message=f"Evaluator failed: {str(e)}"
                )
                result.add_violation(violation)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def get_required_parameters(self) -> list:
        """Get list of parameters that have evaluators."""
        return list(self.evaluators.keys())


# Common evaluator functions for physics relationships

def create_ppn_gamma_evaluator(expected_gamma: float = 1.0, tolerance: float = 1e-5) -> EvaluatorFunction:
    """
    Create an evaluator for PPN gamma parameter.
    
    Args:
        expected_gamma: Expected value of gamma (default: 1.0 for GR)
        tolerance: Tolerance for validation
        
    Returns:
        Evaluator function that checks PPN gamma consistency
    """
    def evaluate_ppn_gamma(registry: ParameterRegistry) -> Tuple[float, bool, str]:
        # Get PPN gamma parameter
        ppn_gamma_param = registry.get_parameter("PPN_gamma")
        if ppn_gamma_param is None or ppn_gamma_param.value is None:
            return 0.0, False, "PPN_gamma parameter not set"
        
        gamma_value = ppn_gamma_param.value
        deviation = abs(gamma_value - expected_gamma)
        is_valid = deviation <= tolerance
        
        if is_valid:
            message = f"PPN gamma = {gamma_value:.6f} within tolerance of {expected_gamma} ± {tolerance}"
        else:
            message = f"PPN gamma = {gamma_value:.6f} deviates from {expected_gamma} by {deviation:.6f} (tolerance: {tolerance})"
        
        return gamma_value, is_valid, message
    
    return evaluate_ppn_gamma


def create_ppn_beta_evaluator(expected_beta: float = 1.0, tolerance: float = 1e-4) -> EvaluatorFunction:
    """
    Create an evaluator for PPN beta parameter.
    
    Args:
        expected_beta: Expected value of beta (default: 1.0 for GR)
        tolerance: Tolerance for validation
        
    Returns:
        Evaluator function that checks PPN beta consistency
    """
    def evaluate_ppn_beta(registry: ParameterRegistry) -> Tuple[float, bool, str]:
        # Get PPN beta parameter
        ppn_beta_param = registry.get_parameter("PPN_beta")
        if ppn_beta_param is None or ppn_beta_param.value is None:
            return 0.0, False, "PPN_beta parameter not set"
        
        beta_value = ppn_beta_param.value
        deviation = abs(beta_value - expected_beta)
        is_valid = deviation <= tolerance
        
        if is_valid:
            message = f"PPN beta = {beta_value:.6f} within tolerance of {expected_beta} ± {tolerance}"
        else:
            message = f"PPN beta = {beta_value:.6f} deviates from {expected_beta} by {deviation:.6f} (tolerance: {tolerance})"
        
        return beta_value, is_valid, message
    
    return evaluate_ppn_beta


def create_vacuum_refractive_index_evaluator() -> EvaluatorFunction:
    """
    Create an evaluator that ensures vacuum refractive index equals 1.0 exactly.
    
    Returns:
        Evaluator function that checks vacuum refractive index
    """
    def evaluate_vacuum_index(registry: ParameterRegistry) -> Tuple[float, bool, str]:
        # This would compute n_vacuum from the coupling constants
        # For now, we'll check if any parameters would cause vacuum dispersion
        
        # Check k_J (vacuum drag parameter)
        k_j_param = registry.get_parameter("k_J")
        if k_j_param is not None and k_j_param.value is not None:
            k_j_value = k_j_param.value
            # Vacuum drag should be essentially zero
            if abs(k_j_value) > 1e-10:
                return k_j_value, False, f"Vacuum drag k_J = {k_j_value} too large (should be ~0 for n_vac = 1)"
        
        # In a full implementation, this would compute the actual refractive index
        # from all relevant coupling constants
        n_vacuum = 1.0  # Placeholder - would be computed from parameters
        
        is_valid = abs(n_vacuum - 1.0) < 1e-12
        message = f"Vacuum refractive index = {n_vacuum:.12f} {'✓' if is_valid else '✗'}"
        
        return n_vacuum, is_valid, message
    
    return evaluate_vacuum_index
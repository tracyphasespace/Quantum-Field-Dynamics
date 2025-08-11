"""
Basic constraint validators for common validation scenarios.

This module provides simple validators that can be used to test
the validation framework and handle common constraint checking.
"""

import time
from typing import List
from .base_validator import BaseValidator, ValidationResult, ValidationViolation, ValidationStatus
from ..registry.parameter_registry import ParameterRegistry, ConstraintType


class BoundsValidator(BaseValidator):
    """Validates that all parameters respect their bounded constraints."""
    
    def __init__(self):
        super().__init__("Bounds Validator")
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """Check all parameters against their bounded constraints."""
        start_time = time.time()
        
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        all_params = registry.get_all_parameters()
        result.parameters_checked = len(all_params)
        
        for param_name, param in all_params.items():
            if param.value is None:
                continue  # Skip unset parameters
                
            bounded_constraints = [
                c for c in param.get_active_constraints() 
                if c.constraint_type == ConstraintType.BOUNDED
            ]
            
            result.constraints_checked += len(bounded_constraints)
            
            for constraint in bounded_constraints:
                if not constraint.is_satisfied(param.value):
                    violation = ValidationViolation(
                        parameter_name=param_name,
                        constraint_realm=constraint.realm,
                        violation_type="bounds_violation",
                        actual_value=param.value,
                        expected_range=(constraint.min_value, constraint.max_value),
                        message=f"Parameter {param_name} = {param.value} violates bounds [{constraint.min_value}, {constraint.max_value}] from {constraint.realm}"
                    )
                    result.add_violation(violation)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result


class FixedValueValidator(BaseValidator):
    """Validates that fixed parameters have their required values."""
    
    def __init__(self):
        super().__init__("Fixed Value Validator")
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """Check all parameters against their fixed value constraints."""
        start_time = time.time()
        
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        all_params = registry.get_all_parameters()
        result.parameters_checked = len(all_params)
        
        for param_name, param in all_params.items():
            fixed_constraints = [
                c for c in param.get_active_constraints() 
                if c.constraint_type == ConstraintType.FIXED
            ]
            
            result.constraints_checked += len(fixed_constraints)
            
            for constraint in fixed_constraints:
                if param.value is None:
                    violation = ValidationViolation(
                        parameter_name=param_name,
                        constraint_realm=constraint.realm,
                        violation_type="missing_fixed_value",
                        expected_value=constraint.target_value,
                        actual_value=None,
                        tolerance=constraint.tolerance,
                        message=f"Fixed parameter {param_name} has no value (expected {constraint.target_value})"
                    )
                    result.add_violation(violation)
                elif not constraint.is_satisfied(param.value):
                    violation = ValidationViolation(
                        parameter_name=param_name,
                        constraint_realm=constraint.realm,
                        violation_type="fixed_value_violation",
                        expected_value=constraint.target_value,
                        actual_value=param.value,
                        tolerance=constraint.tolerance,
                        message=f"Fixed parameter {param_name} = {param.value} does not match required value {constraint.target_value} ± {constraint.tolerance}"
                    )
                    result.add_violation(violation)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result


class TargetValueValidator(BaseValidator):
    """Validates that target parameters are within tolerance of their targets."""
    
    def __init__(self):
        super().__init__("Target Value Validator")
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """Check all parameters against their target value constraints."""
        start_time = time.time()
        
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        all_params = registry.get_all_parameters()
        result.parameters_checked = len(all_params)
        
        for param_name, param in all_params.items():
            target_constraints = [
                c for c in param.get_active_constraints() 
                if c.constraint_type == ConstraintType.TARGET
            ]
            
            result.constraints_checked += len(target_constraints)
            
            for constraint in target_constraints:
                if param.value is None:
                    result.add_warning(f"Target parameter {param_name} has no value (target: {constraint.target_value})")
                elif not constraint.is_satisfied(param.value):
                    violation = ValidationViolation(
                        parameter_name=param_name,
                        constraint_realm=constraint.realm,
                        violation_type="target_value_violation",
                        expected_value=constraint.target_value,
                        actual_value=param.value,
                        tolerance=constraint.tolerance,
                        message=f"Parameter {param_name} = {param.value} is outside target tolerance (target: {constraint.target_value} ± {constraint.tolerance})",
                        severity="warning"  # Target violations are often warnings, not hard errors
                    )
                    result.add_violation(violation)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result


class ConflictValidator(BaseValidator):
    """Validates that there are no conflicting constraints between realms."""
    
    def __init__(self):
        super().__init__("Conflict Validator")
    
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """Check for conflicting constraints."""
        start_time = time.time()
        
        result = ValidationResult(
            validator_name=self.name,
            status=ValidationStatus.VALID
        )
        
        conflicts = registry.get_conflicting_constraints()
        result.parameters_checked = len(registry.get_all_parameters())
        
        for conflict in conflicts:
            param_name = conflict["parameter"]
            conflict_type = conflict["type"]
            
            if conflict_type == "multiple_fixed_values":
                violation = ValidationViolation(
                    parameter_name=param_name,
                    constraint_realm="multiple",
                    violation_type="multiple_fixed_values",
                    message=f"Parameter {param_name} has multiple fixed values from different realms: {conflict['constraints']}"
                )
                result.add_violation(violation)
                
            elif conflict_type == "incompatible_bounds":
                violation = ValidationViolation(
                    parameter_name=param_name,
                    constraint_realm="multiple",
                    violation_type="incompatible_bounds",
                    message=f"Parameter {param_name} has incompatible bounds: max_min={conflict['max_min_bound']} > min_max={conflict['min_max_bound']}"
                )
                result.add_violation(violation)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
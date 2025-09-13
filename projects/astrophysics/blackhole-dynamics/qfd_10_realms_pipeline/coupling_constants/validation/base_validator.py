"""
Base constraint validator interface and validation result structures.

This module defines the abstract base class for all constraint validators
and the data structures used to report validation results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from ..registry.parameter_registry import ParameterRegistry


class ValidationStatus(Enum):
    """Status of a validation check."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    SKIPPED = "skipped"


@dataclass
class ValidationViolation:
    """Represents a single constraint violation."""
    parameter_name: str
    constraint_realm: str
    violation_type: str
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    expected_range: Optional[tuple] = None
    tolerance: Optional[float] = None
    message: str = ""
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Result of a constraint validation check."""
    validator_name: str
    status: ValidationStatus
    violations: List[ValidationViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)
    parameters_checked: int = 0
    constraints_checked: int = 0
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_violation(self, violation: ValidationViolation) -> None:
        """Add a violation to this result."""
        self.violations.append(violation)
        if self.status == ValidationStatus.VALID:
            self.status = ValidationStatus.INVALID
        
        # Update metadata with violation type counts
        if "violation_types" not in self.metadata:
            self.metadata["violation_types"] = {}
        violation_type = violation.violation_type
        self.metadata["violation_types"][violation_type] = self.metadata["violation_types"].get(violation_type, 0) + 1

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        if self.status == ValidationStatus.VALID:
            self.status = ValidationStatus.WARNING

    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.info_messages.append(message)

    def is_valid(self) -> bool:
        """Check if validation passed without errors."""
        return self.status in [ValidationStatus.VALID, ValidationStatus.WARNING]

    def get_summary(self) -> str:
        """Get a summary string of the validation result."""
        summary = f"{self.validator_name}: {self.status.value}"
        if self.violations:
            summary += f" ({len(self.violations)} violations)"
        if self.warnings:
            summary += f" ({len(self.warnings)} warnings)"
        return summary


@dataclass
class ValidationReport:
    """Comprehensive validation report from multiple validators."""
    timestamp: datetime
    overall_status: ValidationStatus
    validator_results: List[ValidationResult] = field(default_factory=list)
    total_parameters: int = 0
    total_constraints: int = 0
    total_violations: int = 0
    total_warnings: int = 0
    execution_time_ms: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def add_result(self, result: ValidationResult) -> None:
        """Add a validator result to this report."""
        self.validator_results.append(result)
        self.total_violations += len(result.violations)
        self.total_warnings += len(result.warnings)
        self.execution_time_ms += result.execution_time_ms
        
        # Update overall status
        if result.status == ValidationStatus.INVALID:
            self.overall_status = ValidationStatus.INVALID
        elif result.status == ValidationStatus.WARNING and self.overall_status == ValidationStatus.VALID:
            self.overall_status = ValidationStatus.WARNING

    def get_all_violations(self) -> List[ValidationViolation]:
        """Get all violations from all validator results."""
        violations = []
        for result in self.validator_results:
            violations.extend(result.violations)
        return violations

    def get_violations_by_parameter(self) -> Dict[str, List[ValidationViolation]]:
        """Group violations by parameter name."""
        violations_by_param = {}
        for violation in self.get_all_violations():
            param_name = violation.parameter_name
            if param_name not in violations_by_param:
                violations_by_param[param_name] = []
            violations_by_param[param_name].append(violation)
        return violations_by_param

    def get_summary(self) -> str:
        """Get a summary string of the entire validation report."""
        summary = f"Validation Report: {self.overall_status.value}\n"
        summary += f"  Parameters: {self.total_parameters}\n"
        summary += f"  Constraints: {self.total_constraints}\n"
        summary += f"  Violations: {self.total_violations}\n"
        summary += f"  Warnings: {self.total_warnings}\n"
        summary += f"  Execution time: {self.execution_time_ms:.2f}ms\n"
        
        if self.validator_results:
            summary += "  Validator Results:\n"
            for result in self.validator_results:
                summary += f"    - {result.get_summary()}\n"
        
        return summary


class BaseValidator(ABC):
    """
    Abstract base class for all constraint validators.
    
    Each validator is responsible for checking specific types of constraints
    or physics requirements (e.g., PPN consistency, vacuum constraints, etc.).
    """
    
    def __init__(self, name: str):
        """
        Initialize the validator.
        
        Args:
            name: Human-readable name for this validator
        """
        self.name = name
        self.enabled = True
        
    @abstractmethod
    def validate(self, registry: ParameterRegistry) -> ValidationResult:
        """
        Perform validation on the parameter registry.
        
        Args:
            registry: ParameterRegistry instance to validate
            
        Returns:
            ValidationResult with the outcome of validation
        """
        pass
    
    def is_applicable(self, registry: ParameterRegistry) -> bool:
        """
        Check if this validator is applicable to the current registry state.
        
        Override this method if the validator should only run under certain conditions.
        
        Args:
            registry: ParameterRegistry instance
            
        Returns:
            True if validator should run, False to skip
        """
        return self.enabled
    
    def get_required_parameters(self) -> List[str]:
        """
        Get list of parameter names required by this validator.
        
        Returns:
            List of parameter names that must be present for validation
        """
        return []
    
    def get_description(self) -> str:
        """
        Get a description of what this validator checks.
        
        Returns:
            Human-readable description of validator purpose
        """
        return f"Validator: {self.name}"


class CompositeValidator:
    """
    Composite validator that runs multiple validators and combines results.
    """
    
    def __init__(self, name: str = "Composite Validator"):
        """Initialize the composite validator."""
        self.name = name
        self.validators: List[BaseValidator] = []
        
    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the composite."""
        self.validators.append(validator)
        
    def remove_validator(self, validator_name: str) -> bool:
        """Remove a validator by name."""
        for i, validator in enumerate(self.validators):
            if validator.name == validator_name:
                del self.validators[i]
                return True
        return False
        
    def validate_all(self, registry: ParameterRegistry) -> ValidationReport:
        """
        Run all validators and generate a comprehensive report.
        
        Args:
            registry: ParameterRegistry instance to validate
            
        Returns:
            ValidationReport with results from all validators
        """
        start_time = datetime.utcnow()
        
        report = ValidationReport(
            timestamp=start_time,
            overall_status=ValidationStatus.VALID,
            total_parameters=len(registry.get_all_parameters())
        )
        
        # Count total constraints
        total_constraints = 0
        for param in registry.get_all_parameters().values():
            total_constraints += len(param.get_active_constraints())
        report.total_constraints = total_constraints
        
        # Run each validator
        for validator in self.validators:
            if validator.is_applicable(registry):
                try:
                    result = validator.validate(registry)
                    report.add_result(result)
                except Exception as e:
                    # Create error result for failed validator
                    error_result = ValidationResult(
                        validator_name=validator.name,
                        status=ValidationStatus.INVALID
                    )
                    error_result.add_violation(ValidationViolation(
                        parameter_name="N/A",
                        constraint_realm="validator",
                        violation_type="validator_error",
                        message=f"Validator failed with error: {str(e)}"
                    ))
                    report.add_result(error_result)
            else:
                # Create skipped result
                skipped_result = ValidationResult(
                    validator_name=validator.name,
                    status=ValidationStatus.SKIPPED
                )
                skipped_result.add_info("Validator skipped (not applicable)")
                report.add_result(skipped_result)
        
        # Calculate total execution time
        end_time = datetime.utcnow()
        report.execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return report
    
    def get_validator_names(self) -> List[str]:
        """Get names of all registered validators."""
        return [v.name for v in self.validators]
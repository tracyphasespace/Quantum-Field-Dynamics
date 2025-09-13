"""
Unit tests for the validation framework.
"""

import unittest
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, ParameterState, Constraint, ConstraintType
)
from coupling_constants.validation.base_validator import (
    BaseValidator, CompositeValidator, ValidationResult, ValidationStatus
)
from coupling_constants.validation.basic_validators import (
    BoundsValidator, FixedValueValidator, TargetValueValidator, ConflictValidator
)


class TestValidationFramework(unittest.TestCase):
    """Test the validation framework components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
    def test_bounds_validator(self):
        """Test the bounds validator."""
        # Add parameter with bounds
        constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        self.registry.add_constraint("test_param", constraint)
        
        validator = BoundsValidator()
        
        # Test valid value
        self.registry.update_parameter("test_param", 0.5, "test_realm")
        result = validator.validate(self.registry)
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertEqual(len(result.violations), 0)
        
        # Test invalid value
        self.registry.update_parameter("test_param", 1.5, "test_realm")
        result = validator.validate(self.registry)
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].violation_type, "bounds_violation")
        
    def test_fixed_value_validator(self):
        """Test the fixed value validator."""
        # Add fixed parameter
        constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6
        )
        self.registry.add_constraint("T_CMB", constraint)
        
        validator = FixedValueValidator()
        
        # Test missing value
        result = validator.validate(self.registry)
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].violation_type, "missing_fixed_value")
        
        # Test correct value (use same realm that added the constraint)
        self.registry.update_parameter("T_CMB", 2.725, "test")
        result = validator.validate(self.registry)
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertEqual(len(result.violations), 0)
        
        # Test incorrect value (use same realm)
        self.registry.update_parameter("T_CMB", 2.8, "test")
        result = validator.validate(self.registry)
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].violation_type, "fixed_value_violation")
        
    def test_target_value_validator(self):
        """Test the target value validator."""
        # Add target parameter
        constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.TARGET,
            target_value=1.0,
            tolerance=1e-5
        )
        self.registry.add_constraint("PPN_gamma", constraint)
        
        validator = TargetValueValidator()
        
        # Test within tolerance
        self.registry.update_parameter("PPN_gamma", 1.000005, "test_realm")
        result = validator.validate(self.registry)
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertEqual(len(result.violations), 0)
        
        # Test outside tolerance
        self.registry.update_parameter("PPN_gamma", 1.00002, "test_realm")
        result = validator.validate(self.registry)
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].violation_type, "target_value_violation")
        
    def test_conflict_validator(self):
        """Test the conflict validator."""
        # Add conflicting constraints
        constraint1 = Constraint(
            realm="realm1",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.5,
            max_value=1.0
        )
        constraint2 = Constraint(
            realm="realm2",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.8,
            max_value=0.6  # This creates a conflict
        )
        
        self.registry.add_constraint("conflict_param", constraint1)
        self.registry.add_constraint("conflict_param", constraint2)
        
        validator = ConflictValidator()
        result = validator.validate(self.registry)
        
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].violation_type, "incompatible_bounds")
        
    def test_composite_validator(self):
        """Test the composite validator."""
        # Set up registry with various constraints
        bounds_constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        fixed_constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6
        )
        
        self.registry.add_constraint("bounded_param", bounds_constraint)
        self.registry.add_constraint("fixed_param", fixed_constraint)
        
        # Set valid values
        self.registry.update_parameter("bounded_param", 0.5, "test")
        self.registry.update_parameter("fixed_param", 2.725, "test")
        
        # Create composite validator
        composite = CompositeValidator()
        composite.add_validator(BoundsValidator())
        composite.add_validator(FixedValueValidator())
        composite.add_validator(ConflictValidator())
        
        # Run validation
        report = composite.validate_all(self.registry)
        
        self.assertEqual(report.overall_status, ValidationStatus.VALID)
        self.assertEqual(len(report.validator_results), 3)
        self.assertEqual(report.total_violations, 0)
        self.assertGreater(report.total_parameters, 0)
        self.assertGreater(report.total_constraints, 0)


if __name__ == "__main__":
    unittest.main()
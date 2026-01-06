"""
Tests for enhanced validation features.
"""

import unittest
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, ParameterState, Constraint, ConstraintType
)
from coupling_constants.validation.ppn_validator import PPNValidator
from coupling_constants.validation.derived_validator import DerivedConstraintValidator, create_ppn_gamma_evaluator
from coupling_constants.config.yaml_loader import load_parameters_from_yaml


class TestEnhancedValidation(unittest.TestCase):
    """Test enhanced validation features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
    def test_enhanced_conflict_detection(self):
        """Test enhanced conflict detection for FIXED vs BOUNDED."""
        # Add FIXED constraint
        fixed_constraint = Constraint(
            realm="realm1",
            constraint_type=ConstraintType.FIXED,
            target_value=5.0,
            tolerance=1e-6
        )
        
        # Add BOUNDED constraint that conflicts
        bounded_constraint = Constraint(
            realm="realm2",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=3.0  # Fixed value 5.0 is outside this range
        )
        
        self.registry.add_constraint("test_param", fixed_constraint)
        self.registry.add_constraint("test_param", bounded_constraint)
        
        conflicts = self.registry.get_conflicting_constraints()
        
        # Should detect fixed_outside_bounds conflict
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "fixed_outside_bounds")
        self.assertEqual(conflicts[0]["violation"], "above_maximum")
        
    def test_fixed_vs_target_conflict(self):
        """Test conflict detection between FIXED and TARGET constraints."""
        # Add FIXED constraint
        fixed_constraint = Constraint(
            realm="realm1",
            constraint_type=ConstraintType.FIXED,
            target_value=1.0,
            tolerance=1e-6
        )
        
        # Add TARGET constraint with different value
        target_constraint = Constraint(
            realm="realm2",
            constraint_type=ConstraintType.TARGET,
            target_value=1.1,
            tolerance=1e-5
        )
        
        self.registry.add_constraint("test_param", fixed_constraint)
        self.registry.add_constraint("test_param", target_constraint)
        
        conflicts = self.registry.get_conflicting_constraints()
        
        # Should detect fixed_vs_target_mismatch
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "fixed_vs_target_mismatch")
        self.assertAlmostEqual(conflicts[0]["actual_difference"], 0.1, places=6)
        
    def test_derived_constraint_validator(self):
        """Test derived constraint validator."""
        # Set up PPN gamma parameter
        self.registry.update_parameter("PPN_gamma", 1.000001, "test_realm")
        
        # Create derived validator with PPN gamma evaluator
        validator = DerivedConstraintValidator()
        validator.add_evaluator("PPN_gamma", create_ppn_gamma_evaluator(1.0, 1e-5))
        
        # Should pass (within tolerance)
        result = validator.validate(self.registry)
        self.assertEqual(result.status.value, "valid")
        self.assertEqual(len(result.violations), 0)
        
        # Test with value outside tolerance
        self.registry.update_parameter("PPN_gamma", 1.00002, "test_realm")
        result = validator.validate(self.registry)
        self.assertEqual(result.status.value, "invalid")
        self.assertEqual(len(result.violations), 1)
        
    def test_ppn_validator(self):
        """Test PPN validator."""
        # Add PPN parameters with target constraints
        gamma_constraint = Constraint(
            realm="ppn_config",
            constraint_type=ConstraintType.TARGET,
            target_value=1.0,
            tolerance=1e-5
        )
        beta_constraint = Constraint(
            realm="ppn_config",
            constraint_type=ConstraintType.TARGET,
            target_value=1.0,
            tolerance=1e-4
        )
        
        self.registry.add_constraint("PPN_gamma", gamma_constraint)
        self.registry.add_constraint("PPN_beta", beta_constraint)
        
        # Set parameter values
        self.registry.update_parameter("PPN_gamma", 1.000001, "test_realm")
        self.registry.update_parameter("PPN_beta", 0.99999, "test_realm")
        
        # Run PPN validator
        validator = PPNValidator()
        result = validator.validate(self.registry)
        
        self.assertEqual(result.status.value, "valid")
        self.assertGreater(result.parameters_checked, 0)
        self.assertGreater(result.constraints_checked, 0)
        
    def test_yaml_schema_validation(self):
        """Test YAML schema validation."""
        # This should work with the actual config file
        try:
            load_parameters_from_yaml("qfd_params/defaults.yaml", self.registry)
            # Should succeed without errors
            self.assertGreater(len(self.registry.get_all_parameters()), 0)
        except Exception as e:
            self.fail(f"YAML loading failed: {e}")
            
    def test_validation_metadata(self):
        """Test that validation results include metadata."""
        # Add parameter with bounds
        constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        self.registry.add_constraint("test_param", constraint)
        
        # Set invalid value
        self.registry.update_parameter("test_param", 1.5, "test_realm")
        
        from coupling_constants.validation.basic_validators import BoundsValidator
        validator = BoundsValidator()
        result = validator.validate(self.registry)
        
        # Check that metadata includes violation type counts
        self.assertIn("violation_types", result.metadata)
        self.assertIn("bounds_violation", result.metadata["violation_types"])
        self.assertEqual(result.metadata["violation_types"]["bounds_violation"], 1)


if __name__ == "__main__":
    unittest.main()
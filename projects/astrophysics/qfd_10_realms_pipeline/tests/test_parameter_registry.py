"""
Unit tests for the parameter registry module.
"""

import unittest
from datetime import datetime
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, ParameterState, Constraint, ConstraintType, ParameterChange
)


class TestParameterRegistry(unittest.TestCase):
    """Test cases for ParameterRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
    def test_parameter_registration(self):
        """Test basic parameter registration."""
        param = ParameterState(name="k_J", value=0.1)
        self.registry.register_parameter(param)
        
        retrieved = self.registry.get_parameter("k_J")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "k_J")
        self.assertEqual(retrieved.value, 0.1)
        
    def test_parameter_update(self):
        """Test parameter value updates."""
        self.registry.update_parameter("k_J", 0.1, "realm0", "Initial value")
        
        param = self.registry.get_parameter("k_J")
        self.assertIsNotNone(param)
        self.assertEqual(param.value, 0.1)
        self.assertEqual(len(param.history), 1)
        self.assertEqual(param.history[0].realm, "realm0")
        
    def test_constraint_addition(self):
        """Test adding constraints to parameters."""
        constraint = Constraint(
            realm="realm0",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0,
            notes="Must be between 0 and 1"
        )
        
        self.registry.add_constraint("k_J", constraint)
        
        param = self.registry.get_parameter("k_J")
        self.assertIsNotNone(param)
        self.assertEqual(len(param.constraints), 1)
        self.assertEqual(param.constraints[0].realm, "realm0")
        
    def test_constraint_validation(self):
        """Test constraint validation."""
        # Add a bounded constraint
        constraint = Constraint(
            realm="realm0",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        self.registry.add_constraint("k_J", constraint)
        
        # Valid value
        self.registry.update_parameter("k_J", 0.5, "realm1")
        param = self.registry.get_parameter("k_J")
        violations = param.validate_constraints()
        self.assertEqual(len(violations), 0)
        
        # Invalid value
        self.registry.update_parameter("k_J", 1.5, "realm1")
        violations = param.validate_constraints()
        self.assertGreater(len(violations), 0)
        
    def test_fixed_parameter_protection(self):
        """Test that fixed parameters cannot be modified by other realms."""
        # Fix parameter in realm0
        constraint = Constraint(
            realm="realm0",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6
        )
        self.registry.add_constraint("T_CMB", constraint)
        self.registry.update_parameter("T_CMB", 2.725, "realm0")
        
        # Try to modify from another realm - should raise error
        with self.assertRaises(ValueError):
            self.registry.update_parameter("T_CMB", 2.8, "realm1")
            
    def test_conflict_detection(self):
        """Test detection of conflicting constraints."""
        # Add conflicting bounds
        constraint1 = Constraint(
            realm="realm0",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.5,
            max_value=1.0
        )
        constraint2 = Constraint(
            realm="realm1", 
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.8,
            max_value=0.6  # This creates a conflict
        )
        
        self.registry.add_constraint("test_param", constraint1)
        self.registry.add_constraint("test_param", constraint2)
        
        conflicts = self.registry.get_conflicting_constraints()
        self.assertGreater(len(conflicts), 0)
        self.assertEqual(conflicts[0]["parameter"], "test_param")
        self.assertEqual(conflicts[0]["type"], "incompatible_bounds")


class TestConstraint(unittest.TestCase):
    """Test cases for Constraint class."""
    
    def test_bounded_constraint_satisfaction(self):
        """Test bounded constraint satisfaction."""
        constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        
        self.assertTrue(constraint.is_satisfied(0.5))
        self.assertTrue(constraint.is_satisfied(0.0))
        self.assertTrue(constraint.is_satisfied(1.0))
        self.assertFalse(constraint.is_satisfied(-0.1))
        self.assertFalse(constraint.is_satisfied(1.1))
        
    def test_fixed_constraint_satisfaction(self):
        """Test fixed constraint satisfaction."""
        constraint = Constraint(
            realm="test",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6
        )
        
        self.assertTrue(constraint.is_satisfied(2.725))
        self.assertTrue(constraint.is_satisfied(2.725 + 5e-7))
        self.assertFalse(constraint.is_satisfied(2.726))


if __name__ == "__main__":
    unittest.main()
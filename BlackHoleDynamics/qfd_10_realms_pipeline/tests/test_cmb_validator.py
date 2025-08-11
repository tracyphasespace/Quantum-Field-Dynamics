"""
Tests for CMB constraint validator.
"""

import unittest
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, ParameterState, Constraint, ConstraintType
)
from coupling_constants.validation.cmb_validator import (
    CMBValidator, create_cmb_temperature_evaluator, create_vacuum_drag_evaluator
)
from coupling_constants.config.yaml_loader import load_parameters_from_yaml


class TestCMBValidator(unittest.TestCase):
    """Test CMB constraint validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
    def test_cmb_temperature_evaluator(self):
        """Test CMB temperature evaluator."""
        evaluator = create_cmb_temperature_evaluator(2.725, 1e-6)
        
        # Test missing parameter
        value, is_valid, message = evaluator(self.registry)
        self.assertFalse(is_valid)
        self.assertIn("not set", message)
        
        # Test valid temperature
        self.registry.update_parameter("T_CMB_K", 2.725, "test_realm")
        value, is_valid, message = evaluator(self.registry)
        self.assertTrue(is_valid)
        self.assertEqual(value, 2.725)
        self.assertIn("within tolerance", message)
        
        # Test invalid temperature
        self.registry.update_parameter("T_CMB_K", 2.8, "test_realm")
        value, is_valid, message = evaluator(self.registry)
        self.assertFalse(is_valid)
        self.assertEqual(value, 2.8)
        self.assertIn("deviates", message)
        
    def test_vacuum_drag_evaluator(self):
        """Test vacuum drag evaluator."""
        evaluator = create_vacuum_drag_evaluator(1e-10)
        
        # Test missing parameter
        value, is_valid, message = evaluator(self.registry)
        self.assertFalse(is_valid)
        self.assertIn("not set", message)
        
        # Test valid drag (small)
        self.registry.update_parameter("k_J", 1e-12, "test_realm")
        value, is_valid, message = evaluator(self.registry)
        self.assertTrue(is_valid)
        self.assertEqual(value, 1e-12)
        self.assertIn("no CMB spectral distortions", message)
        
        # Test invalid drag (too large)
        self.registry.update_parameter("k_J", 1e-8, "test_realm")
        value, is_valid, message = evaluator(self.registry)
        self.assertFalse(is_valid)
        self.assertEqual(value, 1e-8)
        self.assertIn("spectral distortions", message)
        
        # Test negative drag (should take absolute value)
        self.registry.update_parameter("k_J", -1e-12, "test_realm")
        value, is_valid, message = evaluator(self.registry)
        self.assertTrue(is_valid)
        self.assertEqual(value, 1e-12)  # Should be absolute value
        
    def test_cmb_validator_basic(self):
        """Test basic CMB validator functionality."""
        # Add CMB temperature constraint
        t_cmb_constraint = Constraint(
            realm="cmb_config",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6
        )
        self.registry.add_constraint("T_CMB_K", t_cmb_constraint)
        
        # Add vacuum drag bounds
        k_j_constraint = Constraint(
            realm="cmb_config",
            constraint_type=ConstraintType.BOUNDED,
            min_value=-1e-10,
            max_value=1e-10
        )
        self.registry.add_constraint("k_J", k_j_constraint)
        
        # Set parameter values
        self.registry.update_parameter("T_CMB_K", 2.725, "cmb_config")
        self.registry.update_parameter("k_J", 1e-12, "realm0_cmb")
        self.registry.update_parameter("psi_s0", -1.5, "realm0_cmb")
        
        # Run CMB validator
        validator = CMBValidator()
        result = validator.validate(self.registry)
        
        self.assertEqual(result.status.value, "valid")
        self.assertGreater(result.parameters_checked, 0)
        self.assertGreater(result.constraints_checked, 0)
        
    def test_cmb_validator_temperature_mismatch(self):
        """Test CMB validator with temperature mismatch."""
        # Add CMB temperature constraint with wrong value
        t_cmb_constraint = Constraint(
            realm="cmb_config",
            constraint_type=ConstraintType.FIXED,
            target_value=2.8,  # Wrong temperature
            tolerance=1e-6
        )
        self.registry.add_constraint("T_CMB_K", t_cmb_constraint)
        
        validator = CMBValidator(target_temp=2.725)
        result = validator.validate(self.registry)
        
        self.assertEqual(result.status.value, "invalid")
        
        # Check that we have the expected violation type
        violation_types = [v.violation_type for v in result.violations]
        self.assertIn("cmb_temperature_mismatch", violation_types)
        
    def test_cmb_validator_vacuum_drag_warning(self):
        """Test CMB validator with loose vacuum drag bounds."""
        # Add loose vacuum drag bounds
        k_j_constraint = Constraint(
            realm="config",
            constraint_type=ConstraintType.BOUNDED,
            min_value=-1e-6,  # Too loose
            max_value=1e-6    # Too loose
        )
        self.registry.add_constraint("k_J", k_j_constraint)
        
        validator = CMBValidator(max_vacuum_drag=1e-10)
        result = validator.validate(self.registry)
        
        # Should generate warnings about loose bounds
        self.assertGreater(len(result.warnings), 0)
        warning_messages = " ".join(result.warnings)
        self.assertIn("spectral distortions", warning_messages)
        
    def test_cmb_validator_from_config(self):
        """Test creating CMB validator from configuration."""
        config = {
            'cmb_targets': {
                'T_CMB_K': 2.725,
                'polarization': {
                    'allow_vacuum_birefringence': False,
                    'allow_parity_violation': False
                }
            }
        }
        
        validator = CMBValidator.from_config(config)
        
        self.assertEqual(validator.target_temp, 2.725)
        self.assertFalse(validator.allow_vacuum_birefringence)
        self.assertFalse(validator.allow_parity_violation)
        
    def test_cmb_validator_polarization_warnings(self):
        """Test CMB validator polarization constraint warnings."""
        # Create validator that allows problematic polarization effects
        validator = CMBValidator(
            allow_vacuum_birefringence=True,
            allow_parity_violation=True
        )
        
        result = validator.validate(self.registry)
        
        # Should generate warnings about polarization constraints
        warning_messages = " ".join(result.warnings)
        self.assertIn("birefringence", warning_messages)
        self.assertIn("parity", warning_messages)
        
    def test_cmb_validator_with_real_config(self):
        """Test CMB validator with real QFD configuration."""
        # Load real configuration
        load_parameters_from_yaml("qfd_params/defaults.yaml", self.registry)
        
        # Create validator from config
        import yaml
        with open("qfd_params/defaults.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        validator = CMBValidator.from_config(config)
        
        # Should be applicable
        self.assertTrue(validator.is_applicable(self.registry))
        
        # Run validation
        result = validator.validate(self.registry)
        
        # Should complete without errors (may have warnings for unset parameters)
        self.assertIn(result.status.value, ["valid", "warning", "invalid"])
        self.assertGreater(result.parameters_checked, 0)


if __name__ == "__main__":
    unittest.main()
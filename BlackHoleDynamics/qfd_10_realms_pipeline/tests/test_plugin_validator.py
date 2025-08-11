"""
Tests for the plugin validator integration.
"""

import unittest
import tempfile
import os

from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, Constraint, ConstraintType
)
from coupling_constants.plugins.plugin_manager import PluginManager
from coupling_constants.plugins.constraint_plugins.example_plugins import (
    PhotonMassConstraintPlugin, VacuumStabilityPlugin
)
from coupling_constants.validation.plugin_validator import (
    PluginValidator, PluginValidatorFactory
)
from coupling_constants.validation.base_validator import (
    CompositeValidator, ValidationStatus
)


class TestPluginValidator(unittest.TestCase):
    """Test plugin validator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        self.plugin_manager = PluginManager()
        
        # Add test parameters
        self.registry.update_parameter("m_gamma", 1e-20, "test_realm")
        self.registry.update_parameter("n_vac", 1.0, "test_realm")
        self.registry.update_parameter("k_J", 1e-15, "test_realm")
        
        # Register test plugins
        self.photon_plugin = PhotonMassConstraintPlugin()
        self.vacuum_plugin = VacuumStabilityPlugin()
        
        self.plugin_manager.register_plugin(self.photon_plugin)
        self.plugin_manager.register_plugin(self.vacuum_plugin)
        
        self.plugin_validator = PluginValidator(self.plugin_manager)
    
    def test_plugin_validator_initialization(self):
        """Test plugin validator initialization."""
        self.assertEqual(self.plugin_validator.name, "Plugin Validator")
        self.assertEqual(self.plugin_validator.conflict_resolution_strategy, "priority")
        self.assertTrue(self.plugin_validator.enabled)
    
    def test_set_conflict_resolution_strategy(self):
        """Test setting conflict resolution strategy."""
        self.plugin_validator.set_conflict_resolution_strategy("disable_lower")
        self.assertEqual(self.plugin_validator.conflict_resolution_strategy, "disable_lower")
    
    def test_is_applicable(self):
        """Test applicability check."""
        # Should be applicable with active plugins
        self.assertTrue(self.plugin_validator.is_applicable(self.registry))
        
        # Should not be applicable when disabled
        self.plugin_validator.enabled = False
        self.assertFalse(self.plugin_validator.is_applicable(self.registry))
        
        # Re-enable for other tests
        self.plugin_validator.enabled = True
    
    def test_get_required_parameters(self):
        """Test getting required parameters from all plugins."""
        required_params = self.plugin_validator.get_required_parameters()
        
        # Should include parameters from both plugins
        self.assertIn("m_gamma", required_params)
        self.assertIn("n_vac", required_params)
        self.assertIn("k_J", required_params)
    
    def test_validate_with_valid_parameters(self):
        """Test validation with valid parameters."""
        result = self.plugin_validator.validate(self.registry)
        
        self.assertTrue(result.is_valid())
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertEqual(len(result.violations), 0)
        self.assertGreater(result.execution_time_ms, 0)
        
        # Check metadata
        self.assertIn("total_plugins", result.metadata)
        self.assertIn("plugin_violations", result.metadata)
        self.assertIn("plugin_warnings", result.metadata)
        self.assertEqual(result.metadata["total_plugins"], 2)
    
    def test_validate_with_invalid_parameters(self):
        """Test validation with invalid parameters."""
        # Set invalid photon mass
        self.registry.update_parameter("m_gamma", 1e-15, "test_realm")
        
        result = self.plugin_validator.validate(self.registry)
        
        self.assertFalse(result.is_valid())
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertGreater(len(result.violations), 0)
        
        # Check that violation is from photon mass plugin
        violation_types = [v.violation_type for v in result.violations]
        self.assertIn("exceeds_limit", violation_types)
    
    def test_validate_with_no_plugins(self):
        """Test validation with no active plugins."""
        empty_plugin_manager = PluginManager()
        empty_validator = PluginValidator(empty_plugin_manager)
        
        result = empty_validator.validate(self.registry)
        
        self.assertTrue(result.is_valid())
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertIn("No active plugins to validate", result.info_messages)
    
    def test_validate_with_plugin_conflicts(self):
        """Test validation with plugin conflicts."""
        # Create conflicting parameter values
        self.registry.update_parameter("m_gamma", 1e-15, "test_realm")  # Invalid for photon plugin
        self.registry.update_parameter("n_vac", 1.1, "test_realm")     # Invalid for vacuum plugin
        
        result = self.plugin_validator.validate(self.registry)
        
        self.assertFalse(result.is_valid())
        self.assertGreater(len(result.violations), 0)
        
        # Should have warnings about conflicts if any are detected
        if result.warnings:
            conflict_warnings = [w for w in result.warnings if "conflict" in w.lower()]
            # May or may not have conflicts depending on parameter dependencies
    
    def test_get_description(self):
        """Test getting validator description."""
        description = self.plugin_validator.get_description()
        
        self.assertIn("Plugin Validator", description)
        self.assertIn("2/2", description)  # 2 active out of 2 total
        self.assertIn("priority", description)
    
    def test_get_plugin_summary(self):
        """Test getting plugin summary."""
        summary = self.plugin_validator.get_plugin_summary()
        
        self.assertEqual(summary["total_plugins"], 2)
        self.assertEqual(summary["active_plugins"], 2)
        self.assertEqual(summary["inactive_plugins"], 0)
        
        self.assertIn("plugins_by_priority", summary)
        self.assertIn("plugin_details", summary)
        
        # Check plugin details
        self.assertIn("photon_mass_constraint", summary["plugin_details"])
        self.assertIn("vacuum_stability", summary["plugin_details"])
        
        photon_details = summary["plugin_details"]["photon_mass_constraint"]
        self.assertEqual(photon_details["priority"], "HIGH")
        self.assertTrue(photon_details["active"])


class TestPluginValidatorFactory(unittest.TestCase):
    """Test plugin validator factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin_manager = PluginManager()
        
        # Register a test plugin
        photon_plugin = PhotonMassConstraintPlugin()
        self.plugin_manager.register_plugin(photon_plugin)
    
    def test_create_default_plugin_validator(self):
        """Test creating default plugin validator."""
        validator = PluginValidatorFactory.create_default_plugin_validator(self.plugin_manager)
        
        self.assertIsInstance(validator, PluginValidator)
        self.assertEqual(validator.name, "Default Plugin Validator")
        self.assertEqual(validator.conflict_resolution_strategy, "priority")
    
    def test_create_strict_plugin_validator(self):
        """Test creating strict plugin validator."""
        validator = PluginValidatorFactory.create_strict_plugin_validator(self.plugin_manager)
        
        self.assertIsInstance(validator, PluginValidator)
        self.assertEqual(validator.name, "Strict Plugin Validator")
        self.assertEqual(validator.conflict_resolution_strategy, "disable_lower")
    
    def test_create_permissive_plugin_validator(self):
        """Test creating permissive plugin validator."""
        validator = PluginValidatorFactory.create_permissive_plugin_validator(self.plugin_manager)
        
        self.assertIsInstance(validator, PluginValidator)
        self.assertEqual(validator.name, "Permissive Plugin Validator")
        self.assertEqual(validator.conflict_resolution_strategy, "user_choice")


class TestPluginValidatorIntegration(unittest.TestCase):
    """Test plugin validator integration with composite validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        self.plugin_manager = PluginManager()
        
        # Add test parameters
        self.registry.update_parameter("m_gamma", 1e-20, "test_realm")
        self.registry.update_parameter("n_vac", 1.0, "test_realm")
        
        # Register plugins
        photon_plugin = PhotonMassConstraintPlugin()
        vacuum_plugin = VacuumStabilityPlugin()
        
        self.plugin_manager.register_plugin(photon_plugin)
        self.plugin_manager.register_plugin(vacuum_plugin)
        
        # Create composite validator
        self.composite_validator = CompositeValidator("Test Composite Validator")
        
        # Add plugin validator
        plugin_validator = PluginValidator(self.plugin_manager)
        self.composite_validator.add_validator(plugin_validator)
    
    def test_composite_validation_with_plugins(self):
        """Test composite validation including plugins."""
        report = self.composite_validator.validate_all(self.registry)
        
        self.assertEqual(report.overall_status, ValidationStatus.VALID)
        self.assertEqual(len(report.validator_results), 1)  # Only plugin validator
        
        # Check plugin validator result
        plugin_result = report.validator_results[0]
        self.assertEqual(plugin_result.validator_name, "Plugin Validator")
        self.assertTrue(plugin_result.is_valid())
    
    def test_composite_validation_with_invalid_plugins(self):
        """Test composite validation with invalid plugin results."""
        # Set invalid parameter
        self.registry.update_parameter("m_gamma", 1e-15, "test_realm")
        
        report = self.composite_validator.validate_all(self.registry)
        
        self.assertEqual(report.overall_status, ValidationStatus.INVALID)
        self.assertGreater(report.total_violations, 0)
        
        # Check that violations come from plugins
        plugin_result = report.validator_results[0]
        self.assertFalse(plugin_result.is_valid())
        self.assertGreater(len(plugin_result.violations), 0)
    
    def test_multiple_validators_with_plugins(self):
        """Test multiple validators including plugins."""
        # Create a simple mock validator
        class MockValidator:
            def __init__(self, name):
                self.name = name
                self.enabled = True
            
            def is_applicable(self, registry):
                return True
            
            def validate(self, registry):
                from coupling_constants.validation.base_validator import ValidationResult, ValidationStatus
                result = ValidationResult(
                    validator_name=self.name,
                    status=ValidationStatus.VALID
                )
                result.add_info("Mock validation passed")
                return result
        
        # Add mock validator
        mock_validator = MockValidator("Mock Validator")
        self.composite_validator.add_validator(mock_validator)
        
        report = self.composite_validator.validate_all(self.registry)
        
        self.assertEqual(len(report.validator_results), 2)  # Plugin + Mock
        self.assertEqual(report.overall_status, ValidationStatus.VALID)
        
        # Check both validators ran
        validator_names = [r.validator_name for r in report.validator_results]
        self.assertIn("Plugin Validator", validator_names)
        self.assertIn("Mock Validator", validator_names)


if __name__ == "__main__":
    unittest.main()
"""
Tests for the plugin system functionality.
"""

import unittest
import tempfile
import os
import json
from typing import List

from coupling_constants.plugins.plugin_manager import (
    PluginManager, ConstraintPlugin, PluginInfo, PluginPriority
)
from coupling_constants.plugins.constraint_plugins.example_plugins import (
    PhotonMassConstraintPlugin, VacuumStabilityPlugin, CosmologicalConstantPlugin
)
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, Constraint, ConstraintType
)
from coupling_constants.validation.base_validator import ValidationResult, ValidationStatus, ValidationViolation


class TestConstraintPlugin(ConstraintPlugin):
    """Simple test plugin for unit testing."""
    
    def __init__(self, name: str = "test_plugin", priority: PluginPriority = PluginPriority.NORMAL):
        self.name = name
        self.priority = priority
        self.initialized = False
        self.cleaned_up = False
    
    @property
    def plugin_info(self) -> PluginInfo:
        return PluginInfo(
            name=self.name,
            version="1.0.0",
            author="Test Author",
            description="Test plugin for unit testing",
            priority=self.priority,
            dependencies=[],
            active=True
        )
    
    def get_parameter_dependencies(self) -> List[str]:
        return ["test_param"]
    
    def validate_constraint(self, registry: ParameterRegistry) -> ValidationResult:
        param = registry.get_parameter("test_param")
        if not param or param.value is None:
            result = ValidationResult(
                validator_name=self.name,
                status=ValidationStatus.INVALID
            )
            result.add_violation(ValidationViolation(
                parameter_name="test_param",
                constraint_realm=self.name,
                violation_type="missing_parameter",
                message="test_param not found"
            ))
            return result
        
        # Simple constraint: value must be positive
        if param.value > 0:
            result = ValidationResult(
                validator_name=self.name,
                status=ValidationStatus.VALID
            )
            result.add_info("test_param is valid")
            result.metadata = {"value": param.value}
            return result
        else:
            result = ValidationResult(
                validator_name=self.name,
                status=ValidationStatus.INVALID
            )
            result.add_violation(ValidationViolation(
                parameter_name="test_param",
                constraint_realm=self.name,
                violation_type="negative_value",
                actual_value=param.value,
                message="test_param must be positive"
            ))
            result.metadata = {"value": param.value}
            return result
    
    def get_constraint_description(self) -> str:
        return "Test constraint: parameter must be positive"
    
    def initialize(self, config=None):
        self.initialized = True
    
    def cleanup(self):
        self.cleaned_up = True


class TestPluginManager(unittest.TestCase):
    """Test plugin manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin_manager = PluginManager()
        self.registry = ParameterRegistry()
        
        # Add test parameter
        self.registry.update_parameter("test_param", 1.0, "test_realm")
    
    def test_plugin_manager_initialization(self):
        """Test plugin manager initialization."""
        self.assertIsInstance(self.plugin_manager.plugins, dict)
        self.assertIsInstance(self.plugin_manager.plugin_priorities, dict)
        self.assertIsInstance(self.plugin_manager.plugin_dependencies, dict)
        self.assertEqual(len(self.plugin_manager.plugins), 0)
    
    def test_register_plugin(self):
        """Test plugin registration."""
        plugin = TestConstraintPlugin("test1")
        
        # Test successful registration
        result = self.plugin_manager.register_plugin(plugin)
        self.assertTrue(result)
        self.assertIn("test1", self.plugin_manager.plugins)
        self.assertTrue(plugin.initialized)
        
        # Test duplicate registration
        duplicate_plugin = TestConstraintPlugin("test1")
        result = self.plugin_manager.register_plugin(duplicate_plugin)
        self.assertFalse(result)  # Should fail due to duplicate name
    
    def test_unregister_plugin(self):
        """Test plugin unregistration."""
        plugin = TestConstraintPlugin("test1")
        self.plugin_manager.register_plugin(plugin)
        
        # Test successful unregistration
        result = self.plugin_manager.unregister_plugin("test1")
        self.assertTrue(result)
        self.assertNotIn("test1", self.plugin_manager.plugins)
        self.assertTrue(plugin.cleaned_up)
        
        # Test unregistering non-existent plugin
        result = self.plugin_manager.unregister_plugin("nonexistent")
        self.assertFalse(result)
    
    def test_get_registered_plugins(self):
        """Test getting registered plugin information."""
        plugin1 = TestConstraintPlugin("test1", PluginPriority.HIGH)
        plugin2 = TestConstraintPlugin("test2", PluginPriority.LOW)
        
        self.plugin_manager.register_plugin(plugin1)
        self.plugin_manager.register_plugin(plugin2)
        
        plugins_info = self.plugin_manager.get_registered_plugins()
        self.assertEqual(len(plugins_info), 2)
        self.assertIn("test1", plugins_info)
        self.assertIn("test2", plugins_info)
        self.assertEqual(plugins_info["test1"].priority, PluginPriority.HIGH)
        self.assertEqual(plugins_info["test2"].priority, PluginPriority.LOW)
    
    def test_get_active_plugins(self):
        """Test getting active plugins."""
        plugin1 = TestConstraintPlugin("test1")
        plugin2 = TestConstraintPlugin("test2")
        
        self.plugin_manager.register_plugin(plugin1)
        self.plugin_manager.register_plugin(plugin2)
        
        # Both should be active initially
        active_plugins = self.plugin_manager.get_active_plugins()
        self.assertEqual(len(active_plugins), 2)
        
        # Deactivate one plugin by modifying the plugin directly
        # Since plugin_info is a property, we need to modify the plugin's internal state
        plugin1.plugin_info.active = False
        
        # For this test, let's create a modified version that tracks active state
        class ModifiableTestPlugin(TestConstraintPlugin):
            def __init__(self, name, priority=PluginPriority.NORMAL):
                super().__init__(name, priority)
                self._active = True
            
            @property
            def plugin_info(self):
                info = super().plugin_info
                info.active = self._active
                return info
        
        # Replace with modifiable plugins
        self.plugin_manager.plugins.clear()
        self.plugin_manager.plugin_priorities.clear()
        self.plugin_manager.plugin_dependencies.clear()
        
        plugin1_mod = ModifiableTestPlugin("test1")
        plugin2_mod = ModifiableTestPlugin("test2")
        
        self.plugin_manager.register_plugin(plugin1_mod)
        self.plugin_manager.register_plugin(plugin2_mod)
        
        # Both should be active initially
        active_plugins = self.plugin_manager.get_active_plugins()
        self.assertEqual(len(active_plugins), 2)
        
        # Deactivate one plugin
        plugin1_mod._active = False
        active_plugins = self.plugin_manager.get_active_plugins()
        self.assertEqual(len(active_plugins), 1)
        self.assertIn("test2", active_plugins)
    
    def test_validate_all_plugin_constraints(self):
        """Test validating all plugin constraints."""
        plugin1 = TestConstraintPlugin("test1", PluginPriority.HIGH)
        plugin2 = TestConstraintPlugin("test2", PluginPriority.LOW)
        
        self.plugin_manager.register_plugin(plugin1)
        self.plugin_manager.register_plugin(plugin2)
        
        # Test with valid parameter
        results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.is_valid() for result in results))
        
        # Test with invalid parameter
        self.registry.update_parameter("test_param", -1.0, "test_realm")
        results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not result.is_valid() for result in results))
    
    def test_plugin_priority_ordering(self):
        """Test that plugins are executed in priority order."""
        plugin_high = TestConstraintPlugin("high_priority", PluginPriority.HIGH)
        plugin_low = TestConstraintPlugin("low_priority", PluginPriority.LOW)
        plugin_critical = TestConstraintPlugin("critical_priority", PluginPriority.CRITICAL)
        
        # Register in random order
        self.plugin_manager.register_plugin(plugin_low)
        self.plugin_manager.register_plugin(plugin_critical)
        self.plugin_manager.register_plugin(plugin_high)
        
        results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        
        # Should be ordered by priority: CRITICAL, HIGH, LOW
        expected_order = ["plugin_critical_priority", "plugin_high_priority", "plugin_low_priority"]
        actual_order = [result.validator_name for result in results]
        self.assertEqual(actual_order, expected_order)
    
    def test_export_plugin_info(self):
        """Test exporting plugin information."""
        plugin = TestConstraintPlugin("test1")
        self.plugin_manager.register_plugin(plugin)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "plugin_info.json")
            self.plugin_manager.export_plugin_info(output_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Check content
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn("test1", data)
            plugin_data = data["test1"]
            self.assertEqual(plugin_data["name"], "test1")
            self.assertEqual(plugin_data["version"], "1.0.0")
            self.assertEqual(plugin_data["priority"], "NORMAL")
            self.assertEqual(plugin_data["parameter_dependencies"], ["test_param"])


class TestExamplePlugins(unittest.TestCase):
    """Test the example constraint plugins."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        self.plugin_manager = PluginManager()
    
    def test_photon_mass_constraint_plugin(self):
        """Test photon mass constraint plugin."""
        plugin = PhotonMassConstraintPlugin()
        
        # Test plugin info
        info = plugin.plugin_info
        self.assertEqual(info.name, "photon_mass_constraint")
        self.assertEqual(info.priority, PluginPriority.HIGH)
        
        # Test parameter dependencies
        deps = plugin.get_parameter_dependencies()
        self.assertIn("m_gamma", deps)
        
        # Test constraint description
        description = plugin.get_constraint_description()
        self.assertIn("photon mass", description.lower())
        
        # Test validation with valid photon mass
        self.registry.update_parameter("m_gamma", 1e-20, "test_realm")  # Well below limit
        result = plugin.validate_constraint(self.registry)
        self.assertTrue(result.is_valid())
        
        # Test validation with invalid photon mass
        self.registry.update_parameter("m_gamma", 1e-15, "test_realm")  # Above limit
        result = plugin.validate_constraint(self.registry)
        self.assertFalse(result.is_valid())
        self.assertGreater(len(result.violations), 0)
        self.assertIn("exceeds experimental limit", result.violations[0].message)
    
    def test_vacuum_stability_plugin(self):
        """Test vacuum stability plugin."""
        plugin = VacuumStabilityPlugin()
        
        # Test plugin info
        info = plugin.plugin_info
        self.assertEqual(info.name, "vacuum_stability")
        self.assertEqual(info.priority, PluginPriority.CRITICAL)
        
        # Test parameter dependencies
        deps = plugin.get_parameter_dependencies()
        self.assertIn("n_vac", deps)
        
        # Test with stable vacuum
        self.registry.update_parameter("n_vac", 1.0, "test_realm")
        self.registry.update_parameter("k_J", 1e-15, "test_realm")
        result = plugin.validate_constraint(self.registry)
        self.assertTrue(result.is_valid())
        
        # Test with unstable vacuum
        self.registry.update_parameter("n_vac", 1.1, "test_realm")  # Too far from 1
        result = plugin.validate_constraint(self.registry)
        self.assertFalse(result.is_valid())
        self.assertGreater(len(result.violations), 0)
        self.assertIn("deviates from unity", result.violations[0].message)
    
    def test_cosmological_constant_plugin(self):
        """Test cosmological constant plugin."""
        plugin = CosmologicalConstantPlugin()
        
        # Test plugin info
        info = plugin.plugin_info
        self.assertEqual(info.name, "cosmological_constant")
        self.assertEqual(info.priority, PluginPriority.NORMAL)
        
        # Test parameter dependencies
        deps = plugin.get_parameter_dependencies()
        self.assertIn("Lambda", deps)
        
        # Test with correct cosmological constant
        self.registry.update_parameter("Lambda", 1.1e-52, "test_realm")
        result = plugin.validate_constraint(self.registry)
        self.assertTrue(result.is_valid())
        
        # Test with incorrect cosmological constant
        self.registry.update_parameter("Lambda", 1e-50, "test_realm")  # Too large
        result = plugin.validate_constraint(self.registry)
        self.assertFalse(result.is_valid())
        self.assertGreater(len(result.violations), 0)
        self.assertIn("inconsistent with observations", result.violations[0].message)


class TestPluginConflictResolution(unittest.TestCase):
    """Test plugin conflict detection and resolution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin_manager = PluginManager()
        self.registry = ParameterRegistry()
        
        # Add test parameter
        self.registry.update_parameter("test_param", 0.5, "test_realm")
    
    def test_plugin_conflicts(self):
        """Test plugin conflict detection."""
        # Create conflicting plugins
        class PositivePlugin(TestConstraintPlugin):
            def validate_constraint(self, registry):
                param = registry.get_parameter("test_param")
                result = ValidationResult(
                    validator_name="positive_plugin",
                    status=ValidationStatus.VALID if param.value > 0 else ValidationStatus.INVALID
                )
                if param.value <= 0:
                    result.add_violation(ValidationViolation(
                        parameter_name="test_param",
                        constraint_realm="positive_plugin",
                        violation_type="not_positive",
                        message="Must be positive"
                    ))
                return result
        
        class NegativePlugin(TestConstraintPlugin):
            def validate_constraint(self, registry):
                param = registry.get_parameter("test_param")
                result = ValidationResult(
                    validator_name="negative_plugin",
                    status=ValidationStatus.VALID if param.value < 0 else ValidationStatus.INVALID
                )
                if param.value >= 0:
                    result.add_violation(ValidationViolation(
                        parameter_name="test_param",
                        constraint_realm="negative_plugin",
                        violation_type="not_negative",
                        message="Must be negative"
                    ))
                return result
        
        positive_plugin = PositivePlugin("positive_plugin", PluginPriority.HIGH)
        negative_plugin = NegativePlugin("negative_plugin", PluginPriority.LOW)
        
        self.plugin_manager.register_plugin(positive_plugin)
        self.plugin_manager.register_plugin(negative_plugin)
        
        # Get conflicts
        conflicts = self.plugin_manager.get_plugin_conflicts(self.registry)
        
        # Should detect conflict on test_param
        self.assertGreater(len(conflicts), 0)
        conflict = conflicts[0]
        self.assertEqual(conflict["parameter"], "test_param")
        self.assertEqual(conflict["conflict_type"], "validation_conflict")
    
    def test_conflict_resolution(self):
        """Test plugin conflict resolution."""
        # Create conflicting plugins with different priorities
        plugin_high = TestConstraintPlugin("high_priority", PluginPriority.HIGH)
        plugin_low = TestConstraintPlugin("low_priority", PluginPriority.LOW)
        
        self.plugin_manager.register_plugin(plugin_high)
        self.plugin_manager.register_plugin(plugin_low)
        
        # Set parameter to invalid value to create conflict
        self.registry.update_parameter("test_param", -1.0, "test_realm")
        
        conflicts = self.plugin_manager.get_plugin_conflicts(self.registry)
        
        if conflicts:
            # Resolve conflicts using priority strategy
            resolution = self.plugin_manager.resolve_plugin_conflicts(conflicts, "priority")
            
            # Check that lower priority plugin was disabled
            self.assertIn("low_priority", resolution.get("disabled_plugins", []))


class TestPluginIntegration(unittest.TestCase):
    """Test plugin system integration with the main framework."""
    
    def test_plugin_integration_with_validation_framework(self):
        """Test that plugins integrate properly with the validation framework."""
        plugin_manager = PluginManager()
        registry = ParameterRegistry()
        
        # Register example plugins
        photon_plugin = PhotonMassConstraintPlugin()
        vacuum_plugin = VacuumStabilityPlugin()
        
        plugin_manager.register_plugin(photon_plugin)
        plugin_manager.register_plugin(vacuum_plugin)
        
        # Add required parameters
        registry.update_parameter("m_gamma", 1e-20, "test_realm")
        registry.update_parameter("n_vac", 1.0, "test_realm")
        
        # Validate all plugin constraints
        results = plugin_manager.validate_all_plugin_constraints(registry)
        
        # Should have results from both plugins
        self.assertEqual(len(results), 2)
        validator_names = [result.validator_name for result in results]
        self.assertIn("plugin_photon_mass_constraint", validator_names)
        self.assertIn("plugin_vacuum_stability", validator_names)
        
        # All should be valid with good parameters
        self.assertTrue(all(result.is_valid() for result in results))


if __name__ == "__main__":
    unittest.main()
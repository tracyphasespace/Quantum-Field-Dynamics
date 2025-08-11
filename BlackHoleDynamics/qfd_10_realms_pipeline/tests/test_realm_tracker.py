"""
Tests for realm tracker functionality.
"""

import unittest
import tempfile
import os
import json
from coupling_constants.registry.parameter_registry import ParameterRegistry
from coupling_constants.analysis.realm_tracker import RealmTracker, RealmStatus
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.basic_validators import BoundsValidator
from coupling_constants.config.yaml_loader import load_parameters_from_yaml


class TestRealmTracker(unittest.TestCase):
    """Test realm tracker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
        # Add some test parameters
        self.registry.update_parameter("param1", 1.0, "initial")
        self.registry.update_parameter("param2", 2.0, "initial")
        self.registry.update_parameter("param3", 0.5, "initial")
        
        # Create validator
        validator = CompositeValidator()
        validator.add_validator(BoundsValidator())
        
        self.tracker = RealmTracker(self.registry, validator)
        
        # Register test realm functions
        def test_realm1(registry):
            # Modify param1
            registry.update_parameter("param1", 1.1, "test_realm1", "Test modification")
            return {
                "status": "ok",
                "fixed": {"param1": 1.1},
                "notes": ["Modified param1"]
            }
        
        def test_realm2(registry):
            # Modify param2 based on param1
            param1 = registry.get_parameter("param1")
            if param1 and param1.value:
                new_value = param1.value * 2
                registry.update_parameter("param2", new_value, "test_realm2", "Based on param1")
            return {
                "status": "ok",
                "fixed": {"param2": new_value if 'new_value' in locals() else 2.0},
                "notes": ["Modified param2 based on param1"]
            }
        
        def failing_realm(registry):
            raise ValueError("Test failure")
        
        self.tracker.register_realm("test_realm1", test_realm1)
        self.tracker.register_realm("test_realm2", test_realm2, dependencies=["test_realm1"])
        self.tracker.register_realm("failing_realm", failing_realm)
        
    def test_register_realm(self):
        """Test realm registration."""
        def new_realm(registry):
            return {"status": "ok"}
        
        self.tracker.register_realm("new_realm", new_realm, ["test_realm1"])
        
        self.assertIn("new_realm", self.tracker.realm_functions)
        self.assertEqual(self.tracker.realm_dependencies["new_realm"], ["test_realm1"])
        
    def test_execute_single_realm(self):
        """Test executing a single realm."""
        result = self.tracker.execute_realm("test_realm1")
        
        self.assertEqual(result.realm_name, "test_realm1")
        self.assertEqual(result.status, RealmStatus.COMPLETED)
        self.assertIn("param1", result.parameters_modified)
        self.assertGreater(result.execution_time_ms, 0)
        self.assertEqual(result.notes, ["Modified param1"])
        
        # Check parameter was actually modified
        param1 = self.registry.get_parameter("param1")
        self.assertEqual(param1.value, 1.1)
        
    def test_execute_realm_with_dependencies(self):
        """Test executing realm with dependencies."""
        # Try to execute realm2 without executing realm1 first
        result = self.tracker.execute_realm("test_realm2")
        
        self.assertEqual(result.status, RealmStatus.FAILED)
        self.assertIn("Missing dependencies", result.error_message)
        
        # Execute realm1 first, then realm2
        self.tracker.execute_realm("test_realm1")
        result2 = self.tracker.execute_realm("test_realm2")
        
        self.assertEqual(result2.status, RealmStatus.COMPLETED)
        self.assertIn("param2", result2.parameters_modified)
        
    def test_execute_failing_realm(self):
        """Test executing a realm that fails."""
        result = self.tracker.execute_realm("failing_realm")
        
        self.assertEqual(result.status, RealmStatus.FAILED)
        self.assertIn("Test failure", result.error_message)
        self.assertEqual(result.parameters_modified, [])
        
    def test_execute_nonexistent_realm(self):
        """Test executing a realm that doesn't exist."""
        result = self.tracker.execute_realm("nonexistent")
        
        self.assertEqual(result.status, RealmStatus.FAILED)
        self.assertIn("not registered", result.error_message)
        
    def test_convergence_threshold_setting(self):
        """Test setting convergence thresholds."""
        self.tracker.set_convergence_threshold("param1", 1e-8)
        
        self.assertEqual(self.tracker.convergence_thresholds["param1"], 1e-8)
        
    def test_execution_summary(self):
        """Test execution summary generation."""
        # Execute some realms
        self.tracker.execute_realm("test_realm1")
        self.tracker.execute_realm("test_realm2")
        self.tracker.execute_realm("failing_realm")
        
        summary = self.tracker.get_execution_summary()
        
        self.assertEqual(summary["total_executions"], 3)
        self.assertEqual(summary["successful_executions"], 2)
        self.assertEqual(summary["failed_executions"], 1)
        self.assertIn("test_realm1", summary["realms_executed"])
        self.assertIn("test_realm2", summary["realms_executed"])
        self.assertIn("failing_realm", summary["failed_realms"])
        
    def test_export_execution_log(self):
        """Test exporting execution log."""
        # Execute some realms
        self.tracker.execute_realm("test_realm1")
        self.tracker.execute_realm("test_realm2")
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        try:
            self.tracker.export_execution_log(json_file)
            
            # Check file exists and is valid JSON
            self.assertTrue(os.path.exists(json_file))
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check structure
            self.assertIn('execution_summary', data)
            self.assertIn('execution_history', data)
            self.assertIn('realm_dependencies', data)
            
            # Check content
            self.assertEqual(len(data['execution_history']), 2)
            self.assertEqual(data['execution_summary']['successful_executions'], 2)
            
        finally:
            if os.path.exists(json_file):
                os.unlink(json_file)
                
    def test_reset_execution_history(self):
        """Test resetting execution history."""
        # Execute some realms
        self.tracker.execute_realm("test_realm1")
        self.assertEqual(len(self.tracker.execution_history), 1)
        
        # Reset
        self.tracker.reset_execution_history()
        self.assertEqual(len(self.tracker.execution_history), 0)
        self.assertIsNone(self.tracker.current_realm)
        self.assertFalse(self.tracker.sequence_running)


class TestRealmSequenceExecution(unittest.TestCase):
    """Test realm sequence execution with convergence."""
    
    def setUp(self):
        """Set up test fixtures for sequence testing."""
        self.registry = ParameterRegistry()
        
        # Add parameters that will converge
        self.registry.update_parameter("x", 1.0, "initial")
        self.registry.update_parameter("y", 2.0, "initial")
        
        validator = CompositeValidator()
        validator.add_validator(BoundsValidator())
        
        self.tracker = RealmTracker(self.registry, validator)
        
        # Register realms that will converge
        def realm_a(registry):
            x = registry.get_parameter("x")
            if x and x.value:
                # Move x towards 1.5
                new_x = x.value + 0.1 * (1.5 - x.value)
                registry.update_parameter("x", new_x, "realm_a")
            return {"status": "ok", "notes": ["Adjusted x"]}
        
        def realm_b(registry):
            y = registry.get_parameter("y")
            x = registry.get_parameter("x")
            if y and y.value and x and x.value:
                # Move y towards 2*x
                target = 2 * x.value
                new_y = y.value + 0.1 * (target - y.value)
                registry.update_parameter("y", new_y, "realm_b")
            return {"status": "ok", "notes": ["Adjusted y based on x"]}
        
        self.tracker.register_realm("realm_a", realm_a)
        self.tracker.register_realm("realm_b", realm_b)
        
        # Set tight convergence thresholds
        self.tracker.set_convergence_threshold("x", 1e-3)
        self.tracker.set_convergence_threshold("y", 1e-3)
        
    def test_realm_sequence_execution(self):
        """Test executing a complete realm sequence."""
        result = self.tracker.execute_realm_sequence(
            realm_order=["realm_a", "realm_b"],
            max_iterations=5,
            convergence_check=True
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(result.iterations_completed, 0)
        self.assertGreater(len(result.realms_executed), 0)
        self.assertIsNotNone(result.final_validation_report)
        
        # Check that parameters moved towards expected values
        x = self.registry.get_parameter("x")
        y = self.registry.get_parameter("y")
        
        # x should be closer to 1.5
        self.assertGreater(x.value, 1.0)
        self.assertLess(x.value, 1.5)
        
        # y should be closer to 2*x
        expected_y = 2 * x.value
        self.assertLess(abs(y.value - expected_y), 0.5)  # Should be getting closer
        
    def test_realm_sequence_with_no_registered_realms(self):
        """Test sequence execution with no registered realms."""
        empty_tracker = RealmTracker(self.registry)
        
        with self.assertRaises(ValueError):
            empty_tracker.execute_realm_sequence()
            
    def test_realm_sequence_early_termination(self):
        """Test sequence with early termination due to failure."""
        # Register a failing realm
        def failing_realm(registry):
            raise RuntimeError("Intentional failure")
        
        self.tracker.register_realm("failing_realm", failing_realm)
        
        result = self.tracker.execute_realm_sequence(
            realm_order=["realm_a", "failing_realm", "realm_b"],
            max_iterations=3
        )
        
        self.assertIsNotNone(result.early_termination_reason)
        self.assertIn("failed", result.early_termination_reason)


class TestRealmTrackerWithRealQFD(unittest.TestCase):
    """Test realm tracker with real QFD configuration."""
    
    def test_with_real_qfd_data(self):
        """Test realm tracker with real QFD configuration."""
        # Load real configuration
        registry = ParameterRegistry()
        load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
        
        # Set some parameter values
        registry.update_parameter("T_CMB_K", 2.725, "cmb_config")
        registry.update_parameter("k_J", 1e-12, "realm0_cmb")
        registry.update_parameter("PPN_gamma", 1.0, "realm3_scales")
        
        # Create tracker
        tracker = RealmTracker(registry)
        
        # Register a simple test realm
        def test_qfd_realm(registry):
            # Slightly adjust a parameter
            xi = registry.get_parameter("xi")
            if xi:
                registry.update_parameter("xi", 2.0, "test_qfd_realm", "Test adjustment")
            return {"status": "ok", "notes": ["Adjusted xi"]}
        
        tracker.register_realm("test_qfd_realm", test_qfd_realm)
        
        # Execute realm
        result = tracker.execute_realm("test_qfd_realm")
        
        self.assertEqual(result.status, RealmStatus.COMPLETED)
        self.assertIn("xi", result.parameters_modified)
        
        # Check summary
        summary = tracker.get_execution_summary()
        self.assertEqual(summary["successful_executions"], 1)
        self.assertGreater(summary["total_execution_time_ms"], 0)


if __name__ == "__main__":
    unittest.main()
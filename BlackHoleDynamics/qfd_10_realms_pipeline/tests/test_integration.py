"""
Tests for the integration system.
"""

import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock

from coupling_constants.integration.realm_integration import (
    RealmIntegrationManager, RealmExecutionHook
)
from coupling_constants.integration.workflow_scripts import (
    run_integrated_analysis, run_realm_sequence_with_analysis,
    validate_realm_integration, get_default_realm_sequence
)
from coupling_constants.registry.parameter_registry import Constraint, ConstraintType
from coupling_constants.analysis.realm_tracker import RealmStatus


class TestRealmIntegrationManager(unittest.TestCase):
    """Test the RealmIntegrationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create test configuration
        with open(self.config_file, 'w') as f:
            f.write("""
parameters:
  k_J:
    min: 0.0
    max: 1e-6
    note: "Test parameter"
  xi:
    min: 0.0
    max: 100.0
    note: "Another test parameter"
  T_CMB_K:
    min: 2.7
    max: 2.8
    note: "CMB temperature"
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test integration manager initialization."""
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        self.assertIsNotNone(manager.registry)
        self.assertIsNotNone(manager.realm_tracker)
        self.assertIsNotNone(manager.plugin_manager)
        self.assertIsNotNone(manager.validator)
        self.assertEqual(len(manager.registry.get_all_parameters()), 3)
    
    def test_realm_hook_registration(self):
        """Test realm hook registration."""
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        hook = RealmExecutionHook(
            realm_name="test_realm",
            validation_required=True
        )
        
        manager.register_realm_hook(hook)
        self.assertIn("test_realm", manager.realm_hooks)
        self.assertEqual(manager.realm_hooks["test_realm"], hook)
    
    def test_plugin_registration(self):
        """Test plugin registration."""
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        # Test successful registration
        success = manager.register_plugin("vacuum_stability")
        self.assertTrue(success)
        
        # Test unknown plugin
        success = manager.register_plugin("unknown_plugin")
        self.assertFalse(success)
    
    @patch('importlib.import_module')
    def test_realm_execution_with_integration(self, mock_import):
        """Test realm execution with integration."""
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        # Mock realm module
        mock_realm = MagicMock()
        mock_realm.run.return_value = {
            'fixed': {'T_CMB_K': 2.725},
            'narrowed': {'k_J': {'min': 0.0, 'max': 1e-12}},
            'notes': ['Test realm execution']
        }
        mock_import.return_value = mock_realm
        
        # Execute realm
        result = manager.execute_realm_with_integration(
            "test_realm", "test.module", {"test_param": "test_value"}
        )
        
        # Check result
        self.assertEqual(result.realm_name, "test_realm")
        self.assertEqual(result.status.value, "completed")
        self.assertGreater(result.execution_time_ms, 0)
        self.assertIn("T_CMB_K", result.parameters_modified)
        
        # Check parameter was updated
        param = manager.registry.get_parameter("T_CMB_K")
        self.assertEqual(param.value, 2.725)
    
    def test_update_registry_from_realm_result(self):
        """Test updating registry from realm results."""
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        realm_result = {
            'fixed': {'T_CMB_K': 2.725},
            'narrowed': {
                'k_J': {'min': 0.0, 'max': 1e-12},
                'xi': {'target': 2.0, 'tolerance': 0.1}
            }
        }
        
        pre_execution_params = {}
        
        modified = manager._update_registry_from_realm_result(
            "test_realm", realm_result, pre_execution_params
        )
        
        # Check that T_CMB_K was modified
        self.assertIn("T_CMB_K", modified)
        
        # Check parameter value
        param = manager.registry.get_parameter("T_CMB_K")
        self.assertEqual(param.value, 2.725)
        
        # Check constraints were added
        k_j_param = manager.registry.get_parameter("k_J")
        self.assertGreater(len(k_j_param.constraints), 0)
        
        xi_param = manager.registry.get_parameter("xi")
        self.assertGreater(len(xi_param.constraints), 0)
    
    def test_execution_summary(self):
        """Test execution summary generation."""
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        summary = manager.get_execution_summary()
        
        self.assertIn('total_realms_executed', summary)
        self.assertIn('total_parameters', summary)
        self.assertIn('total_constraints', summary)
        self.assertEqual(summary['total_parameters'], 3)


class TestWorkflowScripts(unittest.TestCase):
    """Test workflow scripts."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create test configuration
        with open(self.config_file, 'w') as f:
            f.write("""
parameters:
  k_J:
    min: 0.0
    max: 1e-6
    note: "Test parameter"
  T_CMB_K:
    min: 2.7
    max: 2.8
    note: "CMB temperature"
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_default_realm_sequence(self):
        """Test getting default realm sequence."""
        sequence = get_default_realm_sequence()
        
        self.assertIsInstance(sequence, list)
        self.assertGreater(len(sequence), 0)
        
        # Check first realm is CMB
        self.assertEqual(sequence[0][0], 'realm0_cmb')
        self.assertEqual(sequence[0][1], 'realms.realm0_cmb')
    
    @patch('coupling_constants.integration.workflow_scripts.RealmIntegrationManager')
    def test_run_integrated_analysis(self, mock_manager_class):
        """Test run_integrated_analysis function."""
        # Mock manager
        mock_manager = MagicMock()
        mock_manager.execute_realm_sequence.return_value = []
        mock_manager.generate_analysis_report.return_value = "/test/report/path"
        mock_manager.get_execution_summary.return_value = {
            'total_realms_executed': 2,
            'successful_executions': 2
        }
        mock_manager_class.return_value = mock_manager
        
        # Run analysis
        report_path = run_integrated_analysis(
            config_path=self.config_file,
            output_dir=self.temp_dir,
            realm_sequence=[('test_realm1', 'test.module1'), ('test_realm2', 'test.module2')],
            enable_plugins=['vacuum_stability'],
            generate_visualizations=True,
            verbose=False
        )
        
        # Check that manager was created and methods were called
        mock_manager_class.assert_called_once()
        mock_manager.execute_realm_sequence.assert_called_once()
        mock_manager.generate_analysis_report.assert_called_once_with(include_visualizations=True)
        
        self.assertEqual(report_path, "/test/report/path")
    
    @patch('coupling_constants.integration.workflow_scripts.RealmIntegrationManager')
    def test_run_realm_sequence_with_analysis(self, mock_manager_class):
        """Test run_realm_sequence_with_analysis function."""
        # Mock manager
        mock_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.realm_name = "test_realm"
        mock_result.status.value = "completed"
        mock_result.execution_time_ms = 100.0
        mock_result.parameters_modified = ["test_param"]
        
        mock_manager.execute_realm_sequence.return_value = [mock_result]
        mock_manager.generate_analysis_report.return_value = "/test/report/path"
        mock_manager_class.return_value = mock_manager
        
        # Run analysis
        report_path = run_realm_sequence_with_analysis(
            realm_names=['test_realm'],
            config_path=self.config_file,
            output_dir=self.temp_dir,
            enable_plugins=['photon_mass'],
            verbose=False
        )
        
        # Check that manager was created and methods were called
        mock_manager_class.assert_called_once()
        mock_manager.execute_realm_sequence.assert_called_once()
        mock_manager.generate_analysis_report.assert_called_once()
        
        self.assertEqual(report_path, "/test/report/path")
    
    @patch('coupling_constants.integration.workflow_scripts.RealmIntegrationManager')
    def test_validate_realm_integration(self, mock_manager_class):
        """Test validate_realm_integration function."""
        # Mock manager
        mock_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.status.value = "completed"
        mock_result.execution_time_ms = 50.0
        mock_result.parameters_modified = []
        mock_result.constraints_added = 0
        mock_result.metadata = {}
        
        mock_manager.execute_realm_with_integration.return_value = mock_result
        mock_manager_class.return_value = mock_manager
        
        # Test successful validation
        success = validate_realm_integration(
            config_path=self.config_file,
            realm_name="test_realm",
            enable_plugins=["vacuum_stability"]
        )
        
        self.assertTrue(success)
        mock_manager.execute_realm_with_integration.assert_called_once_with(
            "test_realm", "realms.test_realm"
        )
        
        # Test failed validation
        mock_result.status.value = "failed"
        mock_result.metadata = {"error": "Test error"}
        
        success = validate_realm_integration(
            config_path=self.config_file,
            realm_name="test_realm"
        )
        
        self.assertFalse(success)


class TestIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create comprehensive test configuration
        with open(self.config_file, 'w') as f:
            f.write("""
parameters:
  k_J:
    min: 0.0
    max: 1e-6
    note: "Incoherent photon drag"
  xi:
    min: 0.0
    max: 100.0
    note: "Coupling parameter"
  T_CMB_K:
    min: 2.7
    max: 2.8
    note: "CMB temperature"
  n_vac:
    min: 0.999999
    max: 1.000001
    note: "Vacuum refractive index"
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_manager_workflow(self):
        """Test complete integration manager workflow."""
        # Initialize manager
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        # Register plugins
        manager.register_plugin("vacuum_stability")
        
        # Set up a simple hook
        def test_hook(registry, result):
            # Just add a note, don't try to update fixed parameters
            pass
        
        hook = RealmExecutionHook(
            realm_name="test_realm",
            post_execution_hook=test_hook
        )
        manager.register_realm_hook(hook)
        
        # Mock a simple realm execution
        with patch('importlib.import_module') as mock_import:
            mock_realm = MagicMock()
            mock_realm.run.return_value = {
                'fixed': {'T_CMB_K': 2.725},
                'notes': ['Test execution']
            }
            mock_import.return_value = mock_realm
            
            # Execute realm
            result = manager.execute_realm_with_integration(
                "test_realm", "test.module"
            )
            
            # Check execution was successful
            self.assertEqual(result.status.value, "completed")
            self.assertIn("T_CMB_K", result.parameters_modified)
        
        # Generate report
        report_path = manager.generate_analysis_report(include_visualizations=False)
        
        # Check report was generated
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(os.path.exists(os.path.join(report_path, "parameters.json")))
        self.assertTrue(os.path.exists(os.path.join(report_path, "README.md")))
        
        # Check execution summary
        summary = manager.get_execution_summary()
        self.assertEqual(summary['total_realms_executed'], 1)
        self.assertEqual(summary['successful_executions'], 1)


if __name__ == "__main__":
    unittest.main()
"""
Complete system integration tests.

This module contains end-to-end tests that verify the entire coupling constants
framework works together correctly, including all components and workflows.
"""

import unittest
import tempfile
import os
import json
import shutil
from pathlib import Path

from coupling_constants.registry.parameter_registry import ParameterRegistry
from coupling_constants.config.yaml_loader import load_parameters_from_yaml
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.ppn_validator import PPNValidator
from coupling_constants.validation.cmb_validator import CMBValidator
from coupling_constants.validation.basic_validators import BoundsValidator, FixedValueValidator, TargetValueValidator
from coupling_constants.analysis.dependency_mapper import DependencyMapper
from coupling_constants.analysis.sensitivity_analyzer import SensitivityAnalyzer
from coupling_constants.analysis.realm_tracker import RealmTracker
from coupling_constants.visualization.coupling_visualizer import CouplingVisualizer
from coupling_constants.visualization.export_manager import ExportManager
from coupling_constants.plugins.plugin_manager import PluginManager
from coupling_constants.plugins.constraint_plugins import VacuumStabilityPlugin, PhotonMassConstraintPlugin
from coupling_constants.integration.realm_integration import RealmIntegrationManager
from coupling_constants.integration.workflow_scripts import run_integrated_analysis
from coupling_constants.cli.main import cmd_validate, cmd_export, cmd_analyze


class TestCompleteSystemIntegration(unittest.TestCase):
    """End-to-end integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'qfd_config.yaml')
        
        # Create comprehensive test configuration
        with open(self.config_file, 'w') as f:
            f.write("""
parameters:
  k_J:
    min: 0.0
    max: 1e-6
    note: "Incoherent photon drag; must be ~0 locally"
  xi:
    min: 0.1
    max: 100.0
    note: "Coupling parameter"
  psi_s0:
    min: -5.0
    max: 5.0
    note: "Scalar field parameter"
  n_vac:
    min: 0.999999
    max: 1.000001
    note: "Vacuum refractive index"
  PPN_gamma:
    min: 0.99
    max: 1.01
    note: "PPN gamma parameter"
  PPN_beta:
    min: 0.99
    max: 1.01
    note: "PPN beta parameter"
  T_CMB_K:
    min: 2.7
    max: 2.8
    note: "CMB temperature"
  m_gamma:
    min: 0.0
    max: 1e-15
    note: "Photon mass"
  Lambda:
    min: 1e-53
    max: 1e-51
    note: "Cosmological constant"
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_workflow_integration(self):
        """Test complete workflow from configuration to analysis."""
        # 1. Load configuration
        registry = ParameterRegistry()
        load_parameters_from_yaml(self.config_file, registry)
        
        # Verify parameters loaded
        self.assertGreater(len(registry.get_all_parameters()), 0)
        
        # 2. Set some parameter values
        registry.update_parameter("k_J", 1e-12, "test", "Test value")
        registry.update_parameter("xi", 2.0, "test", "Test value")
        registry.update_parameter("n_vac", 1.0, "test", "Test value")
        registry.update_parameter("PPN_gamma", 1.0, "test", "Test value")
        registry.update_parameter("PPN_beta", 1.0, "test", "Test value")
        registry.update_parameter("T_CMB_K", 2.725, "test", "Test value")
        registry.update_parameter("m_gamma", 1e-20, "test", "Test value")
        registry.update_parameter("Lambda", 1.1e-52, "test", "Test value")
        
        # 3. Set up validation
        validator = CompositeValidator("Complete System Test")
        validator.add_validator(PPNValidator())
        validator.add_validator(CMBValidator())
        validator.add_validator(BoundsValidator())
        validator.add_validator(FixedValueValidator())
        validator.add_validator(TargetValueValidator())
        
        # 4. Run validation
        report = validator.validate_all(registry)
        self.assertIsNotNone(report)
        
        # 5. Set up plugins
        plugin_manager = PluginManager()
        plugin_manager.register_plugin(VacuumStabilityPlugin())
        plugin_manager.register_plugin(PhotonMassConstraintPlugin())
        
        # 6. Run plugin validation
        plugin_results = plugin_manager.validate_all_plugin_constraints(registry)
        self.assertGreater(len(plugin_results), 0)
        
        # 7. Dependency analysis
        dependency_mapper = DependencyMapper(registry)
        dependency_mapper.build_dependency_graph()
        
        # 8. Sensitivity analysis
        sensitivity_analyzer = SensitivityAnalyzer(registry)
        
        # 9. Export and visualization
        export_manager = ExportManager(registry)
        visualizer = CouplingVisualizer(registry)
        
        # Test export functionality
        json_path = os.path.join(self.temp_dir, "test_export.json")
        export_manager.export_parameters_json(json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Verify exported data
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.assertIn('parameters', data)
        self.assertIn('metadata', data)
        
        print("✓ Complete workflow integration test passed")
    
    def test_cli_integration(self):
        """Test CLI integration with the complete system."""
        from unittest.mock import MagicMock
        
        # Test validation command
        args = MagicMock()
        args.config = self.config_file
        args.plugins = ['vacuum_stability']
        args.output = None
        args.output_format = 'text'
        
        result = cmd_validate(args)
        self.assertIn(result, [0, 1])  # Should complete without crashing
        
        # Test export command
        args = MagicMock()
        args.config = self.config_file
        args.format = 'json'
        args.output = os.path.join(self.temp_dir, 'cli_export.json')
        args.parameters = None
        
        result = cmd_export(args)
        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists(args.output))
        
        print("✓ CLI integration test passed")
    
    def test_realm_integration_workflow(self):
        """Test realm integration workflow."""
        # Initialize integration manager
        manager = RealmIntegrationManager(self.config_file, self.temp_dir)
        
        # Register plugins
        manager.register_plugin("vacuum_stability")
        
        # Test execution summary
        summary = manager.get_execution_summary()
        self.assertIn('total_parameters', summary)
        self.assertIn('total_constraints', summary)
        
        # Test report generation
        report_path = manager.generate_analysis_report(include_visualizations=False)
        self.assertTrue(os.path.exists(report_path))
        
        print("✓ Realm integration workflow test passed")
    
    def test_cross_platform_consistency(self):
        """Test cross-platform consistency."""
        # Create registry and set parameters
        registry = ParameterRegistry()
        load_parameters_from_yaml(self.config_file, registry)
        
        # Set consistent parameter values
        test_params = {
            'k_J': 1e-12,
            'xi': 2.0,
            'n_vac': 1.0,
            'PPN_gamma': 1.0,
            'T_CMB_K': 2.725
        }
        
        for param_name, value in test_params.items():
            registry.update_parameter(param_name, value, "consistency_test", "Test value")
        
        # Run validation multiple times
        validator = CompositeValidator("Consistency Test")
        validator.add_validator(BoundsValidator())
        
        results = []
        for i in range(3):
            report = validator.validate_all(registry)
            results.append(report.overall_status.value)
        
        # Results should be consistent
        self.assertEqual(len(set(results)), 1, "Validation results should be consistent")
        
        print("✓ Cross-platform consistency test passed")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        registry = ParameterRegistry()
        
        # Test with missing configuration file
        with self.assertRaises(FileNotFoundError):
            load_parameters_from_yaml("nonexistent.yaml", registry)
        
        # Test with invalid parameter values
        registry.update_parameter("test_param", 1.0, "test", "Test value")
        
        # Try to update with invalid realm
        try:
            registry.update_parameter("test_param", 2.0, "", "Invalid realm")
        except ValueError:
            pass  # Expected
        
        # Test plugin error handling
        plugin_manager = PluginManager()
        
        # Register valid plugin
        success = plugin_manager.register_plugin(VacuumStabilityPlugin())
        self.assertTrue(success)
        
        # Try to register invalid plugin
        success = plugin_manager.register_plugin("invalid_plugin")
        self.assertFalse(success)
        
        print("✓ Error handling and recovery test passed")
    
    def test_performance_under_load(self):
        """Test system performance under load."""
        import time
        
        # Create large parameter set
        registry = ParameterRegistry()
        n_params = 100
        
        start_time = time.time()
        
        for i in range(n_params):
            param_name = f"param_{i:03d}"
            value = i * 0.1
            registry.update_parameter(param_name, value, "load_test", f"Parameter {i}")
        
        creation_time = time.time() - start_time
        
        # Test validation performance
        validator = CompositeValidator("Load Test")
        validator.add_validator(BoundsValidator())
        
        start_time = time.time()
        report = validator.validate_all(registry)
        validation_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(creation_time, 1.0, f"Parameter creation took {creation_time:.2f}s")
        self.assertLess(validation_time, 2.0, f"Validation took {validation_time:.2f}s")
        
        print(f"✓ Performance test passed: {n_params} params in {creation_time:.3f}s + {validation_time:.3f}s")
    
    def test_data_integrity_and_consistency(self):
        """Test data integrity and consistency across operations."""
        registry = ParameterRegistry()
        load_parameters_from_yaml(self.config_file, registry)
        
        # Set initial values
        initial_values = {
            'k_J': 1e-12,
            'xi': 2.0,
            'n_vac': 1.0
        }
        
        for param_name, value in initial_values.items():
            registry.update_parameter(param_name, value, "integrity_test", "Initial value")
        
        # Export data
        export_manager = ExportManager(registry)
        json_path = os.path.join(self.temp_dir, "integrity_test.json")
        export_manager.export_parameters_json(json_path)
        
        # Verify exported data matches registry
        with open(json_path, 'r') as f:
            exported_data = json.load(f)
        
        for param_name, expected_value in initial_values.items():
            exported_value = exported_data['parameters'][param_name]['value']
            self.assertEqual(exported_value, expected_value,
                           f"Exported value for {param_name} doesn't match registry")
        
        # Test parameter history integrity
        registry.update_parameter('k_J', 2e-12, "integrity_test", "Updated value")
        param = registry.get_parameter('k_J')
        
        self.assertEqual(len(param.history), 2, "Parameter should have 2 history entries")
        self.assertEqual(param.history[0].new_value, 1e-12, "First history entry should match")
        self.assertEqual(param.history[1].new_value, 2e-12, "Second history entry should match")
        
        print("✓ Data integrity and consistency test passed")
    
    def test_comprehensive_system_validation(self):
        """Test comprehensive system validation with all components."""
        # Initialize all major components
        registry = ParameterRegistry()
        load_parameters_from_yaml(self.config_file, registry)
        
        # Set realistic parameter values
        realistic_params = {
            'k_J': 1e-15,
            'xi': 2.0,
            'psi_s0': -1.5,
            'n_vac': 1.0,
            'PPN_gamma': 1.0,
            'PPN_beta': 1.0,
            'T_CMB_K': 2.7255,
            'm_gamma': 1e-20,
            'Lambda': 1.1e-52
        }
        
        for param_name, value in realistic_params.items():
            registry.update_parameter(param_name, value, "comprehensive_test", "Realistic value")
        
        # 1. Validation
        validator = CompositeValidator("Comprehensive Test")
        validator.add_validator(PPNValidator())
        validator.add_validator(CMBValidator())
        validator.add_validator(BoundsValidator())
        validator.add_validator(FixedValueValidator())
        validator.add_validator(TargetValueValidator())
        
        report = validator.validate_all(registry)
        self.assertIsNotNone(report)
        
        # 2. Plugin validation
        plugin_manager = PluginManager()
        plugin_manager.register_plugin(VacuumStabilityPlugin())
        plugin_manager.register_plugin(PhotonMassConstraintPlugin())
        
        plugin_results = plugin_manager.validate_all_plugin_constraints(registry)
        valid_plugins = [r for r in plugin_results if r.is_valid()]
        self.assertGreater(len(valid_plugins), 0, "At least some plugins should validate successfully")
        
        # 3. Analysis
        dependency_mapper = DependencyMapper(registry)
        dependency_mapper.build_dependency_graph()
        
        sensitivity_analyzer = SensitivityAnalyzer(registry)
        
        # 4. Export and visualization
        export_manager = ExportManager(registry)
        visualizer = CouplingVisualizer(registry)
        
        # Create comprehensive report
        report_dir = os.path.join(self.temp_dir, "comprehensive_report")
        export_manager.create_comprehensive_report(
            report_dir,
            dependency_mapper=dependency_mapper
        )
        
        # Verify report files exist
        expected_files = [
            "parameters.json",
            "parameters.yaml",
            "parameters.csv",
            "parameters_table.tex",
            "README.md"
        ]
        
        for filename in expected_files:
            file_path = os.path.join(report_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Report file {filename} should exist")
        
        print("✓ Comprehensive system validation test passed")


if __name__ == "__main__":
    unittest.main()
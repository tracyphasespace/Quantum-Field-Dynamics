"""
Tests for visualization and export functionality.
"""

import unittest
import tempfile
import os
import json
import yaml
import csv
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, ParameterState, Constraint, ConstraintType
)
from coupling_constants.visualization.coupling_visualizer import CouplingVisualizer
from coupling_constants.visualization.export_manager import ExportManager
from coupling_constants.analysis.dependency_mapper import DependencyMapper
from coupling_constants.config.yaml_loader import load_parameters_from_yaml


class TestCouplingVisualizer(unittest.TestCase):
    """Test coupling visualizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
        # Add test parameters with constraints
        constraint1 = Constraint(
            realm="test_realm",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        self.registry.add_constraint("param1", constraint1)
        self.registry.update_parameter("param1", 0.5, "test_realm")
        
        constraint2 = Constraint(
            realm="test_realm",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6
        )
        self.registry.add_constraint("param2", constraint2)
        self.registry.update_parameter("param2", 2.725, "test_realm")
        
        self.visualizer = CouplingVisualizer(self.registry)
        
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        self.assertIsNotNone(self.visualizer.registry)
        self.assertEqual(self.visualizer.figure_size, (12, 8))
        self.assertEqual(self.visualizer.dpi, 300)
        self.assertIn('nodes', self.visualizer.color_scheme)
        self.assertIn('edges', self.visualizer.color_scheme)
    
    def test_plot_dependency_graph(self):
        """Test dependency graph plotting."""
        # Create dependency mapper
        mapper = DependencyMapper(self.registry)
        mapper.build_dependency_graph()
        
        # Test plotting (will create file)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_graph.png")
            
            try:
                self.visualizer.plot_dependency_graph(mapper, output_path)
                # Check if file was created (matplotlib might not be available)
                # This is mainly testing that the method doesn't crash
                self.assertTrue(True)  # If we get here, no exception was raised
            except ImportError:
                # matplotlib not available, skip test
                self.skipTest("Matplotlib not available")
            except Exception as e:
                # Other errors are actual test failures
                self.fail(f"plot_dependency_graph failed: {e}")
    
    def test_plot_parameter_constraints(self):
        """Test parameter constraints plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_constraints.png")
            
            try:
                self.visualizer.plot_parameter_constraints(output_path)
                self.assertTrue(True)  # If we get here, no exception was raised
            except ImportError:
                self.skipTest("Matplotlib not available")
            except Exception as e:
                self.fail(f"plot_parameter_constraints failed: {e}")
    
    def test_plot_parameter_evolution(self):
        """Test parameter evolution plotting."""
        # Add some history to parameters
        self.registry.update_parameter("param1", 0.6, "test_realm2")
        self.registry.update_parameter("param1", 0.7, "test_realm3")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_evolution.png")
            
            try:
                self.visualizer.plot_parameter_evolution(["param1"], output_path)
                self.assertTrue(True)
            except ImportError:
                self.skipTest("Matplotlib not available")
            except Exception as e:
                self.fail(f"plot_parameter_evolution failed: {e}")


class TestExportManager(unittest.TestCase):
    """Test export manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
        # Add test parameters with various constraints
        constraint1 = Constraint(
            realm="test_realm",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0,
            notes="Test bounded constraint"
        )
        self.registry.add_constraint("param1", constraint1)
        self.registry.update_parameter("param1", 0.5, "test_realm", "Initial value")
        
        constraint2 = Constraint(
            realm="test_realm2",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6,
            notes="Test fixed constraint"
        )
        self.registry.add_constraint("param2", constraint2)
        self.registry.update_parameter("param2", 2.725, "test_realm2", "Fixed value")
        
        # Add parameter with metadata
        param3 = ParameterState(
            name="param3",
            metadata={"unit": "kg", "note": "Test parameter with metadata"}
        )
        self.registry.register_parameter(param3)
        self.registry.update_parameter("param3", 1.5, "test_realm3", "With metadata")
        
        self.export_manager = ExportManager(self.registry)
    
    def test_export_parameters_json(self):
        """Test JSON export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.json")
            
            self.export_manager.export_parameters_json(output_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Check content
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('metadata', data)
            self.assertIn('parameters', data)
            self.assertIn('param1', data['parameters'])
            self.assertIn('param2', data['parameters'])
            
            # Check parameter data structure
            param1_data = data['parameters']['param1']
            self.assertEqual(param1_data['value'], 0.5)
            self.assertIn('constraints', param1_data)
            self.assertIn('history', param1_data)
    
    def test_export_parameters_yaml(self):
        """Test YAML export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.yaml")
            
            self.export_manager.export_parameters_yaml(output_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Check content
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)
            
            self.assertIn('metadata', data)
            self.assertIn('parameters', data)
            self.assertIn('param1', data['parameters'])
    
    def test_export_parameters_csv(self):
        """Test CSV export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.csv")
            
            self.export_manager.export_parameters_csv(output_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Check content
            with open(output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            self.assertGreater(len(rows), 0)
            
            # Check required columns
            if rows:
                self.assertIn('parameter_name', rows[0])
                self.assertIn('value', rows[0])
                self.assertIn('constraint_count', rows[0])
    
    def test_export_latex_table(self):
        """Test LaTeX table export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_table.tex")
            
            self.export_manager.export_latex_table(output_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Check content
            with open(output_path, 'r') as f:
                content = f.read()
            
            self.assertIn('\\begin{table}', content)
            self.assertIn('\\end{table}', content)
            self.assertIn('\\begin{tabular}', content)
            self.assertIn('param1', content.replace('_', '\\_'))  # LaTeX escaping
    
    def test_create_comprehensive_report(self):
        """Test comprehensive report creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.export_manager.create_comprehensive_report(temp_dir)
            
            # Check that multiple files were created
            expected_files = [
                "parameters.json",
                "parameters.yaml", 
                "parameters.csv",
                "parameters_table.tex",
                "README.md"
            ]
            
            for filename in expected_files:
                file_path = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(file_path), f"Missing file: {filename}")
    
    def test_export_for_publication(self):
        """Test publication export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_params = ["param1", "param2"]
            
            self.export_manager.export_for_publication(
                temp_dir, 
                key_parameters=key_params,
                paper_title="Test Publication"
            )
            
            # Check publication files
            expected_files = [
                "main_results_table.tex",
                "supplementary_data.csv",
                "complete_dataset.json",
                "README.md"
            ]
            
            for filename in expected_files:
                file_path = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(file_path), f"Missing publication file: {filename}")
            
            # Check README content
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            self.assertIn("Test Publication", readme_content)
            self.assertIn("param1", readme_content)
            self.assertIn("param2", readme_content)


class TestVisualizationWithRealData(unittest.TestCase):
    """Test visualization with real QFD data."""
    
    def test_with_real_qfd_data(self):
        """Test visualization and export with real QFD configuration."""
        # Load real configuration
        registry = ParameterRegistry()
        load_parameters_from_yaml("qfd_params/defaults.yaml", registry)
        
        # Set some parameter values
        registry.update_parameter("k_J", 1e-12, "realm0_cmb")
        registry.update_parameter("xi", 2.0, "realm4_em")
        registry.update_parameter("psi_s0", -1.5, "realm0_cmb")
        
        # Test export manager
        export_manager = ExportManager(registry)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON export
            json_path = os.path.join(temp_dir, "real_data.json")
            export_manager.export_parameters_json(json_path)
            self.assertTrue(os.path.exists(json_path))
            
            # Test CSV export
            csv_path = os.path.join(temp_dir, "real_data.csv")
            export_manager.export_parameters_csv(csv_path)
            self.assertTrue(os.path.exists(csv_path))
            
            # Test comprehensive report
            report_dir = os.path.join(temp_dir, "report")
            export_manager.create_comprehensive_report(report_dir)
            self.assertTrue(os.path.exists(os.path.join(report_dir, "README.md")))
        
        # Test visualizer
        visualizer = CouplingVisualizer(registry)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test parameter constraints plot
                constraints_path = os.path.join(temp_dir, "real_constraints.png")
                visualizer.plot_parameter_constraints(constraints_path)
                
                # Test parameter evolution plot
                evolution_path = os.path.join(temp_dir, "real_evolution.png")
                visualizer.plot_parameter_evolution(["k_J", "xi"], evolution_path)
                
                self.assertTrue(True)  # If we get here, no exceptions were raised
            except ImportError:
                self.skipTest("Matplotlib not available for visualization tests")


if __name__ == "__main__":
    unittest.main()
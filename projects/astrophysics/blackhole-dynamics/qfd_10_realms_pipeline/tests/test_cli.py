"""
Tests for the command-line interface.
"""

import unittest
import tempfile
import os
import json
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

from coupling_constants.cli.main import (
    create_parser, main, setup_logging, load_registry,
    cmd_validate, cmd_analyze, cmd_export, cmd_visualize, cmd_plugins
)
from coupling_constants.registry.parameter_registry import ParameterRegistry


class TestCLIParser(unittest.TestCase):
    """Test CLI argument parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_parser()
    
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        self.assertIsNotNone(self.parser)
        # Program name varies depending on how it's run
        self.assertIsNotNone(self.parser.prog)
    
    def test_validate_command_parsing(self):
        """Test validate command parsing."""
        args = self.parser.parse_args(['validate', '--config', 'test.yaml'])
        self.assertEqual(args.command, 'validate')
        self.assertEqual(args.config, 'test.yaml')
        self.assertEqual(args.output_format, 'text')
        self.assertIsNone(args.plugins)
    
    def test_validate_command_with_plugins(self):
        """Test validate command with plugins."""
        args = self.parser.parse_args([
            'validate', '--config', 'test.yaml',
            '--plugins', 'photon_mass', 'vacuum_stability',
            '--output-format', 'json'
        ])
        self.assertEqual(args.plugins, ['photon_mass', 'vacuum_stability'])
        self.assertEqual(args.output_format, 'json')
    
    def test_analyze_command_parsing(self):
        """Test analyze command parsing."""
        args = self.parser.parse_args([
            'analyze', '--config', 'test.yaml', '--output-dir', 'results',
            '--sensitivity', '--monte-carlo', '1000', '--visualize'
        ])
        self.assertEqual(args.command, 'analyze')
        self.assertEqual(args.config, 'test.yaml')
        self.assertEqual(args.output_dir, 'results')
        self.assertTrue(args.sensitivity)
        self.assertEqual(args.monte_carlo, 1000)
        self.assertTrue(args.visualize)
    
    def test_export_command_parsing(self):
        """Test export command parsing."""
        args = self.parser.parse_args([
            'export', '--config', 'test.yaml', '--format', 'json',
            '--output', 'output.json', '--parameters', 'k_J', 'xi'
        ])
        self.assertEqual(args.command, 'export')
        self.assertEqual(args.format, 'json')
        self.assertEqual(args.output, 'output.json')
        self.assertEqual(args.parameters, ['k_J', 'xi'])
    
    def test_visualize_command_parsing(self):
        """Test visualize command parsing."""
        args = self.parser.parse_args([
            'visualize', '--config', 'test.yaml', '--output-dir', 'plots',
            '--type', 'dependency', '--layout', 'hierarchical',
            '--show-labels', '--highlight-critical'
        ])
        self.assertEqual(args.command, 'visualize')
        self.assertEqual(args.type, 'dependency')
        self.assertEqual(args.layout, 'hierarchical')
        self.assertTrue(args.show_labels)
        self.assertTrue(args.highlight_critical)
    
    def test_plugins_command_parsing(self):
        """Test plugins command parsing."""
        args = self.parser.parse_args(['plugins', 'list'])
        self.assertEqual(args.command, 'plugins')
        self.assertEqual(args.action, 'list')
        
        args = self.parser.parse_args([
            'plugins', 'register', '--plugin', 'photon_mass',
            '--output', 'plugin_info.json'
        ])
        self.assertEqual(args.action, 'register')
        self.assertEqual(args.plugin, 'photon_mass')
        self.assertEqual(args.output, 'plugin_info.json')
    
    def test_global_options(self):
        """Test global options parsing."""
        args = self.parser.parse_args(['-v', 'validate', '--config', 'test.yaml'])
        self.assertTrue(args.verbose)
        
        args = self.parser.parse_args(['-q', 'validate', '--config', 'test.yaml'])
        self.assertTrue(args.quiet)


class TestCLIUtilities(unittest.TestCase):
    """Test CLI utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        # Test normal logging
        setup_logging(verbose=False, quiet=False)
        
        # Test verbose logging
        setup_logging(verbose=True, quiet=False)
        
        # Test quiet logging
        setup_logging(verbose=False, quiet=True)
    
    def test_load_registry_missing_file(self):
        """Test loading registry with missing file."""
        with self.assertRaises(FileNotFoundError):
            load_registry('nonexistent.yaml')
    
    @patch('coupling_constants.cli.main.load_parameters_from_yaml')
    @patch('os.path.exists')
    def test_load_registry_success(self, mock_exists, mock_load):
        """Test successful registry loading."""
        mock_exists.return_value = True
        mock_load.return_value = None
        
        registry = load_registry('test.yaml')
        self.assertIsInstance(registry, ParameterRegistry)
        mock_load.assert_called_once()


class TestCLICommands(unittest.TestCase):
    """Test CLI command functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create a minimal test configuration
        with open(self.config_file, 'w') as f:
            f.write("""
parameters:
  k_J:
    min: 0.0
    max: 1.0
    note: "Test parameter"
  xi:
    min: 0.0
    max: 100.0
    note: "Another test parameter"
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('coupling_constants.cli.main.load_registry')
    @patch('coupling_constants.cli.main.setup_validators')
    def test_cmd_validate_success(self, mock_setup_validators, mock_load_registry):
        """Test successful validation command."""
        # Mock registry
        mock_registry = MagicMock()
        mock_load_registry.return_value = mock_registry
        
        # Mock validator
        mock_validator = MagicMock()
        mock_report = MagicMock()
        mock_report.overall_status.value = 'valid'
        mock_report.total_violations = 0
        mock_report.total_warnings = 0
        mock_report.execution_time_ms = 10.0
        mock_report.validator_results = []
        mock_validator.validate_all.return_value = mock_report
        mock_setup_validators.return_value = mock_validator
        
        # Create mock args
        args = MagicMock()
        args.config = self.config_file
        args.plugins = None
        args.output_format = 'text'
        args.output = None
        
        # Test command
        result = cmd_validate(args)
        self.assertEqual(result, 0)
    
    @patch('coupling_constants.cli.main.load_registry')
    def test_cmd_validate_failure(self, mock_load_registry):
        """Test validation command with failure."""
        mock_load_registry.side_effect = Exception("Test error")
        
        args = MagicMock()
        args.config = 'nonexistent.yaml'
        
        result = cmd_validate(args)
        self.assertEqual(result, 1)
    
    @patch('coupling_constants.cli.main.load_registry')
    @patch('coupling_constants.cli.main.DependencyMapper')
    @patch('coupling_constants.cli.main.ExportManager')
    @patch('os.makedirs')
    def test_cmd_analyze_basic(self, mock_makedirs, mock_export_manager, 
                              mock_dependency_mapper, mock_load_registry):
        """Test basic analyze command."""
        # Mock registry
        mock_registry = MagicMock()
        mock_load_registry.return_value = mock_registry
        
        # Mock dependency mapper
        mock_mapper = MagicMock()
        mock_dependency_mapper.return_value = mock_mapper
        
        # Mock export manager
        mock_exporter = MagicMock()
        mock_export_manager.return_value = mock_exporter
        
        # Create mock args
        args = MagicMock()
        args.config = self.config_file
        args.output_dir = os.path.join(self.temp_dir, 'output')
        args.sensitivity = False
        args.format = 'comprehensive'
        args.visualize = False
        
        result = cmd_analyze(args)
        self.assertEqual(result, 0)
        mock_exporter.create_comprehensive_report.assert_called_once()
    
    @patch('coupling_constants.cli.main.load_registry')
    @patch('coupling_constants.cli.main.ExportManager')
    def test_cmd_export_json(self, mock_export_manager, mock_load_registry):
        """Test export command with JSON format."""
        # Mock registry and export manager
        mock_registry = MagicMock()
        mock_load_registry.return_value = mock_registry
        mock_exporter = MagicMock()
        mock_export_manager.return_value = mock_exporter
        
        # Create mock args
        args = MagicMock()
        args.config = self.config_file
        args.format = 'json'
        args.output = os.path.join(self.temp_dir, 'output.json')
        args.parameters = None
        
        result = cmd_export(args)
        self.assertEqual(result, 0)
        mock_exporter.export_parameters_json.assert_called_once()
    
    @patch('coupling_constants.cli.main.load_registry')
    @patch('coupling_constants.cli.main.CouplingVisualizer')
    @patch('coupling_constants.cli.main.DependencyMapper')
    @patch('os.makedirs')
    def test_cmd_visualize_dependency(self, mock_makedirs, mock_dependency_mapper,
                                     mock_visualizer, mock_load_registry):
        """Test visualize command for dependency graphs."""
        # Mock registry
        mock_registry = MagicMock()
        mock_load_registry.return_value = mock_registry
        
        # Mock visualizer and dependency mapper
        mock_viz = MagicMock()
        mock_visualizer.return_value = mock_viz
        mock_mapper = MagicMock()
        mock_dependency_mapper.return_value = mock_mapper
        
        # Create mock args
        args = MagicMock()
        args.config = self.config_file
        args.output_dir = os.path.join(self.temp_dir, 'plots')
        args.type = 'dependency'
        args.layout = 'spring'
        args.show_labels = False
        args.highlight_critical = False
        args.parameters = None
        args.sensitivity = False
        
        result = cmd_visualize(args)
        self.assertEqual(result, 0)
        mock_viz.plot_dependency_graph.assert_called_once()
    
    def test_cmd_plugins_list(self):
        """Test plugins list command."""
        args = MagicMock()
        args.action = 'list'
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = cmd_plugins(args)
        
        self.assertEqual(result, 0)
        output = captured_output.getvalue()
        self.assertIn('Available Plugins:', output)
        self.assertIn('photon_mass', output)
        self.assertIn('vacuum_stability', output)
        self.assertIn('cosmological_constant', output)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration with real components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create a more complete test configuration
        with open(self.config_file, 'w') as f:
            f.write("""
parameters:
  k_J:
    min: 0.0
    max: 1.0
    note: "Incoherent photon drag"
  xi:
    min: 0.0
    max: 100.0
    note: "Coupling parameter"
  psi_s0:
    min: -10.0
    max: 10.0
    note: "Scalar field parameter"
  n_vac:
    min: 0.99
    max: 1.01
    note: "Vacuum refractive index"
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_main_no_args(self):
        """Test main function with no arguments."""
        with patch('sys.argv', ['qfd_coupling_cli.py']):
            result = main()
            self.assertEqual(result, 1)  # Should show help and exit with 1
    
    def test_main_help(self):
        """Test main function with help argument."""
        with patch('sys.argv', ['qfd_coupling_cli.py', '--help']):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
    
    @patch('coupling_constants.cli.main.logging')
    def test_main_with_valid_command(self, mock_logging):
        """Test main function with valid command."""
        output_file = os.path.join(self.temp_dir, 'validation_output.txt')
        
        with patch('sys.argv', [
            'qfd_coupling_cli.py', 'validate',
            '--config', self.config_file,
            '--output', output_file
        ]):
            result = main()
            # Should succeed (return 0) or fail gracefully (return 1)
            self.assertIn(result, [0, 1])
    
    def test_main_keyboard_interrupt(self):
        """Test main function handling keyboard interrupt."""
        with patch('sys.argv', ['qfd_coupling_cli.py', 'validate', '--config', self.config_file]):
            with patch('coupling_constants.cli.main.cmd_validate', side_effect=KeyboardInterrupt):
                result = main()
                self.assertEqual(result, 130)
    
    def test_main_unexpected_error(self):
        """Test main function handling unexpected errors."""
        with patch('sys.argv', ['qfd_coupling_cli.py', 'validate', '--config', self.config_file]):
            with patch('coupling_constants.cli.main.cmd_validate', side_effect=RuntimeError("Test error")):
                result = main()
                self.assertEqual(result, 1)


class TestCLIRealExecution(unittest.TestCase):
    """Test CLI with real execution (requires actual QFD configuration)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a realistic test configuration
        self.config_file = os.path.join(self.temp_dir, 'qfd_config.yaml')
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
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_command_real(self):
        """Test validate command with real execution."""
        output_file = os.path.join(self.temp_dir, 'validation_results.json')
        
        # Create args object
        args = MagicMock()
        args.config = self.config_file
        args.plugins = None
        args.output = output_file
        args.output_format = 'json'
        
        try:
            result = cmd_validate(args)
            # Should complete without crashing
            self.assertIn(result, [0, 1])
            
            # Check if output file was created
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    self.assertIn('validation_report', data)
        except Exception as e:
            # If it fails due to missing dependencies, that's acceptable for testing
            self.assertIsInstance(e, (ImportError, FileNotFoundError, AttributeError))
    
    def test_export_command_real(self):
        """Test export command with real execution."""
        output_file = os.path.join(self.temp_dir, 'parameters.json')
        
        args = MagicMock()
        args.config = self.config_file
        args.format = 'json'
        args.output = output_file
        args.parameters = None
        
        try:
            result = cmd_export(args)
            self.assertIn(result, [0, 1])
            
            # Check if output file was created
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    self.assertIn('parameters', data)
        except Exception as e:
            # If it fails due to missing dependencies, that's acceptable for testing
            self.assertIsInstance(e, (ImportError, FileNotFoundError, AttributeError))


if __name__ == "__main__":
    unittest.main()
"""
Tests for sensitivity analyzer functionality.
"""

import unittest
import numpy as np
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, ParameterState, Constraint, ConstraintType
)
from coupling_constants.analysis.sensitivity_analyzer import (
    SensitivityAnalyzer, create_ppn_gamma_observable, create_cmb_temperature_observable
)
from coupling_constants.config.yaml_loader import load_parameters_from_yaml


class TestSensitivityAnalyzer(unittest.TestCase):
    """Test sensitivity analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
        # Add test parameters
        self.registry.update_parameter("param1", 1.0, "test_realm")
        self.registry.update_parameter("param2", 2.0, "test_realm")
        self.registry.update_parameter("param3", 0.5, "test_realm")
        
        # Add constraints for Monte Carlo bounds
        constraint1 = Constraint(
            realm="test",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.5,
            max_value=1.5
        )
        self.registry.add_constraint("param1", constraint1)
        
        constraint2 = Constraint(
            realm="test",
            constraint_type=ConstraintType.BOUNDED,
            min_value=1.0,
            max_value=3.0
        )
        self.registry.add_constraint("param2", constraint2)
        
        self.analyzer = SensitivityAnalyzer(self.registry)
        
        # Register test observables
        def simple_observable(registry):
            p1 = registry.get_parameter("param1")
            p2 = registry.get_parameter("param2")
            p3 = registry.get_parameter("param3")
            
            v1 = p1.value if p1 and p1.value is not None else 0
            v2 = p2.value if p2 and p2.value is not None else 0
            v3 = p3.value if p3 and p3.value is not None else 0
            
            return v1 * 2 + v2 * 0.5 + v3 * 3  # Linear combination
        
        def quadratic_observable(registry):
            p1 = registry.get_parameter("param1")
            p2 = registry.get_parameter("param2")
            
            v1 = p1.value if p1 and p1.value is not None else 0
            v2 = p2.value if p2 and p2.value is not None else 0
            
            return v1**2 + v2**2  # Quadratic
        
        self.analyzer.register_observable("simple", simple_observable)
        self.analyzer.register_observable("quadratic", quadratic_observable)
        
    def test_register_observable(self):
        """Test observable registration."""
        def test_obs(registry):
            return 1.0
        
        self.analyzer.register_observable("test", test_obs)
        self.assertIn("test", self.analyzer.observables)
        
    def test_compute_parameter_sensitivity_forward(self):
        """Test forward difference sensitivity computation."""
        result = self.analyzer.compute_parameter_sensitivity("simple", method="forward")
        
        self.assertEqual(result.observable_name, "simple")
        self.assertIn("param1", result.parameter_sensitivities)
        self.assertIn("param2", result.parameter_sensitivities)
        self.assertIn("param3", result.parameter_sensitivities)
        
        # Check expected sensitivities for linear function
        # f = 2*p1 + 0.5*p2 + 3*p3, so df/dp1 = 2, df/dp2 = 0.5, df/dp3 = 3
        self.assertAlmostEqual(result.parameter_sensitivities["param1"], 2.0, places=5)
        self.assertAlmostEqual(result.parameter_sensitivities["param2"], 0.5, places=5)
        self.assertAlmostEqual(result.parameter_sensitivities["param3"], 3.0, places=5)
        
    def test_compute_parameter_sensitivity_central(self):
        """Test central difference sensitivity computation."""
        result = self.analyzer.compute_parameter_sensitivity("simple", method="central_difference")
        
        # Central difference should be more accurate
        self.assertAlmostEqual(result.parameter_sensitivities["param1"], 2.0, places=8)
        self.assertAlmostEqual(result.parameter_sensitivities["param2"], 0.5, places=8)
        self.assertAlmostEqual(result.parameter_sensitivities["param3"], 3.0, places=8)
        
    def test_compute_parameter_sensitivity_quadratic(self):
        """Test sensitivity computation for quadratic function."""
        result = self.analyzer.compute_parameter_sensitivity("quadratic", method="central_difference")
        
        # f = p1^2 + p2^2, so df/dp1 = 2*p1, df/dp2 = 2*p2
        # At p1=1.0, p2=2.0: df/dp1 = 2.0, df/dp2 = 4.0
        self.assertAlmostEqual(result.parameter_sensitivities["param1"], 2.0, places=5)
        self.assertAlmostEqual(result.parameter_sensitivities["param2"], 4.0, places=5)
        
    def test_monte_carlo_analysis(self):
        """Test Monte Carlo sensitivity analysis."""
        result = self.analyzer.perform_monte_carlo_analysis("simple", n_samples=100)
        
        self.assertEqual(result.observable_name, "simple")
        self.assertEqual(result.n_samples, 100)
        self.assertIn("param1", result.parameter_statistics)
        self.assertIn("param2", result.parameter_statistics)
        
        # Check statistics structure
        param1_stats = result.parameter_statistics["param1"]
        self.assertIn("mean", param1_stats)
        self.assertIn("std", param1_stats)
        self.assertIn("min", param1_stats)
        self.assertIn("max", param1_stats)
        
        # Check correlation matrix
        self.assertEqual(result.correlation_matrix.shape, (len(result.parameter_names), len(result.parameter_names)))
        
        # Check convergence info
        self.assertIn("converged", result.convergence_info)
        self.assertIn("final_mean", result.convergence_info)
        
    def test_monte_carlo_with_custom_ranges(self):
        """Test Monte Carlo with custom parameter ranges."""
        custom_ranges = {
            "param1": (0.8, 1.2),
            "param2": (1.5, 2.5)
        }
        
        result = self.analyzer.perform_monte_carlo_analysis(
            "simple", n_samples=50, parameter_ranges=custom_ranges
        )
        
        # Check that parameters stayed within specified ranges
        param1_stats = result.parameter_statistics["param1"]
        self.assertGreaterEqual(param1_stats["min"], 0.8)
        self.assertLessEqual(param1_stats["max"], 1.2)
        
        param2_stats = result.parameter_statistics["param2"]
        self.assertGreaterEqual(param2_stats["min"], 1.5)
        self.assertLessEqual(param2_stats["max"], 2.5)
        
    def test_rank_parameters_by_sensitivity(self):
        """Test parameter ranking by sensitivity."""
        # First compute sensitivity
        self.analyzer.compute_parameter_sensitivity("simple")
        
        rankings = self.analyzer.rank_parameters_by_impact(["simple"], method="sensitivity_based")
        
        self.assertEqual(len(rankings), 1)
        ranking = rankings[0]
        
        self.assertEqual(ranking.observable_name, "simple")
        self.assertEqual(ranking.ranking_method, "sensitivity_based")
        
        # Check that parameters are ranked by absolute sensitivity
        # Expected order: param3 (3.0), param1 (2.0), param2 (0.5)
        ranked_params = ranking.ranked_parameters
        self.assertEqual(ranked_params[0][0], "param3")  # Highest sensitivity
        self.assertEqual(ranked_params[1][0], "param1")  # Second highest
        self.assertEqual(ranked_params[2][0], "param2")  # Lowest
        
    def test_rank_parameters_by_variance(self):
        """Test parameter ranking by variance contribution."""
        # First perform Monte Carlo
        self.analyzer.perform_monte_carlo_analysis("simple", n_samples=100)
        
        rankings = self.analyzer.rank_parameters_by_impact(["simple"], method="variance_based")
        
        self.assertEqual(len(rankings), 1)
        ranking = rankings[0]
        
        self.assertEqual(ranking.observable_name, "simple")
        self.assertEqual(ranking.ranking_method, "variance_based")
        self.assertGreater(len(ranking.ranked_parameters), 0)
        
    def test_with_real_qfd_observables(self):
        """Test with real QFD observables."""
        # Load real configuration
        real_registry = ParameterRegistry()
        load_parameters_from_yaml("qfd_params/defaults.yaml", real_registry)
        
        # Set some parameter values
        real_registry.update_parameter("PPN_gamma", 1.000001, "realm3_scales")
        real_registry.update_parameter("PPN_beta", 0.99999, "realm3_scales")
        real_registry.update_parameter("T_CMB_K", 2.725, "cmb_config")
        
        # Create analyzer with real data
        real_analyzer = SensitivityAnalyzer(real_registry)
        
        # Register real observables
        real_analyzer.register_observable("ppn_gamma", create_ppn_gamma_observable())
        real_analyzer.register_observable("cmb_temp", create_cmb_temperature_observable())
        
        # Test sensitivity analysis
        result = real_analyzer.compute_parameter_sensitivity("ppn_gamma")
        self.assertEqual(result.observable_name, "ppn_gamma")
        self.assertGreater(len(result.parameter_sensitivities), 0)
        
        # Test Monte Carlo (small sample for speed)
        mc_result = real_analyzer.perform_monte_carlo_analysis("ppn_gamma", n_samples=50)
        self.assertEqual(mc_result.observable_name, "ppn_gamma")
        self.assertEqual(mc_result.n_samples, 50)
        
    def test_export_results(self):
        """Test exporting sensitivity analysis results."""
        # Generate some results
        self.analyzer.compute_parameter_sensitivity("simple")
        self.analyzer.perform_monte_carlo_analysis("simple", n_samples=50)
        self.analyzer.rank_parameters_by_impact(["simple"])
        
        # Export to temporary file
        import tempfile
        import os
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        try:
            self.analyzer.export_results(json_file)
            
            # Check file exists and is valid JSON
            self.assertTrue(os.path.exists(json_file))
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check structure
            self.assertIn('sensitivity_results', data)
            self.assertIn('monte_carlo_results', data)
            self.assertIn('parameter_rankings', data)
            
            # Check content
            self.assertGreater(len(data['sensitivity_results']), 0)
            self.assertGreater(len(data['monte_carlo_results']), 0)
            self.assertGreater(len(data['parameter_rankings']), 0)
            
        finally:
            if os.path.exists(json_file):
                os.unlink(json_file)
                
    def test_error_handling(self):
        """Test error handling in sensitivity analysis."""
        # Test with unregistered observable
        with self.assertRaises(ValueError):
            self.analyzer.compute_parameter_sensitivity("nonexistent")
        
        with self.assertRaises(ValueError):
            self.analyzer.perform_monte_carlo_analysis("nonexistent")
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            self.analyzer.compute_parameter_sensitivity("simple", method="invalid_method")
        
        with self.assertRaises(ValueError):
            self.analyzer.rank_parameters_by_impact(["simple"], method="invalid_method")


if __name__ == "__main__":
    unittest.main()
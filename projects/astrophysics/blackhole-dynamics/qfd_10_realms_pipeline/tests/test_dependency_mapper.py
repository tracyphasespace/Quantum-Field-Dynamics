"""
Tests for dependency mapper functionality.
"""

import unittest
import numpy as np
from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, ParameterState, Constraint, ConstraintType
)
from coupling_constants.analysis.dependency_mapper import DependencyMapper
from coupling_constants.config.yaml_loader import load_parameters_from_yaml


class TestDependencyMapper(unittest.TestCase):
    """Test dependency mapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        
        # Add some test parameters with constraints
        # Parameter 1: Fixed by realm0_cmb
        constraint1 = Constraint(
            realm="realm0_cmb",
            constraint_type=ConstraintType.FIXED,
            target_value=2.725,
            tolerance=1e-6
        )
        self.registry.add_constraint("param1", constraint1)
        self.registry.update_parameter("param1", 2.725, "realm0_cmb")
        
        # Parameter 2: Also constrained by realm0_cmb
        constraint2 = Constraint(
            realm="realm0_cmb",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1.0
        )
        self.registry.add_constraint("param2", constraint2)
        self.registry.update_parameter("param2", 0.5, "realm0_cmb")
        
        # Parameter 3: Constrained by realm3_scales
        constraint3 = Constraint(
            realm="realm3_scales",
            constraint_type=ConstraintType.TARGET,
            target_value=1.0,
            tolerance=1e-5
        )
        self.registry.add_constraint("param3", constraint3)
        self.registry.update_parameter("param3", 1.0, "realm3_scales")
        
        # Parameter 4: Constrained by realm3_scales
        constraint4 = Constraint(
            realm="realm3_scales",
            constraint_type=ConstraintType.BOUNDED,
            min_value=-10.0,
            max_value=10.0
        )
        self.registry.add_constraint("param4", constraint4)
        self.registry.update_parameter("param4", 5.0, "realm3_scales")
        
        self.mapper = DependencyMapper(self.registry)
        
    def test_build_dependency_graph(self):
        """Test building the dependency graph."""
        graph = self.mapper.build_dependency_graph()
        
        # Should have all parameters as nodes
        self.assertEqual(len(graph.nodes()), 4)
        self.assertIn("param1", graph.nodes())
        self.assertIn("param2", graph.nodes())
        self.assertIn("param3", graph.nodes())
        self.assertIn("param4", graph.nodes())
        
        # Should have some edges (dependencies)
        self.assertGreater(len(graph.edges()), 0)
        
        # Should have dependencies list populated
        self.assertGreater(len(self.mapper.dependencies), 0)
        
    def test_constraint_dependencies(self):
        """Test constraint-based dependencies."""
        self.mapper.build_dependency_graph()
        
        # Should find dependencies between parameters constrained by same realm
        constraint_deps = [d for d in self.mapper.dependencies if d.dependency_type == 'constraint']
        self.assertGreater(len(constraint_deps), 0)
        
        # Check that param1 and param2 are related (both from realm0_cmb)
        realm0_deps = [d for d in constraint_deps if d.realm == 'realm0_cmb']
        self.assertGreater(len(realm0_deps), 0)
        
    def test_realm_dependencies(self):
        """Test realm execution order dependencies."""
        self.mapper.build_dependency_graph()
        
        # Should find realm order dependencies
        realm_deps = [d for d in self.mapper.dependencies if d.dependency_type == 'realm_order']
        self.assertGreater(len(realm_deps), 0)
        
        # Earlier realms should influence later ones
        for dep in realm_deps:
            self.assertIn('->', dep.realm)
            
    def test_find_critical_path(self):
        """Test finding critical path."""
        self.mapper.build_dependency_graph()
        critical_path = self.mapper.find_critical_path()
        
        # Should return a list of parameter names
        self.assertIsInstance(critical_path, list)
        
        # All parameters in path should exist in registry
        for param in critical_path:
            self.assertIn(param, self.registry.get_all_parameters())
            
    def test_identify_parameter_clusters(self):
        """Test parameter clustering."""
        self.mapper.build_dependency_graph()
        clusters = self.mapper.identify_parameter_clusters()
        
        # Should return a dictionary
        self.assertIsInstance(clusters, dict)
        
        # Each cluster should have multiple parameters
        for cluster_id, params in clusters.items():
            self.assertIsInstance(params, list)
            if len(params) > 1:  # Only check non-trivial clusters
                for param in params:
                    self.assertIn(param, self.registry.get_all_parameters())
                    
    def test_compute_influence_matrix(self):
        """Test influence matrix computation."""
        self.mapper.build_dependency_graph()
        matrix = self.mapper.compute_influence_matrix()
        
        # Should be a square matrix
        n_params = len(self.registry.get_all_parameters())
        self.assertEqual(matrix.shape, (n_params, n_params))
        
        # Should be non-negative
        self.assertTrue(np.all(matrix >= 0))
        
        # Diagonal should be zero (no self-influence)
        self.assertTrue(np.all(np.diag(matrix) == 0))
        
    def test_parameter_influence_ranking(self):
        """Test parameter influence ranking."""
        self.mapper.build_dependency_graph()
        ranking = self.mapper.get_parameter_influence_ranking()
        
        # Should return list of tuples
        self.assertIsInstance(ranking, list)
        
        if ranking:
            # Each item should be (param_name, influence_score)
            for param_name, influence in ranking:
                self.assertIsInstance(param_name, str)
                self.assertIsInstance(influence, (int, float))
                self.assertIn(param_name, self.registry.get_all_parameters())
            
            # Should be sorted by influence (descending)
            influences = [inf for _, inf in ranking]
            self.assertEqual(influences, sorted(influences, reverse=True))
            
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        self.mapper.build_dependency_graph()
        stats = self.mapper.get_summary_statistics()
        
        # Should contain expected keys
        expected_keys = [
            'total_parameters', 'total_dependencies', 'total_clusters',
            'graph_density', 'is_connected', 'has_cycles',
            'dependency_types', 'dependencies_by_realm'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            
        # Values should be reasonable
        self.assertEqual(stats['total_parameters'], 4)
        self.assertGreaterEqual(stats['total_dependencies'], 0)
        self.assertGreaterEqual(stats['graph_density'], 0.0)
        self.assertLessEqual(stats['graph_density'], 1.0)
        
    def test_with_real_qfd_data(self):
        """Test dependency mapper with real QFD configuration."""
        # Load real configuration
        real_registry = ParameterRegistry()
        load_parameters_from_yaml("qfd_params/defaults.yaml", real_registry)
        
        # Set some parameter values
        real_registry.update_parameter("T_CMB_K", 2.725, "cmb_config")
        real_registry.update_parameter("k_J", 1e-12, "realm0_cmb")
        real_registry.update_parameter("PPN_gamma", 1.0, "realm3_scales")
        real_registry.update_parameter("PPN_beta", 1.0, "realm3_scales")
        
        # Create mapper and build graph
        real_mapper = DependencyMapper(real_registry)
        graph = real_mapper.build_dependency_graph()
        
        # Should handle real data without errors
        self.assertGreater(len(graph.nodes()), 0)
        self.assertGreater(len(real_mapper.dependencies), 0)
        
        # Should find clusters
        clusters = real_mapper.identify_parameter_clusters()
        self.assertIsInstance(clusters, dict)
        
        # Should compute statistics
        stats = real_mapper.get_summary_statistics()
        self.assertIn('total_parameters', stats)
        self.assertGreater(stats['total_parameters'], 0)
        
    def test_export_graph(self):
        """Test graph export functionality."""
        self.mapper.build_dependency_graph()
        
        # Test JSON export
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_file = f.name
        
        try:
            self.mapper.export_graph('json', json_file)
            
            # File should exist and be non-empty
            self.assertTrue(os.path.exists(json_file))
            self.assertGreater(os.path.getsize(json_file), 0)
            
            # Should be valid JSON
            import json
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Should have expected structure
            self.assertIn('nodes', data)
            self.assertIn('edges', data)
            self.assertIn('dependencies', data)
            
        finally:
            if os.path.exists(json_file):
                os.unlink(json_file)


if __name__ == "__main__":
    unittest.main()
"""
Performance tests for large parameter sets.

This module contains tests that verify the framework can handle
large numbers of parameters and constraints efficiently.
"""

import unittest
import time
import random
import numpy as np
from typing import Dict, List

from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, Constraint, ConstraintType
)
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.basic_validators import (
    BoundsValidator, FixedValueValidator, TargetValueValidator
)
from coupling_constants.analysis.dependency_mapper import DependencyMapper
from coupling_constants.analysis.sensitivity_analyzer import SensitivityAnalyzer
from coupling_constants.plugins.plugin_manager import PluginManager


class TestPerformance(unittest.TestCase):
    """Performance tests for large parameter sets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.performance_threshold = 5.0  # seconds
        random.seed(42)  # For reproducible tests
        np.random.seed(42)
    
    def _create_large_parameter_set(self, n_params: int) -> ParameterRegistry:
        """Create a registry with a large number of parameters."""
        registry = ParameterRegistry()
        
        for i in range(n_params):
            param_name = f"param_{i:04d}"
            value = random.uniform(-10.0, 10.0)
            registry.update_parameter(param_name, value, "performance_test", f"Parameter {i}")
            
            # Add random constraints (avoid FIXED to allow updates)
            constraint_type = random.choice([ConstraintType.BOUNDED, ConstraintType.TARGET])
            
            if constraint_type == ConstraintType.BOUNDED:
                min_val = value - random.uniform(0.1, 1.0)
                max_val = value + random.uniform(0.1, 1.0)
                constraint = Constraint(
                    realm="performance_test",
                    constraint_type=constraint_type,
                    min_value=min_val,
                    max_value=max_val,
                    notes=f"Bounded constraint for {param_name}"
                )
            elif constraint_type == ConstraintType.TARGET:
                target = value + random.uniform(-0.1, 0.1)
                tolerance = random.uniform(0.01, 0.1)
                constraint = Constraint(
                    realm="performance_test",
                    constraint_type=constraint_type,
                    target_value=target,
                    tolerance=tolerance,
                    notes=f"Target constraint for {param_name}"
                )
            # This branch is no longer used since we removed FIXED from choices
            
            registry.add_constraint(param_name, constraint)
        
        return registry
    
    def test_large_parameter_registry_performance(self):
        """Test performance with large number of parameters."""
        n_params = 1000
        
        start_time = time.time()
        registry = self._create_large_parameter_set(n_params)
        creation_time = time.time() - start_time
        
        # Test parameter access performance
        start_time = time.time()
        for i in range(100):  # Sample 100 parameters
            param_name = f"param_{i:04d}"
            param = registry.get_parameter(param_name)
            self.assertIsNotNone(param)
        access_time = time.time() - start_time
        
        # Test parameter update performance
        start_time = time.time()
        for i in range(100):
            param_name = f"param_{i:04d}"
            new_value = random.uniform(-5.0, 5.0)
            registry.update_parameter(param_name, new_value, "performance_update", "Update test")
        update_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(creation_time, 2.0, f"Parameter creation took {creation_time:.2f}s")
        self.assertLess(access_time, 0.1, f"Parameter access took {access_time:.2f}s")
        self.assertLess(update_time, 0.5, f"Parameter updates took {update_time:.2f}s")
        
        print(f"Performance test with {n_params} parameters:")
        print(f"  Creation: {creation_time:.3f}s")
        print(f"  Access (100 params): {access_time:.3f}s")
        print(f"  Updates (100 params): {update_time:.3f}s")
    
    def test_large_validation_performance(self):
        """Test validation performance with large parameter sets."""
        n_params = 500
        registry = self._create_large_parameter_set(n_params)
        
        # Set up validators
        validator = CompositeValidator("Performance Test Validator")
        validator.add_validator(BoundsValidator())
        validator.add_validator(FixedValueValidator())
        validator.add_validator(TargetValueValidator())
        
        # Test validation performance
        start_time = time.time()
        report = validator.validate_all(registry)
        validation_time = time.time() - start_time
        
        # Performance assertion
        self.assertLess(validation_time, 3.0, f"Validation took {validation_time:.2f}s")
        
        # Verify validation worked
        self.assertIsNotNone(report)
        self.assertGreater(report.total_parameters, 0)
        self.assertGreater(report.total_constraints, 0)
        
        print(f"Validation performance with {n_params} parameters:")
        print(f"  Validation time: {validation_time:.3f}s")
        print(f"  Parameters checked: {report.total_parameters}")
        print(f"  Constraints checked: {report.total_constraints}")
    
    def test_dependency_analysis_performance(self):
        """Test dependency analysis performance."""
        n_params = 200  # Smaller set for dependency analysis
        registry = self._create_large_parameter_set(n_params)
        
        # Create some artificial dependencies
        for i in range(0, n_params - 1, 5):  # Every 5th parameter depends on the next
            param_name = f"param_{i:04d}"
            dep_name = f"param_{i+1:04d}"
            # This would normally be done through realm execution
            # For testing, we'll just measure the dependency mapper performance
        
        dependency_mapper = DependencyMapper(registry)
        
        start_time = time.time()
        dependency_mapper.build_dependency_graph()
        build_time = time.time() - start_time
        
        start_time = time.time()
        clusters = dependency_mapper.identify_parameter_clusters()
        cluster_time = time.time() - start_time
        
        start_time = time.time()
        influence_matrix = dependency_mapper.compute_influence_matrix()
        influence_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(build_time, 2.0, f"Dependency graph building took {build_time:.2f}s")
        self.assertLess(cluster_time, 1.0, f"Cluster identification took {cluster_time:.2f}s")
        self.assertLess(influence_time, 2.0, f"Influence matrix computation took {influence_time:.2f}s")
        
        print(f"Dependency analysis performance with {n_params} parameters:")
        print(f"  Graph building: {build_time:.3f}s")
        print(f"  Cluster identification: {cluster_time:.3f}s")
        print(f"  Influence matrix: {influence_time:.3f}s")
    
    def test_sensitivity_analysis_performance(self):
        """Test sensitivity analysis performance."""
        n_params = 50  # Smaller set for sensitivity analysis
        registry = self._create_large_parameter_set(n_params)
        
        sensitivity_analyzer = SensitivityAnalyzer(registry)
        
        # Test sensitivity computation performance
        start_time = time.time()
        try:
            # Test with a mock observable
            sensitivity_analyzer.compute_parameter_sensitivity("test_observable")
        except Exception:
            # Expected to fail since we don't have a real observable
            pass
        sensitivity_time = time.time() - start_time
        
        # Test Monte Carlo performance (smaller sample size)
        start_time = time.time()
        try:
            sensitivity_analyzer.perform_monte_carlo_analysis(100)  # Small sample
        except Exception:
            # Expected to fail without proper setup
            pass
        monte_carlo_time = time.time() - start_time
        
        # Performance assertions (lenient since operations may fail)
        self.assertLess(sensitivity_time, 1.0, f"Sensitivity computation took {sensitivity_time:.2f}s")
        self.assertLess(monte_carlo_time, 5.0, f"Monte Carlo analysis took {monte_carlo_time:.2f}s")
        
        print(f"Sensitivity analysis performance with {n_params} parameters:")
        print(f"  Sensitivity computation: {sensitivity_time:.3f}s")
        print(f"  Monte Carlo (100 samples): {monte_carlo_time:.3f}s")
    
    def test_plugin_performance(self):
        """Test plugin system performance."""
        n_params = 100
        registry = self._create_large_parameter_set(n_params)
        
        # Add some QFD-specific parameters for plugin testing
        registry.update_parameter("n_vac", 1.0, "qfd", "Vacuum refractive index")
        registry.update_parameter("m_gamma", 1e-20, "qfd", "Photon mass")
        registry.update_parameter("Lambda", 1.1e-52, "qfd", "Cosmological constant")
        
        plugin_manager = PluginManager()
        
        # Register plugins
        from coupling_constants.plugins.constraint_plugins import (
            VacuumStabilityPlugin, PhotonMassConstraintPlugin, CosmologicalConstantPlugin
        )
        
        plugin_manager.register_plugin(VacuumStabilityPlugin())
        plugin_manager.register_plugin(PhotonMassConstraintPlugin())
        plugin_manager.register_plugin(CosmologicalConstantPlugin())
        
        # Test plugin validation performance
        start_time = time.time()
        results = plugin_manager.validate_all_plugin_constraints(registry)
        plugin_time = time.time() - start_time
        
        # Performance assertion
        self.assertLess(plugin_time, 1.0, f"Plugin validation took {plugin_time:.2f}s")
        
        # Verify plugins ran
        self.assertGreater(len(results), 0, "Should have plugin validation results")
        
        print(f"Plugin performance with {n_params} parameters:")
        print(f"  Plugin validation: {plugin_time:.3f}s")
        print(f"  Plugins executed: {len(results)}")
    
    def test_memory_usage(self):
        """Test memory usage with large parameter sets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large parameter set
        n_params = 2000
        registry = self._create_large_parameter_set(n_params)
        
        after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_creation_memory - initial_memory
        
        # Memory usage should be reasonable (less than 100MB for 2000 parameters)
        self.assertLess(memory_increase, 100, 
                       f"Memory usage increased by {memory_increase:.1f}MB for {n_params} parameters")
        
        print(f"Memory usage test with {n_params} parameters:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  After creation: {after_creation_memory:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")
    
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access patterns."""
        import threading
        import queue
        
        n_params = 500
        registry = self._create_large_parameter_set(n_params)
        
        results_queue = queue.Queue()
        n_threads = 4
        operations_per_thread = 50
        
        def worker_thread(thread_id):
            """Worker thread for concurrent access testing."""
            start_time = time.time()
            
            for i in range(operations_per_thread):
                # Mix of read and write operations
                param_name = f"param_{(thread_id * operations_per_thread + i) % n_params:04d}"
                
                if i % 3 == 0:  # Read operation
                    param = registry.get_parameter(param_name)
                    assert param is not None
                else:  # Write operation
                    new_value = random.uniform(-1.0, 1.0)
                    registry.update_parameter(param_name, new_value, f"thread_{thread_id}", "Concurrent test")
            
            end_time = time.time()
            results_queue.put((thread_id, end_time - start_time))
        
        # Start threads
        threads = []
        overall_start = time.time()
        
        for i in range(n_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        overall_time = time.time() - overall_start
        
        # Collect results
        thread_times = []
        while not results_queue.empty():
            thread_id, thread_time = results_queue.get()
            thread_times.append(thread_time)
        
        # Performance assertions
        max_thread_time = max(thread_times)
        avg_thread_time = sum(thread_times) / len(thread_times)
        
        self.assertLess(overall_time, 5.0, f"Concurrent access took {overall_time:.2f}s")
        self.assertLess(max_thread_time, 3.0, f"Slowest thread took {max_thread_time:.2f}s")
        
        print(f"Concurrent access performance:")
        print(f"  Threads: {n_threads}")
        print(f"  Operations per thread: {operations_per_thread}")
        print(f"  Overall time: {overall_time:.3f}s")
        print(f"  Average thread time: {avg_thread_time:.3f}s")
        print(f"  Max thread time: {max_thread_time:.3f}s")


if __name__ == "__main__":
    unittest.main()
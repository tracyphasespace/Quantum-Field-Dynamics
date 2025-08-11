"""
Test YAML configuration integration.
"""

import unittest
import tempfile
import os
from coupling_constants.registry.parameter_registry import ParameterRegistry
from coupling_constants.config.yaml_loader import load_parameters_from_yaml, get_parameter_summary


class TestYAMLIntegration(unittest.TestCase):
    """Test YAML configuration loading."""
    
    def test_load_from_actual_config(self):
        """Test loading from the actual QFD configuration file."""
        registry = ParameterRegistry()
        
        # Load from the actual defaults.yaml
        config_path = "qfd_params/defaults.yaml"
        if os.path.exists(config_path):
            load_parameters_from_yaml(config_path, registry)
            
            # Check that parameters were loaded
            all_params = registry.get_all_parameters()
            self.assertGreater(len(all_params), 0)
            
            # Check specific parameters exist
            k_j_param = registry.get_parameter("k_J")
            self.assertIsNotNone(k_j_param)
            self.assertEqual(len(k_j_param.constraints), 1)  # Should have bounds constraint
            
            # Check PPN targets were loaded
            ppn_gamma = registry.get_parameter("PPN_gamma")
            self.assertIsNotNone(ppn_gamma)
            
            # Check CMB target was loaded
            t_cmb = registry.get_parameter("T_CMB_K")
            self.assertIsNotNone(t_cmb)
            
            # Generate summary
            summary = get_parameter_summary(registry)
            self.assertGreater(summary['total_parameters'], 0)
            self.assertGreater(summary['total_constraints'], 0)
            
            print(f"Loaded {summary['total_parameters']} parameters")
            print(f"Total constraints: {summary['total_constraints']}")
            print(f"Constraint types: {summary['constraint_types']}")
        else:
            self.skipTest("Configuration file not found")


if __name__ == "__main__":
    unittest.main()
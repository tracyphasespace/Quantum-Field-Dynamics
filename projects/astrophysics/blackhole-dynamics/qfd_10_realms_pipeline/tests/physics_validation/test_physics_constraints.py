"""
Tests for physics constraint validation.

This module contains tests that verify the framework correctly validates
fundamental physics constraints and relationships.
"""

import unittest
import numpy as np
import math

from coupling_constants.registry.parameter_registry import (
    ParameterRegistry, Constraint, ConstraintType
)
from coupling_constants.validation.base_validator import CompositeValidator
from coupling_constants.validation.ppn_validator import PPNValidator
from coupling_constants.validation.cmb_validator import CMBValidator
from coupling_constants.validation.basic_validators import (
    BoundsValidator, FixedValueValidator, TargetValueValidator
)
from coupling_constants.plugins.plugin_manager import PluginManager
from coupling_constants.plugins.constraint_plugins import (
    PhotonMassConstraintPlugin, VacuumStabilityPlugin, CosmologicalConstantPlugin
)


class TestPhysicsConstraints(unittest.TestCase):
    """Test physics constraint validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ParameterRegistry()
        self.validator = CompositeValidator("Physics Constraints Validator")
        self.plugin_manager = PluginManager()
        
        # Set up validators
        self.validator.add_validator(PPNValidator())
        self.validator.add_validator(CMBValidator())
        self.validator.add_validator(BoundsValidator())
        self.validator.add_validator(FixedValueValidator())
        self.validator.add_validator(TargetValueValidator())
        
        # Set up plugins
        self.plugin_manager.register_plugin(PhotonMassConstraintPlugin())
        self.plugin_manager.register_plugin(VacuumStabilityPlugin())
        self.plugin_manager.register_plugin(CosmologicalConstantPlugin())
    
    def test_vacuum_stability_constraints(self):
        """Test vacuum stability constraints."""
        # Set up vacuum parameters
        self.registry.update_parameter("n_vac", 1.0, "vacuum", "Vacuum refractive index")
        self.registry.update_parameter("k_J", 1e-15, "vacuum", "Photon drag")
        self.registry.update_parameter("xi", 2.0, "vacuum", "Coupling parameter")
        
        # Add vacuum stability constraint
        n_vac_constraint = Constraint(
            realm="vacuum_physics",
            constraint_type=ConstraintType.FIXED,
            target_value=1.0,
            tolerance=1e-12,
            notes="Vacuum refractive index must be exactly 1"
        )
        self.registry.add_constraint("n_vac", n_vac_constraint)
        
        # Validate
        report = self.validator.validate_all(self.registry)
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        
        # Check that vacuum stability is maintained
        vacuum_plugin_result = next((r for r in plugin_results if "vacuum_stability" in r.validator_name), None)
        self.assertIsNotNone(vacuum_plugin_result, "Vacuum stability plugin should run")
        self.assertTrue(vacuum_plugin_result.is_valid(), "Vacuum should be stable")
        
        # Test violation of vacuum stability by creating a new registry
        violation_registry = ParameterRegistry()
        violation_registry.update_parameter("n_vac", 1.1, "violation_test", "Violate vacuum stability")
        violation_registry.update_parameter("k_J", 1e-15, "violation_test", "Photon drag")
        violation_registry.update_parameter("xi", 2.0, "violation_test", "Coupling parameter")
        
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(violation_registry)
        vacuum_plugin_result = next((r for r in plugin_results if "vacuum_stability" in r.validator_name), None)
        self.assertFalse(vacuum_plugin_result.is_valid(), "Vacuum should be unstable with n_vac != 1")
    
    def test_photon_mass_constraints(self):
        """Test photon mass constraints."""
        # Set up photon mass parameter within experimental limits
        self.registry.update_parameter("m_gamma", 1e-20, "experiment", "Photon mass")
        self.registry.update_parameter("k_J", 1e-15, "experiment", "Photon drag")
        self.registry.update_parameter("xi", 2.0, "experiment", "Coupling parameter")
        
        # Validate with plugin
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        photon_plugin_result = next((r for r in plugin_results if "photon_mass" in r.validator_name), None)
        
        self.assertIsNotNone(photon_plugin_result, "Photon mass plugin should run")
        self.assertTrue(photon_plugin_result.is_valid(), "Photon mass should be within limits")
        
        # Test violation of photon mass limit
        self.registry.update_parameter("m_gamma", 1e-15, "violation_test", "Violate photon mass limit")
        
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        photon_plugin_result = next((r for r in plugin_results if "photon_mass" in r.validator_name), None)
        self.assertFalse(photon_plugin_result.is_valid(), "Large photon mass should violate constraints")
    
    def test_cosmological_constraints(self):
        """Test cosmological parameter constraints."""
        # Set up cosmological parameters
        self.registry.update_parameter("Lambda", 1.1e-52, "cosmology", "Cosmological constant")
        self.registry.update_parameter("H0", 67.4, "cosmology", "Hubble constant")
        self.registry.update_parameter("psi_s0", -1.5, "cosmology", "Scalar field")
        self.registry.update_parameter("xi", 2.0, "cosmology", "Coupling parameter")
        
        # Validate with plugin
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        cosmo_plugin_result = next((r for r in plugin_results if "cosmological" in r.validator_name), None)
        
        self.assertIsNotNone(cosmo_plugin_result, "Cosmological constant plugin should run")
        self.assertTrue(cosmo_plugin_result.is_valid(), "Cosmological parameters should be consistent")
        
        # Test violation of cosmological constraints
        self.registry.update_parameter("Lambda", 1e-50, "violation_test", "Violate cosmological constant")
        
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        cosmo_plugin_result = next((r for r in plugin_results if "cosmological" in r.validator_name), None)
        self.assertFalse(cosmo_plugin_result.is_valid(), "Large cosmological constant should violate constraints")
    
    def test_ppn_constraints(self):
        """Test PPN parameter constraints."""
        # Set up PPN parameters for General Relativity
        self.registry.update_parameter("PPN_gamma", 1.0, "gr", "PPN gamma parameter")
        self.registry.update_parameter("PPN_beta", 1.0, "gr", "PPN beta parameter")
        
        # Add PPN constraints
        gamma_constraint = Constraint(
            realm="solar_system",
            constraint_type=ConstraintType.TARGET,
            target_value=1.0,
            tolerance=2.3e-5,  # Cassini constraint
            notes="Solar system constraint on gamma"
        )
        self.registry.add_constraint("PPN_gamma", gamma_constraint)
        
        beta_constraint = Constraint(
            realm="solar_system",
            constraint_type=ConstraintType.TARGET,
            target_value=1.0,
            tolerance=8e-5,  # Lunar laser ranging constraint
            notes="Solar system constraint on beta"
        )
        self.registry.add_constraint("PPN_beta", beta_constraint)
        
        # Validate
        report = self.validator.validate_all(self.registry)
        
        # Should be valid for GR values
        self.assertIn(report.overall_status.value, ['valid', 'warning'])
        
        # Test violation of PPN constraints
        self.registry.update_parameter("PPN_gamma", 1.1, "violation_test", "Violate PPN gamma")
        
        report = self.validator.validate_all(self.registry)
        self.assertEqual(report.overall_status.value, 'invalid', "Large PPN gamma should violate constraints")
    
    def test_cmb_constraints(self):
        """Test CMB parameter constraints."""
        # Set up CMB parameters
        self.registry.update_parameter("T_CMB_K", 2.7255, "cmb", "CMB temperature")
        self.registry.update_parameter("k_J", 1e-15, "cmb", "Photon drag")
        self.registry.update_parameter("psi_s0", -1.5, "cmb", "Scalar field")
        
        # Add CMB temperature constraint
        temp_constraint = Constraint(
            realm="cmb_observation",
            constraint_type=ConstraintType.TARGET,
            target_value=2.7255,
            tolerance=0.0006,  # FIRAS precision
            notes="FIRAS measurement of CMB temperature"
        )
        self.registry.add_constraint("T_CMB_K", temp_constraint)
        
        # Validate
        report = self.validator.validate_all(self.registry)
        
        # Should be valid for correct CMB temperature
        self.assertIn(report.overall_status.value, ['valid', 'warning'])
        
        # Test violation of CMB constraints
        self.registry.update_parameter("T_CMB_K", 3.0, "violation_test", "Violate CMB temperature")
        
        report = self.validator.validate_all(self.registry)
        self.assertEqual(report.overall_status.value, 'invalid', "Wrong CMB temperature should violate constraints")
    
    def test_fundamental_constant_relationships(self):
        """Test relationships between fundamental constants."""
        # Set up fundamental constants
        self.registry.update_parameter("c", 299792458.0, "si", "Speed of light")
        self.registry.update_parameter("hbar", 1.054571817e-34, "si", "Reduced Planck constant")
        self.registry.update_parameter("G", 6.67430e-11, "si", "Gravitational constant")
        self.registry.update_parameter("alpha_em", 7.2973525693e-3, "si", "Fine structure constant")
        
        # Test that speed of light is exactly defined
        c_constraint = Constraint(
            realm="si_definition",
            constraint_type=ConstraintType.FIXED,
            target_value=299792458.0,
            tolerance=0.0,
            notes="Speed of light is defined exactly in SI"
        )
        self.registry.add_constraint("c", c_constraint)
        
        # Validate
        report = self.validator.validate_all(self.registry)
        
        # Should be valid
        self.assertIn(report.overall_status.value, ['valid', 'warning'])
        
        # Test violation of exact constant
        self.registry.update_parameter("c", 299792459.0, "violation_test", "Violate speed of light")
        
        report = self.validator.validate_all(self.registry)
        self.assertEqual(report.overall_status.value, 'invalid', "Wrong speed of light should violate constraints")
    
    def test_constraint_consistency(self):
        """Test consistency between different constraint types."""
        # Set up parameter with multiple constraint types
        param_name = "test_param"
        self.registry.update_parameter(param_name, 5.0, "test", "Test parameter")
        
        # Add bounded constraint
        bounded_constraint = Constraint(
            realm="bounds_test",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=10.0,
            notes="Bounded constraint"
        )
        self.registry.add_constraint(param_name, bounded_constraint)
        
        # Add target constraint (should be consistent)
        target_constraint = Constraint(
            realm="target_test",
            constraint_type=ConstraintType.TARGET,
            target_value=5.0,
            tolerance=0.1,
            notes="Target constraint"
        )
        self.registry.add_constraint(param_name, target_constraint)
        
        # Validate - should be consistent
        report = self.validator.validate_all(self.registry)
        self.assertIn(report.overall_status.value, ['valid', 'warning'])
        
        # Add conflicting fixed constraint
        fixed_constraint = Constraint(
            realm="fixed_test",
            constraint_type=ConstraintType.FIXED,
            target_value=15.0,  # Outside bounds
            tolerance=1e-10,
            notes="Conflicting fixed constraint"
        )
        self.registry.add_constraint(param_name, fixed_constraint)
        
        # This should create a conflict
        conflicts = self.registry.get_conflicting_constraints()
        self.assertGreater(len(conflicts), 0, "Should detect constraint conflicts")
    
    def test_physics_unit_consistency(self):
        """Test unit consistency in physics calculations."""
        # Set up parameters with known units
        self.registry.update_parameter("mass_kg", 1.0, "units", "Mass in kg")
        self.registry.update_parameter("length_m", 1.0, "units", "Length in m")
        self.registry.update_parameter("time_s", 1.0, "units", "Time in s")
        
        # Test dimensional analysis constraints
        # Energy should have units of kg⋅m²⋅s⁻²
        mass = self.registry.get_parameter("mass_kg").value
        length = self.registry.get_parameter("length_m").value
        time = self.registry.get_parameter("time_s").value
        
        # Calculate energy in SI units
        energy_si = mass * (length**2) / (time**2)  # kg⋅m²⋅s⁻²
        
        self.assertEqual(energy_si, 1.0, "Energy calculation should be dimensionally correct")
        
        # Test that fundamental constants have correct dimensions
        c = 299792458.0  # m/s
        hbar = 1.054571817e-34  # J⋅s = kg⋅m²⋅s⁻¹
        
        # Planck length should have units of meters
        planck_length = math.sqrt(hbar * 6.67430e-11 / (c**3))  # Should be in meters
        
        self.assertAlmostEqual(planck_length, 1.616255e-35, places=40,
                              msg="Planck length should have correct value and units")


if __name__ == "__main__":
    unittest.main()
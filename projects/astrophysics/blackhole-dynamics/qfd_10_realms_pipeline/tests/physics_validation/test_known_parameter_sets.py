"""
Tests with known parameter sets that satisfy physical constraints.

This module contains test cases using well-established parameter values
from physics literature and experimental measurements.
"""

import unittest
import numpy as np
from typing import Dict, Any

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


class TestKnownParameterSets(unittest.TestCase):
    """Test with known parameter sets from physics literature."""
    
    def setUp(self):
        """Set up test fixtures with known physics parameters."""
        self.registry = ParameterRegistry()
        self.validator = CompositeValidator("Physics Validation")
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
    
    def test_standard_model_parameters(self):
        """Test with Standard Model parameters."""
        # Known Standard Model values
        sm_params = {
            'alpha_em': 7.2973525693e-3,  # Fine structure constant
            'G_F': 1.1663787e-5,          # Fermi coupling constant (GeV^-2)
            'alpha_s': 0.1181,            # Strong coupling constant at MZ
            'm_electron': 0.5109989461,   # Electron mass (MeV)
            'm_muon': 105.6583745,        # Muon mass (MeV)
            'm_tau': 1776.86,             # Tau mass (MeV)
        }
        
        # Register parameters with appropriate constraints
        for param_name, value in sm_params.items():
            self.registry.update_parameter(param_name, value, "standard_model", "Known SM value")
            
            # Add tight constraints around known values
            constraint = Constraint(
                realm="standard_model",
                constraint_type=ConstraintType.TARGET,
                target_value=value,
                tolerance=abs(value) * 0.001,  # 0.1% tolerance
                notes=f"Standard Model value for {param_name}"
            )
            self.registry.add_constraint(param_name, constraint)
        
        # Validate that SM parameters satisfy constraints
        report = self.validator.validate_all(self.registry)
        
        self.assertEqual(report.overall_status.value, 'valid',
                        f"Standard Model parameters should be valid: {report.get_summary()}")
        self.assertEqual(report.total_violations, 0)
    
    def test_cosmological_parameters(self):
        """Test with known cosmological parameters."""
        # Planck 2018 cosmological parameters
        cosmo_params = {
            'H0': 67.4,                   # Hubble constant (km/s/Mpc)
            'Omega_m': 0.315,             # Matter density parameter
            'Omega_Lambda': 0.685,        # Dark energy density parameter
            'Omega_b': 0.049,             # Baryon density parameter
            'sigma_8': 0.811,             # Matter fluctuation amplitude
            'n_s': 0.965,                 # Scalar spectral index
            'T_CMB_K': 2.7255,            # CMB temperature (K)
        }
        
        # Register cosmological parameters
        for param_name, value in cosmo_params.items():
            self.registry.update_parameter(param_name, value, "planck2018", "Planck 2018 value")
            
            # Add constraints based on Planck uncertainties
            if param_name == 'H0':
                tolerance = 0.5  # ±0.5 km/s/Mpc
            elif param_name == 'T_CMB_K':
                tolerance = 0.0006  # ±0.6 mK
            else:
                tolerance = abs(value) * 0.01  # 1% for other parameters
            
            constraint = Constraint(
                realm="planck2018",
                constraint_type=ConstraintType.TARGET,
                target_value=value,
                tolerance=tolerance,
                notes=f"Planck 2018 measurement for {param_name}"
            )
            self.registry.add_constraint(param_name, constraint)
        
        # Validate cosmological parameters
        report = self.validator.validate_all(self.registry)
        
        # Should be valid (allowing for some warnings due to missing parameters)
        self.assertIn(report.overall_status.value, ['valid', 'warning'],
                     f"Cosmological parameters validation: {report.get_summary()}")
    
    def test_ppn_parameters(self):
        """Test with known PPN (Parametrized Post-Newtonian) parameters."""
        # Solar system tests give tight constraints on PPN parameters
        ppn_params = {
            'PPN_gamma': 1.0,             # Deflection of light
            'PPN_beta': 1.0,              # Perihelion precession
            'PPN_alpha1': 0.0,            # Preferred frame effects
            'PPN_alpha2': 0.0,            # Preferred frame effects
            'PPN_alpha3': 0.0,            # Preferred frame effects
            'PPN_zeta1': 0.0,             # Violation of momentum conservation
            'PPN_zeta2': 0.0,             # Violation of momentum conservation
            'PPN_zeta3': 0.0,             # Violation of momentum conservation
            'PPN_zeta4': 0.0,             # Violation of momentum conservation
        }
        
        # Experimental constraints on PPN parameters
        ppn_tolerances = {
            'PPN_gamma': 2.3e-5,          # Cassini constraint
            'PPN_beta': 8e-5,             # Lunar laser ranging
            'PPN_alpha1': 1e-4,           # Pulsar timing
            'PPN_alpha2': 4e-7,           # Lunar laser ranging
            'PPN_alpha3': 4e-20,          # Pulsar timing
            'PPN_zeta1': 2e-2,            # Lunar laser ranging
            'PPN_zeta2': 4e-5,            # Lunar laser ranging
            'PPN_zeta3': 1e-8,            # Pulsar timing
            'PPN_zeta4': 1e-6,            # Lunar laser ranging
        }
        
        # Register PPN parameters with experimental constraints
        for param_name, value in ppn_params.items():
            self.registry.update_parameter(param_name, value, "solar_system", "Solar system test")
            
            tolerance = ppn_tolerances.get(param_name, 1e-3)
            constraint = Constraint(
                realm="solar_system",
                constraint_type=ConstraintType.TARGET,
                target_value=value,
                tolerance=tolerance,
                notes=f"Solar system constraint on {param_name}"
            )
            self.registry.add_constraint(param_name, constraint)
        
        # Validate PPN parameters
        report = self.validator.validate_all(self.registry)
        
        self.assertIn(report.overall_status.value, ['valid', 'warning'],
                     f"PPN parameters validation: {report.get_summary()}")
    
    def test_qfd_vacuum_parameters(self):
        """Test with QFD vacuum parameters that should satisfy constraints."""
        # QFD parameters that maintain vacuum stability
        qfd_params = {
            'n_vac': 1.0,                 # Vacuum refractive index (exactly 1)
            'k_J': 1e-15,                 # Incoherent photon drag (very small)
            'xi': 2.0,                    # Coupling parameter
            'psi_s0': -1.5,               # Scalar field parameter
            'm_gamma': 1e-20,             # Photon mass (very small)
            'Lambda': 1.1e-52,            # Cosmological constant (m^-2)
        }
        
        # Register QFD parameters
        for param_name, value in qfd_params.items():
            self.registry.update_parameter(param_name, value, "qfd_vacuum", "QFD vacuum value")
        
        # Add vacuum stability constraint
        n_vac_constraint = Constraint(
            realm="qfd_vacuum",
            constraint_type=ConstraintType.FIXED,
            target_value=1.0,
            tolerance=1e-10,
            notes="Vacuum refractive index must be exactly 1"
        )
        self.registry.add_constraint("n_vac", n_vac_constraint)
        
        # Add photon drag constraint
        k_j_constraint = Constraint(
            realm="qfd_vacuum",
            constraint_type=ConstraintType.BOUNDED,
            min_value=0.0,
            max_value=1e-10,
            notes="Photon drag must be negligible in vacuum"
        )
        self.registry.add_constraint("k_J", k_j_constraint)
        
        # Validate with plugins
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        
        # All plugin validations should pass
        for result in plugin_results:
            self.assertTrue(result.is_valid(), 
                          f"Plugin {result.validator_name} failed: {result.violations}")
    
    def test_particle_physics_parameters(self):
        """Test with known particle physics parameters."""
        # Particle Data Group 2020 values
        pdg_params = {
            'm_proton': 938.272081,       # Proton mass (MeV)
            'm_neutron': 939.565413,      # Neutron mass (MeV)
            'm_W': 80379.0,               # W boson mass (MeV)
            'm_Z': 91187.6,               # Z boson mass (MeV)
            'm_higgs': 125100.0,          # Higgs boson mass (MeV)
            'tau_neutron': 879.4,         # Neutron lifetime (s)
            'g_A': 1.2723,                # Axial coupling constant
        }
        
        # Register particle physics parameters
        for param_name, value in pdg_params.items():
            self.registry.update_parameter(param_name, value, "pdg2020", "PDG 2020 value")
            
            # Add constraints with PDG uncertainties
            if param_name == 'm_proton':
                tolerance = 0.000006  # MeV
            elif param_name == 'm_neutron':
                tolerance = 0.000006  # MeV
            elif param_name == 'tau_neutron':
                tolerance = 0.6  # s
            else:
                tolerance = abs(value) * 0.001  # 0.1% for others
            
            constraint = Constraint(
                realm="pdg2020",
                constraint_type=ConstraintType.TARGET,
                target_value=value,
                tolerance=tolerance,
                notes=f"PDG 2020 measurement for {param_name}"
            )
            self.registry.add_constraint(param_name, constraint)
        
        # Validate particle physics parameters
        report = self.validator.validate_all(self.registry)
        
        self.assertIn(report.overall_status.value, ['valid', 'warning'],
                     f"Particle physics parameters validation: {report.get_summary()}")
    
    def test_combined_parameter_set(self):
        """Test with a combined set of parameters from different domains."""
        # Combined parameter set that should be mutually consistent
        combined_params = {
            # Fundamental constants
            'c': 299792458.0,             # Speed of light (m/s)
            'hbar': 1.054571817e-34,      # Reduced Planck constant (J⋅s)
            'G': 6.67430e-11,             # Gravitational constant (m³/kg⋅s²)
            
            # Electromagnetic
            'alpha_em': 7.2973525693e-3,  # Fine structure constant
            'e': 1.602176634e-19,         # Elementary charge (C)
            
            # Cosmological
            'H0': 67.4,                   # Hubble constant (km/s/Mpc)
            'T_CMB_K': 2.7255,            # CMB temperature (K)
            
            # QFD specific
            'n_vac': 1.0,                 # Vacuum refractive index
            'k_J': 1e-15,                 # Incoherent photon drag
            'xi': 2.0,                    # Coupling parameter
        }
        
        # Register all parameters
        for param_name, value in combined_params.items():
            self.registry.update_parameter(param_name, value, "combined_test", "Combined test value")
        
        # Add fundamental physics constraints
        # Speed of light is exactly defined
        c_constraint = Constraint(
            realm="si_definition",
            constraint_type=ConstraintType.FIXED,
            target_value=299792458.0,
            tolerance=0.0,
            notes="Speed of light is defined exactly in SI units"
        )
        self.registry.add_constraint("c", c_constraint)
        
        # Vacuum refractive index must be 1
        n_vac_constraint = Constraint(
            realm="vacuum_physics",
            constraint_type=ConstraintType.FIXED,
            target_value=1.0,
            tolerance=1e-15,
            notes="Vacuum refractive index must be exactly 1"
        )
        self.registry.add_constraint("n_vac", n_vac_constraint)
        
        # Validate combined parameter set
        report = self.validator.validate_all(self.registry)
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        
        # Should be valid or have only warnings
        self.assertIn(report.overall_status.value, ['valid', 'warning'],
                     f"Combined parameter set validation: {report.get_summary()}")
        
        # Plugin validations should mostly pass
        valid_plugins = [r for r in plugin_results if r.is_valid()]
        self.assertGreaterEqual(len(valid_plugins), len(plugin_results) // 2,
                               "At least half of plugin validations should pass")
    
    def test_extreme_but_valid_parameters(self):
        """Test with extreme but physically valid parameter values."""
        # Parameters at the edge of experimental constraints
        extreme_params = {
            'PPN_gamma': 1.0 + 2e-5,      # Near Cassini limit
            'PPN_beta': 1.0 + 7e-5,       # Near LLR limit
            'm_gamma': 1e-18,             # Near experimental photon mass limit
            'k_J': 1e-10,                 # Small but non-zero photon drag
            'xi': 0.1,                    # Small coupling
            'Lambda': 1.2e-52,            # Slightly larger cosmological constant
        }
        
        # Register extreme parameters
        for param_name, value in extreme_params.items():
            self.registry.update_parameter(param_name, value, "extreme_test", "Extreme but valid value")
        
        # Add appropriate constraints
        for param_name, value in extreme_params.items():
            if param_name.startswith('PPN_'):
                # PPN parameters should be close to 1 or 0
                if 'gamma' in param_name or 'beta' in param_name:
                    target = 1.0
                    tolerance = 1e-4
                else:
                    target = 0.0
                    tolerance = 1e-3
                
                constraint = Constraint(
                    realm="extreme_test",
                    constraint_type=ConstraintType.TARGET,
                    target_value=target,
                    tolerance=tolerance,
                    notes=f"Extreme test constraint for {param_name}"
                )
                self.registry.add_constraint(param_name, constraint)
        
        # Validate extreme parameters
        report = self.validator.validate_all(self.registry)
        plugin_results = self.plugin_manager.validate_all_plugin_constraints(self.registry)
        
        # Should be valid or have warnings, but not invalid
        self.assertNotEqual(report.overall_status.value, 'invalid',
                           f"Extreme parameters should not be invalid: {report.get_summary()}")
        
        # Most plugin validations should pass
        valid_plugins = [r for r in plugin_results if r.is_valid()]
        failed_plugins = [r for r in plugin_results if not r.is_valid()]
        
        if failed_plugins:
            print(f"Failed plugin validations: {[r.validator_name for r in failed_plugins]}")
        
        # At least some validations should pass
        self.assertGreater(len(valid_plugins), 0, "At least some plugin validations should pass")


if __name__ == "__main__":
    unittest.main()
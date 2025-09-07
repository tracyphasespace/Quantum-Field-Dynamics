"""
Test data utilities for loading and validating reference data

This module provides utilities for loading reference test data,
validating data integrity, and comparing computed results against
stored reference values.
"""

import json
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings


class ReferenceDataLoader:
    """Utility class for loading and validating reference test data"""
    
    def __init__(self, reference_dir: str = "tests/reference_data"):
        self.reference_dir = Path(reference_dir)
        self._cache = {}
    
    def load_reference_data(self, filename: str, validate_hash: bool = True) -> Dict[str, Any]:
        """Load reference data from JSON file with optional hash validation"""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = self.reference_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Reference data file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            full_data = json.load(f)
        
        if "metadata" not in full_data or "data" not in full_data:
            raise ValueError(f"Invalid reference data format in {filename}")
        
        metadata = full_data["metadata"]
        data = full_data["data"]
        
        # Validate data integrity if requested
        if validate_hash and "data_hash" in metadata:
            computed_hash = self._compute_data_hash(data)
            stored_hash = metadata["data_hash"]
            
            if computed_hash != stored_hash:
                warnings.warn(f"Data hash mismatch in {filename}. "
                            f"Expected {stored_hash}, got {computed_hash}. "
                            f"Reference data may be corrupted or outdated.")
        
        # Cache the data
        self._cache[filename] = {
            "metadata": metadata,
            "data": data
        }
        
        return self._cache[filename]
    
    def get_power_spectrum_reference(self, parameter_set: str = "planck_fiducial") -> Tuple[np.ndarray, np.ndarray]:
        """Get reference power spectrum data"""
        ref_data = self.load_reference_data("power_spectrum_reference.json")
        data = ref_data["data"]
        
        if parameter_set not in data["parameter_sets"]:
            available = list(data["parameter_sets"].keys())
            raise ValueError(f"Parameter set '{parameter_set}' not found. Available: {available}")
        
        k_values = np.array(data["k_values"])
        Pk_values = np.array(data["parameter_sets"][parameter_set]["Pk_values"])
        
        return k_values, Pk_values
    
    def get_window_function_reference(self, configuration: str = "standard_grid") -> Tuple[np.ndarray, np.ndarray]:
        """Get reference window function data"""
        ref_data = self.load_reference_data("window_function_reference.json")
        data = ref_data["data"]
        
        if configuration not in data["configurations"]:
            available = list(data["configurations"].keys())
            raise ValueError(f"Configuration '{configuration}' not found. Available: {available}")
        
        config_data = data["configurations"][configuration]
        chi_grid = np.array(config_data["chi_grid"])
        Wchi = np.array(config_data["Wchi"])
        
        return chi_grid, Wchi
    
    def get_correlation_reference(self, model: str = "standard_model") -> Tuple[np.ndarray, np.ndarray]:
        """Get reference TE correlation data"""
        ref_data = self.load_reference_data("correlation_reference.json")
        data = ref_data["data"]
        
        if model not in data["models"]:
            available = list(data["models"].keys())
            raise ValueError(f"Model '{model}' not found. Available: {available}")
        
        ells = np.array(data["ells"])
        rho_values = np.array(data["models"][model]["rho_values"])
        
        return ells, rho_values
    
    def get_spectrum_reference(self, ell_range: str = "medium_ell") -> Dict[str, np.ndarray]:
        """Get reference CMB spectrum data"""
        ref_data = self.load_reference_data("spectra_reference.json")
        data = ref_data["data"]
        
        if ell_range not in data["spectra"]:
            available = list(data["spectra"].keys())
            raise ValueError(f"Ell range '{ell_range}' not found. Available: {available}")
        
        spectrum_data = data["spectra"][ell_range]
        
        return {
            "ells": np.array(spectrum_data["ells"]),
            "C_TT": np.array(spectrum_data["C_TT"]),
            "C_EE": np.array(spectrum_data["C_EE"]),
            "C_TE": np.array(spectrum_data["C_TE"]),
            "rho": np.array(spectrum_data["rho"])
        }
    
    def get_mueller_reference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get reference Mueller coefficient data"""
        ref_data = self.load_reference_data("mueller_reference.json")
        data = ref_data["data"]
        
        mu_values = np.array(data["mu_values"])
        w_T_values = np.array(data["w_T_values"])
        w_E_values = np.array(data["w_E_values"])
        
        return mu_values, w_T_values, w_E_values
    
    def _compute_data_hash(self, data: Dict[str, Any]) -> str:
        """Compute hash of data for integrity checking"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def list_available_data(self) -> Dict[str, Dict[str, Any]]:
        """List all available reference data files and their metadata"""
        available_data = {}
        
        if not self.reference_dir.exists():
            return available_data
        
        for json_file in self.reference_dir.glob("*.json"):
            try:
                ref_data = self.load_reference_data(json_file.name, validate_hash=False)
                available_data[json_file.name] = ref_data["metadata"]
            except Exception as e:
                available_data[json_file.name] = {"error": str(e)}
        
        return available_data


class ReferenceDataValidator:
    """Utility class for validating computed results against reference data"""
    
    def __init__(self, loader: Optional[ReferenceDataLoader] = None):
        self.loader = loader or ReferenceDataLoader()
    
    def validate_power_spectrum(self, k_values: np.ndarray, Pk_computed: np.ndarray,
                              parameter_set: str = "planck_fiducial",
                              rtol: float = 1e-10, atol: float = 1e-15) -> bool:
        """Validate computed power spectrum against reference"""
        k_ref, Pk_ref = self.loader.get_power_spectrum_reference(parameter_set)
        
        # Interpolate reference to match computed k values
        Pk_ref_interp = np.interp(k_values, k_ref, Pk_ref)
        
        try:
            np.testing.assert_allclose(Pk_computed, Pk_ref_interp, rtol=rtol, atol=atol)
            return True
        except AssertionError as e:
            print(f"Power spectrum validation failed: {e}")
            return False
    
    def validate_window_function(self, chi_grid: np.ndarray, Wchi_computed: np.ndarray,
                               configuration: str = "standard_grid",
                               rtol: float = 1e-10, atol: float = 1e-15) -> bool:
        """Validate computed window function against reference"""
        chi_ref, Wchi_ref = self.loader.get_window_function_reference(configuration)
        
        # Check if grids match
        if not np.allclose(chi_grid, chi_ref, rtol=1e-12):
            print("Warning: chi grids don't match exactly, interpolating reference")
            Wchi_ref_interp = np.interp(chi_grid, chi_ref, Wchi_ref)
        else:
            Wchi_ref_interp = Wchi_ref
        
        try:
            np.testing.assert_allclose(Wchi_computed, Wchi_ref_interp, rtol=rtol, atol=atol)
            return True
        except AssertionError as e:
            print(f"Window function validation failed: {e}")
            return False
    
    def validate_correlation(self, ells: np.ndarray, rho_computed: np.ndarray,
                           model: str = "standard_model",
                           rtol: float = 1e-10, atol: float = 1e-15) -> bool:
        """Validate computed TE correlation against reference"""
        ells_ref, rho_ref = self.loader.get_correlation_reference(model)
        
        # Find common ell range
        ell_min = max(ells.min(), ells_ref.min())
        ell_max = min(ells.max(), ells_ref.max())
        
        # Extract common range
        mask_computed = (ells >= ell_min) & (ells <= ell_max)
        mask_ref = (ells_ref >= ell_min) & (ells_ref <= ell_max)
        
        ells_common = ells[mask_computed]
        rho_computed_common = rho_computed[mask_computed]
        
        # Interpolate reference to match
        rho_ref_interp = np.interp(ells_common, ells_ref[mask_ref], rho_ref[mask_ref])
        
        try:
            np.testing.assert_allclose(rho_computed_common, rho_ref_interp, rtol=rtol, atol=atol)
            return True
        except AssertionError as e:
            print(f"TE correlation validation failed: {e}")
            return False
    
    def validate_spectra(self, ells: np.ndarray, C_TT: np.ndarray, C_EE: np.ndarray, C_TE: np.ndarray,
                        ell_range: str = "medium_ell",
                        rtol: float = 1e-10, atol: float = 1e-15) -> Dict[str, bool]:
        """Validate computed CMB spectra against reference"""
        ref_data = self.loader.get_spectrum_reference(ell_range)
        
        results = {}
        
        # Find common ell range
        ell_min = max(ells.min(), ref_data["ells"].min())
        ell_max = min(ells.max(), ref_data["ells"].max())
        
        mask_computed = (ells >= ell_min) & (ells <= ell_max)
        mask_ref = (ref_data["ells"] >= ell_min) & (ref_data["ells"] <= ell_max)
        
        ells_common = ells[mask_computed]
        
        # Validate each spectrum
        for spectrum_name, computed_values in [("TT", C_TT), ("EE", C_EE), ("TE", C_TE)]:
            ref_values = ref_data[f"C_{spectrum_name}"]
            
            computed_common = computed_values[mask_computed]
            ref_interp = np.interp(ells_common, ref_data["ells"][mask_ref], ref_values[mask_ref])
            
            try:
                np.testing.assert_allclose(computed_common, ref_interp, rtol=rtol, atol=atol)
                results[spectrum_name] = True
            except AssertionError as e:
                print(f"{spectrum_name} spectrum validation failed: {e}")
                results[spectrum_name] = False
        
        return results
    
    def validate_mueller_coefficients(self, mu_values: np.ndarray, 
                                    w_T_computed: np.ndarray, w_E_computed: np.ndarray,
                                    rtol: float = 1e-10, atol: float = 1e-15) -> Dict[str, bool]:
        """Validate computed Mueller coefficients against reference"""
        mu_ref, w_T_ref, w_E_ref = self.loader.get_mueller_reference()
        
        results = {}
        
        # Interpolate reference to match computed mu values
        w_T_ref_interp = np.interp(mu_values, mu_ref, w_T_ref)
        w_E_ref_interp = np.interp(mu_values, mu_ref, w_E_ref)
        
        # Validate intensity weights
        try:
            np.testing.assert_allclose(w_T_computed, w_T_ref_interp, rtol=rtol, atol=atol)
            results["w_T"] = True
        except AssertionError as e:
            print(f"Intensity weight validation failed: {e}")
            results["w_T"] = False
        
        # Validate polarization weights
        try:
            np.testing.assert_allclose(w_E_computed, w_E_ref_interp, rtol=rtol, atol=atol)
            results["w_E"] = True
        except AssertionError as e:
            print(f"Polarization weight validation failed: {e}")
            results["w_E"] = False
        
        return results


# Global instances for easy access
reference_loader = ReferenceDataLoader()
reference_validator = ReferenceDataValidator(reference_loader)
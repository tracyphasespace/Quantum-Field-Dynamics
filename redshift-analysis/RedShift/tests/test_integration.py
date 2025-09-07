"""
Integration tests for QFD CMB workflows

This module contains end-to-end tests that validate complete computation
pipelines, including the demo script and fitting workflows.
"""

import pytest
import numpy as np
import pandas as pd
import os
import subprocess
import tempfile
import json
from pathlib import Path
import sys

# Import modules for direct testing
from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.kernels import te_correlation_phase
from qfd_cmb.projector import project_limber
from qfd_cmb.figures import plot_TT, plot_EE, plot_TE


class TestDemoWorkflow:
    """Test the complete demo script workflow"""
    
    def test_demo_script_execution(self, temp_output_dir):
        """Test that run_demo.py executes successfully and produces expected outputs"""
        # Run the demo script
        cmd = [
            sys.executable, "run_demo.py",
            "--outdir", str(temp_output_dir),
            "--lmin", "2",
            "--lmax", "100"  # Use smaller range for faster testing
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check that script executed successfully
        assert result.returncode == 0, f"Demo script failed: {result.stderr}"
        
        # Check that expected files were created
        expected_files = [
            "qfd_demo_spectra.csv",
            "TT.png",
            "EE.png", 
            "TE.png"
        ]
        
        for filename in expected_files:
            filepath = temp_output_dir / filename
            assert filepath.exists(), f"Expected output file {filename} not created"
            assert filepath.stat().st_size > 0, f"Output file {filename} is empty"
    
    def test_demo_csv_output_format(self, temp_output_dir):
        """Test that demo CSV output has correct format and content"""
        # Run demo with small parameter range
        cmd = [
            sys.executable, "run_demo.py",
            "--outdir", str(temp_output_dir),
            "--lmin", "2",
            "--lmax", "50"
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Load and validate CSV output
        csv_path = temp_output_dir / "qfd_demo_spectra.csv"
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ["ell", "C_TT", "C_TE", "C_EE"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data properties
        assert len(df) == 49, "Expected 49 rows for ell range 2-50"
        assert df["ell"].min() == 2, "Minimum ell should be 2"
        assert df["ell"].max() == 50, "Maximum ell should be 50"
        
        # Check that spectra are positive (TT and EE)
        assert np.all(df["C_TT"] > 0), "TT spectrum must be positive"
        assert np.all(df["C_EE"] > 0), "EE spectrum must be positive"
        
        # Check that values are finite
        for col in required_columns:
            assert np.all(np.isfinite(df[col])), f"Non-finite values in {col}"
    
    def test_demo_plot_generation(self, temp_output_dir):
        """Test that demo generates valid plot files"""
        # Run demo
        cmd = [
            sys.executable, "run_demo.py", 
            "--outdir", str(temp_output_dir),
            "--lmin", "2",
            "--lmax", "20"
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check plot files
        plot_files = ["TT.png", "EE.png", "TE.png"]
        for plot_file in plot_files:
            plot_path = temp_output_dir / plot_file
            assert plot_path.exists(), f"Plot file {plot_file} not created"
            
            # Check file size (should be reasonable for PNG)
            file_size = plot_path.stat().st_size
            assert file_size > 1000, f"Plot file {plot_file} too small ({file_size} bytes)"
            assert file_size < 1e6, f"Plot file {plot_file} too large ({file_size} bytes)"


class TestEndToEndComputationPipeline:
    """Test complete computation pipelines programmatically"""
    
    def test_full_spectrum_computation_pipeline(self, planck_parameters):
        """Test complete spectrum computation from parameters to output"""
        # Extract parameters
        lA = planck_parameters['lA']
        rpsi = planck_parameters['rpsi']
        chi_star = planck_parameters['chi_star']
        sigma_chi = planck_parameters['sigma_chi']
        
        # Define computation parameters
        lmin, lmax = 2, 100
        ells = np.arange(lmin, lmax + 1)
        
        # Step 1: Create window function
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 501)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Validate window function
        assert len(Wchi) == len(chi_grid), "Window function length mismatch"
        assert np.all(np.isfinite(Wchi)), "Window function contains non-finite values"
        assert np.all(Wchi >= 0), "Window function must be non-negative"
        
        # Step 2: Define power spectrum
        Pk = lambda k: oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
        
        # Test power spectrum at sample points
        k_test = np.logspace(-3, 1, 10)
        Pk_test = Pk(k_test)
        assert np.all(np.isfinite(Pk_test)), "Power spectrum contains non-finite values"
        assert np.all(Pk_test > 0), "Power spectrum must be positive"
        
        # Step 3: Project to get TT spectrum
        Ctt = project_limber(ells, Pk, Wchi, chi_grid)
        
        # Validate TT spectrum
        assert len(Ctt) == len(ells), "TT spectrum length mismatch"
        assert np.all(np.isfinite(Ctt)), "TT spectrum contains non-finite values"
        assert np.all(Ctt > 0), "TT spectrum must be positive"
        
        # Step 4: Compute EE spectrum (simplified)
        Cee = 0.25 * Ctt
        assert np.all(Cee > 0), "EE spectrum must be positive"
        
        # Step 5: Compute TE spectrum with correlation
        rho = np.array([te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star) for l in ells])
        Cte = rho * np.sqrt(Ctt * Cee)
        
        # Validate TE spectrum
        assert len(Cte) == len(ells), "TE spectrum length mismatch"
        assert np.all(np.isfinite(Cte)), "TE spectrum contains non-finite values"
        assert np.all(np.abs(rho) <= 1.0), "Correlation coefficient must be |rho| <= 1"
        
        # Step 6: Test output formatting
        df = pd.DataFrame({"ell": ells, "C_TT": Ctt, "C_TE": Cte, "C_EE": Cee})
        
        # Validate DataFrame
        assert len(df) == len(ells), "DataFrame length mismatch"
        assert list(df.columns) == ["ell", "C_TT", "C_TE", "C_EE"], "Incorrect column names"
        
        # Test CSV serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            
            # Read back and validate
            df_read = pd.read_csv(f.name)
            pd.testing.assert_frame_equal(df, df_read)
            
            # Clean up
            os.unlink(f.name)
    
    def test_plotting_pipeline(self, temp_output_dir):
        """Test complete plotting pipeline"""
        # Generate sample data
        ells = np.arange(2, 51)
        Ctt = 1e-10 * (ells / 100.0)**(-1.0)  # Simple power law
        Cee = 0.25 * Ctt
        Cte = 0.1 * np.sqrt(Ctt * Cee)
        
        # Test each plotting function
        plot_functions = [
            (plot_TT, Ctt, "TT_test.png"),
            (plot_EE, Cee, "EE_test.png"),
            (plot_TE, Cte, "TE_test.png")
        ]
        
        for plot_func, spectrum, filename in plot_functions:
            output_path = temp_output_dir / filename
            
            # Generate plot
            plot_func(ells, spectrum, str(output_path))
            
            # Validate output
            assert output_path.exists(), f"Plot file {filename} not created"
            assert output_path.stat().st_size > 0, f"Plot file {filename} is empty"
    
    def test_parameter_variation_pipeline(self, planck_parameters):
        """Test pipeline with different parameter values"""
        base_params = planck_parameters.copy()
        
        # Test parameter variations
        variations = [
            {"rpsi": 100.0},  # Different oscillation scale
            {"Aosc": 0.3},    # Different oscillation amplitude
            {"ns": 0.95},     # Different spectral index
            {"sigma_osc": 0.05}  # Different oscillation width
        ]
        
        ells = np.arange(2, 21)  # Small range for speed
        chi_star = base_params['chi_star']
        sigma_chi = base_params['sigma_chi']
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 101)
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        for variation in variations:
            # Update parameters
            params = base_params.copy()
            params.update(variation)
            
            # Compute spectrum
            Pk = lambda k: oscillatory_psik(
                k, 
                ns=params.get('ns', 0.96),
                rpsi=params.get('rpsi', 147.0),
                Aosc=params.get('Aosc', 0.55),
                sigma_osc=params.get('sigma_osc', 0.025)
            )
            
            Ctt = project_limber(ells, Pk, Wchi, chi_grid)
            
            # Validate results
            assert np.all(np.isfinite(Ctt)), f"Non-finite values with variation {variation}"
            assert np.all(Ctt > 0), f"Non-positive values with variation {variation}"
            assert len(Ctt) == len(ells), f"Length mismatch with variation {variation}"


class TestFittingWorkflow:
    """Test the fitting workflow components"""
    
    def test_fitting_data_preparation(self, temp_output_dir):
        """Test data preparation for fitting workflow"""
        # Create sample data file
        ells = np.arange(2, 101)
        Ctt = 1e-10 * (ells / 100.0)**(-1.0)
        Cee = 0.25 * Ctt
        Cte = 0.1 * np.sqrt(Ctt * Cee)
        
        # Add realistic errors
        error_tt = 0.05 * Ctt
        error_ee = 0.1 * Cee
        error_te = 0.2 * np.abs(Cte)
        
        df = pd.DataFrame({
            "ell": ells,
            "C_TT": Ctt,
            "C_TE": Cte,
            "C_EE": Cee,
            "error_TT": error_tt,
            "error_TE": error_te,
            "error_EE": error_ee
        })
        
        # Save test data
        data_file = temp_output_dir / "test_data.csv"
        df.to_csv(data_file, index=False)
        
        # Validate data file
        assert data_file.exists(), "Test data file not created"
        
        # Read back and validate
        df_read = pd.read_csv(data_file)
        required_cols = ["ell", "C_TT", "C_TE", "C_EE"]
        for col in required_cols:
            assert col in df_read.columns, f"Missing column {col}"
        
        assert len(df_read) == len(ells), "Data length mismatch"
        assert np.all(df_read["C_TT"] > 0), "TT data must be positive"
        assert np.all(df_read["C_EE"] > 0), "EE data must be positive"
    
    @pytest.mark.slow
    def test_fitting_script_execution(self, temp_output_dir):
        """Test that fit_planck.py executes successfully (minimal run)"""
        # Create minimal test data
        ells = np.arange(2, 21)  # Very small range for speed
        Ctt = 1e-10 * (ells / 100.0)**(-1.0)
        
        df = pd.DataFrame({
            "ell": ells,
            "C_TT": Ctt,
            "error_TT": 0.1 * Ctt
        })
        
        data_file = temp_output_dir / "fit_test_data.csv"
        df.to_csv(data_file, index=False)
        
        output_file = temp_output_dir / "fit_results.json"
        
        # Run fitting script with minimal parameters
        cmd = [
            sys.executable, "fit_planck.py",
            "--data", str(data_file),
            "--which", "TT",
            "--steps", "10",  # Very few steps for testing
            "--walkers", "4",  # Few walkers for speed
            "--out", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check execution
        assert result.returncode == 0, f"Fitting script failed: {result.stderr}"
        
        # Check output file
        assert output_file.exists(), "Fitting output file not created"
        
        # Validate JSON output
        with open(output_file, 'r') as f:
            fit_results = json.load(f)
        
        assert "best_params" in fit_results, "Missing best_params in output"
        assert "mean_params" in fit_results, "Missing mean_params in output"
        assert len(fit_results["best_params"]) == 5, "Expected 5 parameters"


class TestOutputValidation:
    """Test output format validation and file generation"""
    
    def test_csv_output_validation(self, temp_output_dir):
        """Test CSV output format validation"""
        # Generate test data
        ells = np.arange(2, 101)
        Ctt = 1e-10 * (ells / 100.0)**(-1.0)
        Cee = 0.25 * Ctt
        Cte = 0.1 * np.sqrt(Ctt * Cee)
        
        df = pd.DataFrame({"ell": ells, "C_TT": Ctt, "C_TE": Cte, "C_EE": Cee})
        
        # Test CSV writing and reading
        csv_file = temp_output_dir / "validation_test.csv"
        df.to_csv(csv_file, index=False)
        
        # Validate file properties
        assert csv_file.exists(), "CSV file not created"
        assert csv_file.stat().st_size > 0, "CSV file is empty"
        
        # Read and validate content
        df_read = pd.read_csv(csv_file)
        
        # Check structure
        assert list(df_read.columns) == ["ell", "C_TT", "C_TE", "C_EE"], "Incorrect columns"
        assert len(df_read) == len(ells), "Incorrect number of rows"
        
        # Check data integrity
        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
        
        # Check numerical properties
        assert np.all(df_read["ell"] == ells), "Ell values corrupted"
        assert np.allclose(df_read["C_TT"], Ctt), "TT values corrupted"
        assert np.allclose(df_read["C_TE"], Cte), "TE values corrupted"
        assert np.allclose(df_read["C_EE"], Cee), "EE values corrupted"
    
    def test_json_output_validation(self, temp_output_dir):
        """Test JSON output format validation"""
        # Create test results data
        test_results = {
            "parameters": {
                "ns": 0.96,
                "rpsi": 147.0,
                "Aosc": 0.55,
                "sigma_osc": 0.025
            },
            "spectra": {
                "ells": list(range(2, 11)),
                "C_TT": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10, 8e-10, 9e-10],
                "C_EE": [2.5e-11, 5e-11, 7.5e-11, 1e-10, 1.25e-10, 1.5e-10, 1.75e-10, 2e-10, 2.25e-10]
            },
            "metadata": {
                "computation_time": 1.23,
                "grid_points": 501,
                "lmax": 10
            }
        }
        
        # Test JSON writing and reading
        json_file = temp_output_dir / "validation_test.json"
        with open(json_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Validate file properties
        assert json_file.exists(), "JSON file not created"
        assert json_file.stat().st_size > 0, "JSON file is empty"
        
        # Read and validate content
        with open(json_file, 'r') as f:
            results_read = json.load(f)
        
        # Check structure
        assert "parameters" in results_read, "Missing parameters section"
        assert "spectra" in results_read, "Missing spectra section"
        assert "metadata" in results_read, "Missing metadata section"
        
        # Check data integrity
        assert results_read == test_results, "JSON data corrupted"
    
    def test_output_file_permissions(self, temp_output_dir):
        """Test that output files have correct permissions"""
        # Create test files
        csv_file = temp_output_dir / "perm_test.csv"
        json_file = temp_output_dir / "perm_test.json"
        png_file = temp_output_dir / "perm_test.png"
        
        # Write test files
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(csv_file, index=False)
        
        with open(json_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Create dummy PNG (just touch the file)
        png_file.touch()
        
        # Check files exist and are readable
        for test_file in [csv_file, json_file, png_file]:
            assert test_file.exists(), f"File {test_file} not created"
            assert os.access(test_file, os.R_OK), f"File {test_file} not readable"
            assert test_file.stat().st_size >= 0, f"File {test_file} has negative size"


# Integration test markers for pytest
pytestmark = [pytest.mark.integration]
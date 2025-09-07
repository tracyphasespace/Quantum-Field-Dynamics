"""
Unit tests for qfd_cmb.figures module

Tests the plot generation functions for plot creation without displaying,
different output formats, and error handling.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from qfd_cmb.figures import plot_TT, plot_EE, plot_TE

# Use non-interactive backend for testing
matplotlib.use('Agg')


class TestPlotTT:
    """Test suite for plot_TT function"""
    
    def test_basic_functionality(self, temp_output_dir):
        """Test basic plot generation for TT spectrum"""
        ells = np.array([2, 10, 50, 100, 500, 1000])
        Ctt = np.array([1e-10, 5e-10, 2e-9, 1e-9, 5e-10, 2e-10])
        
        output_path = temp_output_dir / "test_tt.png"
        
        # Should not raise any exceptions
        plot_TT(ells, Ctt, str(output_path))
        
        # Check that file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_different_output_formats(self, temp_output_dir):
        """Test different output file formats"""
        ells = np.logspace(0.5, 3, 50)
        Ctt = 1e-10 * (ells / 100)**(-2)
        
        formats = ['png', 'pdf', 'svg', 'jpg']
        
        for fmt in formats:
            output_path = temp_output_dir / f"test_tt.{fmt}"
            
            try:
                plot_TT(ells, Ctt, str(output_path))
                assert output_path.exists()
                assert output_path.stat().st_size > 0
            except Exception as e:
                # Some formats might not be available in all environments
                if "format" not in str(e).lower():
                    raise
    
    def test_array_shapes(self, temp_output_dir):
        """Test with different array shapes and sizes"""
        output_path = temp_output_dir / "test_tt_shapes.png"
        
        # Small array
        ells_small = np.array([2, 10, 100])
        Ctt_small = np.array([1e-10, 5e-10, 1e-10])
        plot_TT(ells_small, Ctt_small, str(output_path))
        assert output_path.exists()
        
        # Large array
        output_path2 = temp_output_dir / "test_tt_large.png"
        ells_large = np.arange(2, 2001)
        Ctt_large = 1e-10 * (ells_large / 100)**(-2)
        plot_TT(ells_large, Ctt_large, str(output_path2))
        assert output_path2.exists()
    
    def test_physical_values(self, temp_output_dir, planck_parameters):
        """Test with physically reasonable CMB values"""
        ells = np.logspace(0.5, 3.5, 100)
        
        # Typical TT spectrum shape
        Ctt = 1e-10 * (ells / 100)**(-2) * np.exp(-(ells / 1000)**2)
        
        output_path = temp_output_dir / "test_tt_physical.png"
        plot_TT(ells, Ctt, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should be reasonable file size
    
    def test_edge_cases(self, temp_output_dir):
        """Test edge cases and boundary conditions"""
        # Very small values
        ells = np.array([2, 10, 100])
        Ctt_small = np.array([1e-20, 1e-19, 1e-20])
        
        output_path = temp_output_dir / "test_tt_small.png"
        plot_TT(ells, Ctt_small, str(output_path))
        assert output_path.exists()
        
        # Very large values
        Ctt_large = np.array([1e-5, 1e-4, 1e-5])
        output_path2 = temp_output_dir / "test_tt_large_vals.png"
        plot_TT(ells, Ctt_large, str(output_path2))
        assert output_path2.exists()
        
        # Single point
        ells_single = np.array([100])
        Ctt_single = np.array([1e-10])
        output_path3 = temp_output_dir / "test_tt_single.png"
        plot_TT(ells_single, Ctt_single, str(output_path3))
        assert output_path3.exists()
    
    def test_error_handling(self, temp_output_dir):
        """Test error handling for invalid inputs"""
        ells = np.array([2, 10, 100])
        Ctt = np.array([1e-10, 5e-10, 1e-10])
        
        # Invalid path (directory doesn't exist)
        invalid_path = temp_output_dir / "nonexistent" / "test.png"
        
        with pytest.raises((OSError, FileNotFoundError)):
            plot_TT(ells, Ctt, str(invalid_path))
        
        # Mismatched array sizes
        Ctt_wrong = np.array([1e-10, 5e-10])  # One element short
        output_path = temp_output_dir / "test_error.png"
        
        with pytest.raises((ValueError, IndexError)):
            plot_TT(ells, Ctt_wrong, str(output_path))
    
    def test_no_display(self, temp_output_dir):
        """Test that plots are created without displaying"""
        ells = np.array([2, 10, 50, 100])
        Ctt = np.array([1e-10, 5e-10, 2e-10, 1e-10])
        
        output_path = temp_output_dir / "test_no_display.png"
        
        # Should not open any display windows
        plot_TT(ells, Ctt, str(output_path))
        
        # Check that no figures are left open
        assert len(plt.get_fignums()) == 0


class TestPlotEE:
    """Test suite for plot_EE function"""
    
    def test_basic_functionality(self, temp_output_dir):
        """Test basic plot generation for EE spectrum"""
        ells = np.array([2, 10, 50, 100, 500, 1000])
        Cee = np.array([1e-12, 5e-12, 2e-11, 1e-11, 5e-12, 2e-12])
        
        output_path = temp_output_dir / "test_ee.png"
        
        plot_EE(ells, Cee, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_different_output_formats(self, temp_output_dir):
        """Test different output file formats"""
        ells = np.logspace(0.5, 3, 50)
        Cee = 1e-12 * (ells / 100)**(-2)
        
        formats = ['png', 'pdf', 'svg']
        
        for fmt in formats:
            output_path = temp_output_dir / f"test_ee.{fmt}"
            
            try:
                plot_EE(ells, Cee, str(output_path))
                assert output_path.exists()
                assert output_path.stat().st_size > 0
            except Exception as e:
                if "format" not in str(e).lower():
                    raise
    
    def test_physical_values(self, temp_output_dir):
        """Test with physically reasonable EE spectrum values"""
        ells = np.logspace(0.5, 3.5, 100)
        
        # EE spectrum is typically smaller than TT
        Cee = 1e-12 * (ells / 100)**(-2) * np.exp(-(ells / 800)**2)
        
        output_path = temp_output_dir / "test_ee_physical.png"
        plot_EE(ells, Cee, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 1000
    
    def test_scaling_behavior(self, temp_output_dir):
        """Test that EE plot handles different amplitude scales"""
        ells = np.logspace(1, 3, 50)
        
        # Test different amplitude scales
        scales = [1e-15, 1e-12, 1e-9]
        
        for i, scale in enumerate(scales):
            Cee = scale * (ells / 100)**(-1.5)
            output_path = temp_output_dir / f"test_ee_scale_{i}.png"
            
            plot_EE(ells, Cee, str(output_path))
            assert output_path.exists()
    
    def test_no_display(self, temp_output_dir):
        """Test that plots are created without displaying"""
        ells = np.array([2, 10, 50, 100])
        Cee = np.array([1e-12, 5e-12, 2e-12, 1e-12])
        
        output_path = temp_output_dir / "test_ee_no_display.png"
        
        plot_EE(ells, Cee, str(output_path))
        
        # Check that no figures are left open
        assert len(plt.get_fignums()) == 0


class TestPlotTE:
    """Test suite for plot_TE function"""
    
    def test_basic_functionality(self, temp_output_dir):
        """Test basic plot generation for TE spectrum"""
        ells = np.array([2, 10, 50, 100, 500, 1000])
        Cte = np.array([1e-11, -5e-12, 2e-12, -1e-12, 5e-13, -2e-13])
        
        output_path = temp_output_dir / "test_te.png"
        
        plot_TE(ells, Cte, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_positive_and_negative_values(self, temp_output_dir):
        """Test TE spectrum with both positive and negative values"""
        ells = np.logspace(1, 3, 100)
        
        # TE spectrum oscillates between positive and negative
        Cte = 1e-12 * np.sin(ells / 50) * (ells / 100)**(-1.5)
        
        output_path = temp_output_dir / "test_te_oscillating.png"
        plot_TE(ells, Cte, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 1000
    
    def test_zero_crossing(self, temp_output_dir):
        """Test TE spectrum that crosses zero"""
        ells = np.array([10, 50, 100, 200, 500])
        Cte = np.array([1e-12, -5e-13, 0.0, 3e-13, -1e-13])
        
        output_path = temp_output_dir / "test_te_zero.png"
        plot_TE(ells, Cte, str(output_path))
        
        assert output_path.exists()
    
    def test_all_positive_values(self, temp_output_dir):
        """Test TE spectrum with all positive values"""
        ells = np.array([2, 10, 50, 100])
        Cte = np.array([1e-12, 5e-12, 2e-12, 1e-12])
        
        output_path = temp_output_dir / "test_te_positive.png"
        plot_TE(ells, Cte, str(output_path))
        
        assert output_path.exists()
    
    def test_all_negative_values(self, temp_output_dir):
        """Test TE spectrum with all negative values"""
        ells = np.array([2, 10, 50, 100])
        Cte = np.array([-1e-12, -5e-12, -2e-12, -1e-12])
        
        output_path = temp_output_dir / "test_te_negative.png"
        plot_TE(ells, Cte, str(output_path))
        
        assert output_path.exists()
    
    def test_sign_handling(self, temp_output_dir):
        """Test that sign handling works correctly"""
        ells = np.array([10, 50, 100, 200])
        
        # Test with very small values near zero
        Cte_small = np.array([1e-20, -1e-20, 1e-21, -1e-21])
        
        output_path = temp_output_dir / "test_te_small_signs.png"
        plot_TE(ells, Cte_small, str(output_path))
        
        assert output_path.exists()
    
    def test_different_output_formats(self, temp_output_dir):
        """Test different output file formats"""
        ells = np.logspace(0.5, 3, 50)
        Cte = 1e-12 * np.cos(ells / 30) * (ells / 100)**(-1.5)
        
        formats = ['png', 'pdf', 'svg']
        
        for fmt in formats:
            output_path = temp_output_dir / f"test_te.{fmt}"
            
            try:
                plot_TE(ells, Cte, str(output_path))
                assert output_path.exists()
                assert output_path.stat().st_size > 0
            except Exception as e:
                if "format" not in str(e).lower():
                    raise
    
    def test_no_display(self, temp_output_dir):
        """Test that plots are created without displaying"""
        ells = np.array([2, 10, 50, 100])
        Cte = np.array([1e-12, -5e-12, 2e-12, -1e-12])
        
        output_path = temp_output_dir / "test_te_no_display.png"
        
        plot_TE(ells, Cte, str(output_path))
        
        # Check that no figures are left open
        assert len(plt.get_fignums()) == 0


class TestPlottingIntegration:
    """Integration tests for plotting functions"""
    
    def test_all_spectra_together(self, temp_output_dir):
        """Test generating all three types of spectra"""
        ells = np.logspace(0.5, 3, 100)
        
        # Generate realistic-looking spectra
        Ctt = 1e-10 * (ells / 100)**(-2) * np.exp(-(ells / 1000)**2)
        Cee = 0.1 * Ctt  # EE typically smaller than TT
        Cte = 0.5 * np.sqrt(Ctt * Cee) * np.cos(ells / 50)  # Correlated
        
        # Generate all plots
        plot_TT(ells, Ctt, str(temp_output_dir / "integration_tt.png"))
        plot_EE(ells, Cee, str(temp_output_dir / "integration_ee.png"))
        plot_TE(ells, Cte, str(temp_output_dir / "integration_te.png"))
        
        # Check all files exist
        assert (temp_output_dir / "integration_tt.png").exists()
        assert (temp_output_dir / "integration_ee.png").exists()
        assert (temp_output_dir / "integration_te.png").exists()
        
        # Check no figures left open
        assert len(plt.get_fignums()) == 0
    
    def test_consistent_ell_ranges(self, temp_output_dir):
        """Test that all plotting functions handle the same ell ranges"""
        ell_ranges = [
            np.array([2, 10, 100]),  # Minimal
            np.arange(2, 101),       # Dense low-ell
            np.logspace(0.5, 3.5, 50),  # Log-spaced
            np.arange(2, 2001, 10)   # High-ell
        ]
        
        for i, ells in enumerate(ell_ranges):
            # Generate appropriate spectra
            Ctt = 1e-10 * (ells / 100)**(-2)
            Cee = 1e-12 * (ells / 100)**(-2)
            Cte = 1e-12 * np.sin(ells / 50) * (ells / 100)**(-1.5)
            
            # Test all plotting functions
            plot_TT(ells, Ctt, str(temp_output_dir / f"range_tt_{i}.png"))
            plot_EE(ells, Cee, str(temp_output_dir / f"range_ee_{i}.png"))
            plot_TE(ells, Cte, str(temp_output_dir / f"range_te_{i}.png"))
            
            # Verify files created
            assert (temp_output_dir / f"range_tt_{i}.png").exists()
            assert (temp_output_dir / f"range_ee_{i}.png").exists()
            assert (temp_output_dir / f"range_te_{i}.png").exists()
    
    def test_matplotlib_cleanup(self, temp_output_dir):
        """Test that matplotlib resources are properly cleaned up"""
        ells = np.array([2, 10, 50, 100])
        Ctt = np.array([1e-10, 5e-10, 2e-10, 1e-10])
        
        # Check initial state
        initial_figs = len(plt.get_fignums())
        
        # Generate multiple plots
        for i in range(5):
            plot_TT(ells, Ctt, str(temp_output_dir / f"cleanup_test_{i}.png"))
        
        # Should not accumulate figures
        final_figs = len(plt.get_fignums())
        assert final_figs == initial_figs
    
    def test_concurrent_plotting(self, temp_output_dir):
        """Test that plotting functions can be called in sequence without interference"""
        ells = np.array([2, 10, 50, 100, 500])
        
        # Different spectra
        Ctt = np.array([1e-10, 5e-10, 2e-10, 1e-10, 5e-11])
        Cee = np.array([1e-12, 5e-12, 2e-12, 1e-12, 5e-13])
        Cte = np.array([1e-12, -5e-12, 2e-12, -1e-12, 5e-13])
        
        # Interleave plotting calls
        plot_TT(ells, Ctt, str(temp_output_dir / "concurrent_tt1.png"))
        plot_EE(ells, Cee, str(temp_output_dir / "concurrent_ee1.png"))
        plot_TE(ells, Cte, str(temp_output_dir / "concurrent_te1.png"))
        plot_TT(ells, Ctt * 2, str(temp_output_dir / "concurrent_tt2.png"))
        plot_EE(ells, Cee * 2, str(temp_output_dir / "concurrent_ee2.png"))
        plot_TE(ells, Cte * 2, str(temp_output_dir / "concurrent_te2.png"))
        
        # Check all files exist
        for spec in ['tt', 'ee', 'te']:
            for i in [1, 2]:
                assert (temp_output_dir / f"concurrent_{spec}{i}.png").exists()
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, temp_output_dir):
        """Test plotting performance with large datasets"""
        # Large ell range
        ells = np.arange(2, 5001)
        Ctt = 1e-10 * (ells / 100)**(-2) * np.exp(-(ells / 2000)**2)
        
        output_path = temp_output_dir / "large_dataset.png"
        
        # Should complete without timeout or memory issues
        plot_TT(ells, Ctt, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 1000
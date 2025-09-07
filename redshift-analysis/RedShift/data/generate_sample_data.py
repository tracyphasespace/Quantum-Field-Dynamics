#!/usr/bin/env python3
"""
Sample Data Generation Script for QFD CMB Module

This script generates minimal test datasets and sample Planck-like data
for demonstration and validation purposes.
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path

# Import QFD CMB modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
from qfd_cmb.kernels import te_correlation_phase


class SampleDataGenerator:
    """Class for generating various types of sample data."""
    
    def __init__(self, output_dir="data/sample"):
        """Initialize the sample data generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard Planck-anchored parameters
        self.planck_params = {
            'lA': 301.0,
            'rpsi': 147.0,
            'chi_star': 14065.0,  # lA * rpsi / pi
            'sigma_chi': 250.0,
            'ns': 0.96,
            'Aosc': 0.55,
            'sigma_osc': 0.025
        }
        
        print(f"Sample data generator initialized")
        print(f"Output directory: {self.output_dir}")
    
    def generate_minimal_test_data(self):
        """Generate minimal test datasets for quick validation."""
        print("\nGenerating minimal test datasets...")
        
        # Small multipole range for fast computation
        ells_minimal = np.arange(2, 101)  # ell = 2 to 100
        
        # Setup visibility window
        chi_star = self.planck_params['chi_star']
        sigma_chi = self.planck_params['sigma_chi']
        chi_grid = np.linspace(chi_star - 3*sigma_chi, chi_star + 3*sigma_chi, 100)
        W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Define power spectrum function
        def Pk_minimal(k):
            return oscillatory_psik(
                k, 
                ns=self.planck_params['ns'],
                rpsi=self.planck_params['rpsi'],
                Aosc=self.planck_params['Aosc'],
                sigma_osc=self.planck_params['sigma_osc']
            )
        
        # Compute spectra
        Ctt_minimal = project_limber(ells_minimal, Pk_minimal, W_chi, chi_grid)
        Cee_minimal = 0.25 * Ctt_minimal  # Simplified EE model
        
        # Compute TE with correlation
        rho_minimal = np.array([
            te_correlation_phase((ell + 0.5)/chi_star, self.planck_params['rpsi'], ell, chi_star)
            for ell in ells_minimal
        ])
        Cte_minimal = rho_minimal * np.sqrt(Ctt_minimal * Cee_minimal)
        
        # Create DataFrame
        df_minimal = pd.DataFrame({
            'ell': ells_minimal,
            'C_TT': Ctt_minimal,
            'C_EE': Cee_minimal,
            'C_TE': Cte_minimal,
            'rho_TE': rho_minimal
        })
        
        # Save to CSV
        output_file = self.output_dir / "minimal_test_spectra.csv"
        df_minimal.to_csv(output_file, index=False, float_format='%.6e')
        
        print(f"  Saved minimal test data: {output_file}")
        print(f"  Multipole range: {ells_minimal.min()} to {ells_minimal.max()}")
        print(f"  Data points: {len(ells_minimal)}")
        
        return df_minimal
    
    def generate_planck_like_data(self):
        """Generate sample Planck-like data for demonstration."""
        print("\nGenerating Planck-like demonstration data...")
        
        # Planck-like multipole range
        ells_planck = np.arange(2, 2501)  # ell = 2 to 2500
        
        # Setup visibility window with higher resolution
        chi_star = self.planck_params['chi_star']
        sigma_chi = self.planck_params['sigma_chi']
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 500)
        W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        # Define power spectrum function
        def Pk_planck(k):
            return oscillatory_psik(
                k,
                ns=self.planck_params['ns'],
                rpsi=self.planck_params['rpsi'],
                Aosc=self.planck_params['Aosc'],
                sigma_osc=self.planck_params['sigma_osc']
            )
        
        # Compute TT spectrum
        print("  Computing TT spectrum...")
        Ctt_planck = project_limber(ells_planck, Pk_planck, W_chi, chi_grid)
        
        # Compute EE spectrum (more realistic model)
        print("  Computing EE spectrum...")
        def Pk_EE(k):
            # EE has different amplitude and scale dependence
            base_spectrum = oscillatory_psik(k, ns=self.planck_params['ns'])
            ee_factor = 0.25 * (1 + 0.1 * np.sin(k * self.planck_params['rpsi'] * 0.5))
            return ee_factor * base_spectrum
        
        Cee_planck = project_limber(ells_planck, Pk_EE, W_chi, chi_grid)
        
        # Compute TE spectrum with correlation
        print("  Computing TE spectrum...")
        rho_planck = np.array([
            te_correlation_phase((ell + 0.5)/chi_star, self.planck_params['rpsi'], ell, chi_star)
            for ell in ells_planck
        ])
        Cte_planck = rho_planck * np.sqrt(Ctt_planck * Cee_planck)
        
        # Add realistic noise levels (approximate Planck sensitivity)
        noise_level_tt = 0.02  # 2% relative error
        noise_level_ee = 0.05  # 5% relative error  
        noise_level_te = 0.03  # 3% relative error
        
        # Generate error estimates
        error_tt = noise_level_tt * Ctt_planck
        error_ee = noise_level_ee * Cee_planck
        error_te = noise_level_te * np.abs(Cte_planck)
        
        # Create DataFrame
        df_planck = pd.DataFrame({
            'ell': ells_planck,
            'C_TT': Ctt_planck,
            'C_EE': Cee_planck,
            'C_TE': Cte_planck,
            'error_TT': error_tt,
            'error_EE': error_ee,
            'error_TE': error_te,
            'rho_TE': rho_planck
        })
        
        # Save to CSV
        output_file = self.output_dir / "planck_like_spectra.csv"
        df_planck.to_csv(output_file, index=False, float_format='%.6e')
        
        print(f"  Saved Planck-like data: {output_file}")
        print(f"  Multipole range: {ells_planck.min()} to {ells_planck.max()}")
        print(f"  Data points: {len(ells_planck)}")
        
        return df_planck
    
    def generate_parameter_variations(self):
        """Generate data for different parameter combinations."""
        print("\nGenerating parameter variation datasets...")
        
        # Parameter variations to explore
        variations = {
            'rpsi_variations': {
                'base_params': self.planck_params.copy(),
                'vary_param': 'rpsi',
                'values': [130.0, 147.0, 165.0],
                'description': 'Oscillation scale variations'
            },
            'aosc_variations': {
                'base_params': self.planck_params.copy(),
                'vary_param': 'Aosc',
                'values': [0.0, 0.3, 0.55, 0.8],
                'description': 'Oscillation amplitude variations'
            },
            'ns_variations': {
                'base_params': self.planck_params.copy(),
                'vary_param': 'ns',
                'values': [0.94, 0.96, 0.98],
                'description': 'Spectral index variations'
            }
        }
        
        # Multipole range for variations
        ells_var = np.arange(2, 1001)  # ell = 2 to 1000
        
        # Setup visibility window
        chi_star = self.planck_params['chi_star']
        sigma_chi = self.planck_params['sigma_chi']
        chi_grid = np.linspace(chi_star - 4*sigma_chi, chi_star + 4*sigma_chi, 300)
        W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        for var_name, var_config in variations.items():
            print(f"  Generating {var_config['description']}...")
            
            results = []
            
            for value in var_config['values']:
                # Update parameter
                params = var_config['base_params'].copy()
                params[var_config['vary_param']] = value
                
                # Define power spectrum
                def Pk_var(k):
                    return oscillatory_psik(
                        k,
                        ns=params['ns'],
                        rpsi=params['rpsi'],
                        Aosc=params['Aosc'],
                        sigma_osc=params['sigma_osc']
                    )
                
                # Compute spectrum
                Ctt_var = project_limber(ells_var, Pk_var, W_chi, chi_grid)
                
                # Store results
                for i, ell in enumerate(ells_var):
                    results.append({
                        'ell': ell,
                        'C_TT': Ctt_var[i],
                        var_config['vary_param']: value,
                        'parameter_set': f"{var_config['vary_param']}_{value}"
                    })
            
            # Create DataFrame and save
            df_var = pd.DataFrame(results)
            output_file = self.output_dir / f"{var_name}.csv"
            df_var.to_csv(output_file, index=False, float_format='%.6e')
            
            print(f"    Saved: {output_file}")
        
        print(f"  Generated {len(variations)} parameter variation datasets")
    
    def generate_reference_data(self):
        """Generate reference data for regression testing."""
        print("\nGenerating reference data for regression testing...")
        
        # Fixed parameters for reproducible reference
        ref_params = {
            'ns': 0.96,
            'rpsi': 147.0,
            'Aosc': 0.55,
            'sigma_osc': 0.025,
            'chi_star': 14065.0,
            'sigma_chi': 250.0
        }
        
        # Reference multipole points
        ells_ref = np.array([2, 10, 50, 100, 200, 500, 1000, 1500, 2000])
        
        # Setup calculation
        chi_grid = np.linspace(ref_params['chi_star'] - 5*ref_params['sigma_chi'],
                              ref_params['chi_star'] + 5*ref_params['sigma_chi'], 400)
        W_chi = gaussian_window_chi(chi_grid, ref_params['chi_star'], ref_params['sigma_chi'])
        
        # Compute reference spectra
        def Pk_ref(k):
            return oscillatory_psik(
                k,
                ns=ref_params['ns'],
                rpsi=ref_params['rpsi'],
                Aosc=ref_params['Aosc'],
                sigma_osc=ref_params['sigma_osc']
            )
        
        Ctt_ref = project_limber(ells_ref, Pk_ref, W_chi, chi_grid)
        Cee_ref = 0.25 * Ctt_ref
        
        rho_ref = np.array([
            te_correlation_phase((ell + 0.5)/ref_params['chi_star'], ref_params['rpsi'], 
                               ell, ref_params['chi_star'])
            for ell in ells_ref
        ])
        Cte_ref = rho_ref * np.sqrt(Ctt_ref * Cee_ref)
        
        # Create reference data structure
        reference_data = {
            'metadata': {
                'description': 'Reference data for QFD CMB Module regression testing',
                'parameters': ref_params,
                'computation_settings': {
                    'chi_grid_points': len(chi_grid),
                    'chi_range_factor': 5.0
                },
                'tolerances': {
                    'relative': 1e-10,
                    'absolute': 1e-12
                }
            },
            'data': {
                'ell': ells_ref.tolist(),
                'C_TT': Ctt_ref.tolist(),
                'C_EE': Cee_ref.tolist(),
                'C_TE': Cte_ref.tolist(),
                'rho_TE': rho_ref.tolist()
            }
        }
        
        # Save as JSON for easy loading in tests
        output_file = self.output_dir / "reference_spectra.json"
        with open(output_file, 'w') as f:
            json.dump(reference_data, f, indent=2)
        
        print(f"  Saved reference data: {output_file}")
        print(f"  Reference multipoles: {ells_ref}")
        
        return reference_data
    
    def generate_metadata(self):
        """Generate metadata and documentation for all datasets."""
        print("\nGenerating dataset metadata...")
        
        metadata = {
            'qfd_cmb_sample_data': {
                'version': '1.0.0',
                'description': 'Sample datasets for QFD CMB Module demonstration and testing',
                'generated_by': 'generate_sample_data.py',
                'planck_parameters': self.planck_params,
                'datasets': {
                    'minimal_test_spectra.csv': {
                        'description': 'Minimal test dataset for quick validation',
                        'multipole_range': [2, 100],
                        'data_points': 99,
                        'columns': ['ell', 'C_TT', 'C_EE', 'C_TE', 'rho_TE'],
                        'use_case': 'Fast unit testing and basic validation'
                    },
                    'planck_like_spectra.csv': {
                        'description': 'Planck-like dataset for realistic demonstrations',
                        'multipole_range': [2, 2500],
                        'data_points': 2499,
                        'columns': ['ell', 'C_TT', 'C_EE', 'C_TE', 'error_TT', 'error_EE', 'error_TE', 'rho_TE'],
                        'use_case': 'Realistic parameter fitting and analysis examples'
                    },
                    'reference_spectra.json': {
                        'description': 'Reference data for regression testing',
                        'multipole_points': [2, 10, 50, 100, 200, 500, 1000, 1500, 2000],
                        'data_points': 9,
                        'format': 'JSON with metadata',
                        'use_case': 'Automated regression testing'
                    },
                    'parameter_variations/': {
                        'description': 'Parameter variation studies',
                        'files': ['rpsi_variations.csv', 'aosc_variations.csv', 'ns_variations.csv'],
                        'use_case': 'Parameter sensitivity analysis'
                    }
                },
                'data_format': {
                    'ell': 'Multipole moment (integer)',
                    'C_TT': 'TT angular power spectrum (dimensionless)',
                    'C_EE': 'EE angular power spectrum (dimensionless)',
                    'C_TE': 'TE angular power spectrum (dimensionless)',
                    'error_TT': 'TT spectrum uncertainty (dimensionless)',
                    'error_EE': 'EE spectrum uncertainty (dimensionless)',
                    'error_TE': 'TE spectrum uncertainty (dimensionless)',
                    'rho_TE': 'TE correlation coefficient (dimensionless)'
                },
                'physical_units': {
                    'note': 'All spectra are dimensionless and normalized',
                    'conversion': 'Multiply by appropriate factors for physical units'
                }
            }
        }
        
        # Save metadata
        output_file = self.output_dir / "dataset_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata: {output_file}")
        
        return metadata


def main():
    """Main function to generate all sample datasets."""
    parser = argparse.ArgumentParser(description='Generate sample data for QFD CMB Module')
    parser.add_argument('--output-dir', default='data/sample',
                       help='Output directory for sample data (default: data/sample)')
    parser.add_argument('--minimal-only', action='store_true',
                       help='Generate only minimal test data (faster)')
    parser.add_argument('--no-variations', action='store_true',
                       help='Skip parameter variation datasets')
    
    args = parser.parse_args()
    
    print("QFD CMB Module - Sample Data Generation")
    print("=" * 50)
    
    # Initialize generator
    generator = SampleDataGenerator(args.output_dir)
    
    # Generate datasets
    try:
        # Always generate minimal test data
        generator.generate_minimal_test_data()
        
        if not args.minimal_only:
            # Generate full datasets
            generator.generate_planck_like_data()
            generator.generate_reference_data()
            
            if not args.no_variations:
                generator.generate_parameter_variations()
        
        # Generate metadata
        generator.generate_metadata()
        
        print("\n" + "=" * 50)
        print("Sample data generation completed successfully!")
        print(f"Output directory: {generator.output_dir}")
        print("\nGenerated files:")
        
        # List generated files
        for file_path in sorted(generator.output_dir.rglob('*')):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.relative_to(generator.output_dir)} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"\nError during data generation: {e}")
        raise


if __name__ == "__main__":
    main()
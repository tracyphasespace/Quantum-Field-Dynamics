#!/usr/bin/env python3
"""
qfd_predictions_framework.py

Framework for testing unique, falsifiable QFD predictions that ΛCDM doesn't naturally create:

1. Phase-colour law: Δμ_plasma(t,λ) ∝ (1-e^(-t/τ)) * (λ_B/λ)^β
2. Flux-dependence (FDR): Dimming scales as ~(1/D) + (1/D³)
3. Cross-SN generalization: Parameters fitted on subset predict held-out SNe

This provides the framework - specific tests can be added as needed.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Constants
C_KMS = 299792.458  # km/s
AB_ZP_JY = 3631.0   # AB magnitude zero point in Jy
LAMBDA_B = 440.0    # Reference wavelength (nm) for β scaling

class QFDPredictionsAnalyzer:
    """Framework for testing unique QFD predictions."""

    def __init__(self, verbose: bool = False):
        self.setup_logging(verbose)

    def setup_logging(self, verbose: bool):
        """Configure logging."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_lightcurve_data(self, file_path: str, snid: str) -> pd.DataFrame:
        """Load light curve data for specific supernova."""
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        # Filter for specific SN
        df = df[df['snid'] == snid].copy()

        if len(df) == 0:
            raise ValueError(f"No data found for supernova {snid}")

        # Quality cuts
        df = df.dropna(subset=['mjd', 'mag', 'band'])
        df = df[df['mag'] > 0]

        logging.info(f"Loaded {len(df)} points for {snid}")
        return df.sort_values(['band', 'mjd']).reset_index(drop=True)

    def estimate_t_max(self, df: pd.DataFrame) -> float:
        """Estimate time of maximum brightness."""
        optical_bands = ['g', 'r', 'i', 'v', 'b']
        optical_data = df[df['band'].isin(optical_bands)]

        if len(optical_data) > 0:
            brightest_idx = optical_data['mag'].idxmin()
            t_max = optical_data.loc[brightest_idx, 'mjd']
        else:
            brightest_idx = df['mag'].idxmin()
            t_max = df.loc[brightest_idx, 'mjd']

        return t_max

    def calculate_phase_and_wavelength(self, df: pd.DataFrame, t_max: float) -> pd.DataFrame:
        """Add phase and rest-frame wavelength columns."""
        df = df.copy()
        df['phase'] = df['mjd'] - t_max
        # Assume wavelength_eff_nm exists or use band-based estimates
        if 'wavelength_eff_nm' not in df.columns:
            # Basic wavelength mapping
            wl_map = {
                'u': 365, 'b': 445, 'v': 551, 'g': 477, 'r': 623,
                'i': 763, 'z': 905, 'y': 1020, 'j': 1220, 'h': 1630, 'k': 2190
            }
            df['wavelength_eff_nm'] = df['band'].map(lambda b: wl_map.get(b.lower(), 500))

        return df

    def test_phase_colour_law(self, df: pd.DataFrame, t_max: float,
                             qfd_params: Dict) -> Dict:
        """
        Test 1: Phase-colour law
        QFD predicts: Δμ_plasma(t,λ) ∝ (1-e^(-t/τ)) * (λ_B/λ)^β
        """
        df = self.calculate_phase_and_wavelength(df, t_max)

        # Extract QFD parameters
        A_plasma = qfd_params.get('A_plasma', 0.1)
        tau_decay = qfd_params.get('tau_decay_days', 68.0)
        beta = qfd_params.get('beta', 1.0)

        # Calculate predicted QFD effect
        def qfd_effect(phase, wavelength):
            if phase <= 0:
                return 0.0
            temporal = 1.0 - np.exp(-phase / tau_decay)
            spectral = (LAMBDA_B / wavelength) ** beta
            return A_plasma * temporal * spectral

        df['qfd_effect_pred'] = df.apply(
            lambda row: qfd_effect(row['phase'], row['wavelength_eff_nm']),
            axis=1
        )

        # Calculate residuals from simple model (proxy for what QFD should explain)
        # For now, use deviations from median magnitude per band
        band_medians = df.groupby('band')['mag'].median()
        df['simple_residual'] = df.apply(
            lambda row: row['mag'] - band_medians[row['band']],
            axis=1
        )

        # Test correlations
        # Phase correlation
        phase_mask = (df['phase'] > 0) & (df['phase'] < 100)  # Post-max, pre-tail
        if np.sum(phase_mask) > 10:
            phase_corr, phase_p = pearsonr(
                df.loc[phase_mask, 'qfd_effect_pred'],
                df.loc[phase_mask, 'simple_residual']
            )
        else:
            phase_corr, phase_p = np.nan, np.nan

        # Wavelength correlation
        wl_corr, wl_p = pearsonr(df['qfd_effect_pred'], df['simple_residual'])

        # Combined correlation
        combined_corr, combined_p = pearsonr(df['qfd_effect_pred'], df['simple_residual'])

        return {
            'test_name': 'phase_colour_law',
            'n_points': len(df),
            'phase_correlation': {
                'r': float(phase_corr) if not np.isnan(phase_corr) else None,
                'p_value': float(phase_p) if not np.isnan(phase_p) else None,
                'n_points': int(np.sum(phase_mask))
            },
            'wavelength_correlation': {
                'r': float(wl_corr),
                'p_value': float(wl_p),
                'n_points': len(df)
            },
            'combined_correlation': {
                'r': float(combined_corr),
                'p_value': float(combined_p),
                'interpretation': self._interpret_correlation(combined_corr, combined_p)
            }
        }

    def test_flux_dependence(self, df: pd.DataFrame, distance_proxy: str = 'mag') -> Dict:
        """
        Test 2: Flux-dependence (FDR)
        QFD predicts dimming scales as ~(1/D) + (1/D³)
        """
        # Use magnitude as distance proxy (brighter = closer)
        # This is a simplified test - in practice would need actual distances

        df = df.copy()

        # Bin by brightness (proxy for distance)
        df_sorted = df.sort_values(distance_proxy)
        n_bins = min(5, len(df) // 20)  # At least 20 points per bin

        if n_bins < 3:
            return {
                'test_name': 'flux_dependence',
                'status': 'insufficient_data',
                'n_points': len(df)
            }

        df['distance_bin'] = pd.cut(df[distance_proxy], n_bins, labels=False)

        # Calculate residuals per bin
        bin_stats = []
        for bin_id in range(n_bins):
            bin_data = df[df['distance_bin'] == bin_id]
            if len(bin_data) > 5:
                median_mag = bin_data['mag'].median()
                residuals = bin_data['mag'] - median_mag
                bin_stats.append({
                    'bin_id': bin_id,
                    'median_mag': median_mag,
                    'residual_std': residuals.std(),
                    'n_points': len(bin_data)
                })

        # Test for trend in residual scatter vs distance
        if len(bin_stats) >= 3:
            mags = [bs['median_mag'] for bs in bin_stats]
            stds = [bs['residual_std'] for bs in bin_stats]
            trend_corr, trend_p = pearsonr(mags, stds)
        else:
            trend_corr, trend_p = np.nan, np.nan

        return {
            'test_name': 'flux_dependence',
            'n_bins': n_bins,
            'bin_statistics': bin_stats,
            'trend_correlation': {
                'r': float(trend_corr) if not np.isnan(trend_corr) else None,
                'p_value': float(trend_p) if not np.isnan(trend_p) else None,
                'interpretation': self._interpret_correlation(trend_corr, trend_p)
            }
        }

    def test_cross_sn_generalization(self, data_dict: Dict[str, pd.DataFrame],
                                   qfd_params_dict: Dict[str, Dict]) -> Dict:
        """
        Test 3: Cross-SN generalization
        If QFD captures real physics, parameters should generalize across SNe
        """
        sn_ids = list(data_dict.keys())

        if len(sn_ids) < 2:
            return {
                'test_name': 'cross_sn_generalization',
                'status': 'insufficient_supernovae',
                'n_supernovae': len(sn_ids)
            }

        # Split SNe into train/test
        train_sns, test_sns = train_test_split(sn_ids, test_size=0.3, random_state=42)

        # Extract parameters from training set
        train_params = []
        for sn_id in train_sns:
            if sn_id in qfd_params_dict:
                params = qfd_params_dict[sn_id]
                train_params.append([
                    params.get('A_plasma', 0.1),
                    params.get('tau_decay_days', 68.0),
                    params.get('beta', 1.0)
                ])

        if len(train_params) == 0:
            return {
                'test_name': 'cross_sn_generalization',
                'status': 'no_training_parameters'
            }

        # Calculate median parameters from training set
        train_params = np.array(train_params)
        median_params = {
            'A_plasma': np.median(train_params[:, 0]),
            'tau_decay_days': np.median(train_params[:, 1]),
            'beta': np.median(train_params[:, 2])
        }

        # Test generalization on test set
        generalization_scores = []
        for sn_id in test_sns:
            if sn_id in data_dict and sn_id in qfd_params_dict:
                # Calculate prediction error using median parameters
                test_params = qfd_params_dict[sn_id]

                # Simple prediction score: correlation between predicted and actual parameters
                param_diff = np.abs([
                    test_params.get('A_plasma', 0.1) - median_params['A_plasma'],
                    test_params.get('tau_decay_days', 68.0) - median_params['tau_decay_days'],
                    test_params.get('beta', 1.0) - median_params['beta']
                ])

                # Normalized prediction score (lower is better)
                prediction_score = np.mean(param_diff / np.array([0.1, 68.0, 1.0]))  # Normalize by typical scales

                generalization_scores.append({
                    'sn_id': sn_id,
                    'prediction_score': float(prediction_score)
                })

        return {
            'test_name': 'cross_sn_generalization',
            'train_supernovae': train_sns,
            'test_supernovae': test_sns,
            'median_parameters': median_params,
            'generalization_scores': generalization_scores,
            'mean_generalization': np.mean([gs['prediction_score'] for gs in generalization_scores])
        }

    def _interpret_correlation(self, r: float, p: float) -> str:
        """Interpret correlation strength and significance."""
        if np.isnan(r) or np.isnan(p):
            return "insufficient_data"

        # Significance
        if p > 0.05:
            significance = "not_significant"
        elif p > 0.01:
            significance = "marginally_significant"
        else:
            significance = "significant"

        # Strength
        abs_r = abs(r)
        if abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.6:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "positive" if r > 0 else "negative"

        return f"{strength}_{direction}_{significance}"

    def run_comprehensive_analysis(self, data_file: str, snid: str,
                                 qfd_params: Dict, outdir: str) -> Dict:
        """Run all QFD prediction tests on a single supernova."""
        os.makedirs(outdir, exist_ok=True)

        # Load data
        df = self.load_lightcurve_data(data_file, snid)
        t_max = self.estimate_t_max(df)

        # Run tests
        results = {
            'supernova': snid,
            't_max': float(t_max),
            'n_total_points': len(df),
            'qfd_parameters': qfd_params,
            'tests': {}
        }

        # Test 1: Phase-colour law
        logging.info("Running phase-colour law test...")
        results['tests']['phase_colour'] = self.test_phase_colour_law(df, t_max, qfd_params)

        # Test 2: Flux dependence
        logging.info("Running flux dependence test...")
        results['tests']['flux_dependence'] = self.test_flux_dependence(df)

        # Save results
        output_file = os.path.join(outdir, f'{snid}_qfd_predictions.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to: {output_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description="QFD unique predictions framework")
    parser.add_argument("--data", required=True, help="Path to light curve data")
    parser.add_argument("--snid", required=True, help="Supernova identifier")
    parser.add_argument("--outdir", default="./qfd_predictions", help="Output directory")
    parser.add_argument("--A-plasma", type=float, default=0.1, help="QFD A_plasma parameter")
    parser.add_argument("--tau-decay", type=float, default=68.0, help="QFD tau_decay parameter (days)")
    parser.add_argument("--beta", type=float, default=1.0, help="QFD beta parameter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup QFD parameters
    qfd_params = {
        'A_plasma': args.A_plasma,
        'tau_decay_days': args.tau_decay,
        'beta': args.beta
    }

    # Run analysis
    analyzer = QFDPredictionsAnalyzer(verbose=args.verbose)

    try:
        results = analyzer.run_comprehensive_analysis(
            args.data, args.snid, qfd_params, args.outdir
        )

        print(f"\n=== QFD Predictions Analysis for {args.snid} ===")

        # Summary
        phase_test = results['tests']['phase_colour']
        if phase_test['combined_correlation']['r'] is not None:
            print(f"Phase-Colour Law: r = {phase_test['combined_correlation']['r']:.3f}")
            print(f"  Interpretation: {phase_test['combined_correlation']['interpretation']}")

        flux_test = results['tests']['flux_dependence']
        if 'trend_correlation' in flux_test and flux_test['trend_correlation']['r'] is not None:
            print(f"Flux Dependence: r = {flux_test['trend_correlation']['r']:.3f}")
            print(f"  Interpretation: {flux_test['trend_correlation']['interpretation']}")

        print(f"\nDetailed results saved to: {args.outdir}/")

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
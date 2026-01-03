#!/usr/bin/env python3
"""
Two-Center Harmonic Model for Deformed Nuclei (A > 161)

Extends single-center spherical model to account for prolate/oblate deformation.
Physical basis: Soliton topology phase transition at A ≈ 161 (rare earth region).

Author: Tracy McSheery
Date: 2026-01-02
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import json

class TwoCenterHarmonicModel:
    """
    Two-center extension of harmonic family model for deformed nuclei.

    Key features:
    - Deformation parameter β estimation
    - Modified Z_pred formula accounting for ellipsoid geometry
    - Epsilon calculation for two-center resonance modes
    - Validation against half-life correlation
    """

    def __init__(self, params_single_center=None):
        """
        Initialize two-center model.

        Parameters
        ----------
        params_single_center : dict, optional
            Single-center parameters from fit to A ≤ 161
            If None, will need to be fitted
        """
        self.params_1c = params_single_center
        self.params_2c = None  # Fitted parameters for two-center
        self.beta_model = None  # Deformation predictor

    # ========================================================================
    # Deformation Parameter Estimation
    # ========================================================================

    @staticmethod
    def beta_empirical(A, Z):
        """
        Estimate deformation parameter β from empirical systematics.

        Based on measured β₂ values from nuclear data compilations.

        Parameters
        ----------
        A : float or array
            Mass number
        Z : float or array
            Proton number

        Returns
        -------
        beta : float or array
            Deformation parameter (0 = spherical, 0.3 = highly deformed)

        Notes
        -----
        Empirical systematics:
        - A < 150: Spherical (β ≈ 0)
        - 150 < A < 170: Transition (β ≈ 0 → 0.3, rare earths)
        - 170 < A < 190: Deformed (β ≈ 0.25-0.35)
        - 190 < A < 210: Transition to actinides
        - A > 210: Actinides (β ≈ 0.25-0.30)
        """
        A = np.asarray(A)
        Z = np.asarray(Z)

        beta = np.zeros_like(A, dtype=float)

        # Spherical region
        mask_spherical = (A < 150)
        beta[mask_spherical] = 0.0

        # Transition to deformation (rare earths)
        mask_transition1 = (A >= 150) & (A < 170)
        beta[mask_transition1] = 0.015 * (A[mask_transition1] - 150)

        # Deformed rare earths
        mask_deformed = (A >= 170) & (A < 190)
        beta[mask_deformed] = 0.25 + 0.005 * (A[mask_deformed] - 170)

        # Transition to actinides
        mask_transition2 = (A >= 190) & (A < 210)
        beta[mask_transition2] = 0.35 - 0.005 * (A[mask_transition2] - 190)

        # Actinides and beyond
        mask_actinides = (A >= 210)
        beta[mask_actinides] = 0.25 + 0.002 * (A[mask_actinides] - 210)

        # Refinement by Z (proton-rich → smaller β)
        N = A - Z
        asymmetry = (N - Z) / A  # Neutron excess
        beta_correction = 0.05 * asymmetry  # Neutron-rich nuclei more deformed

        beta = beta + beta_correction

        # Physical bounds
        beta = np.clip(beta, 0.0, 0.45)

        return beta if beta.shape else float(beta)

    @staticmethod
    def beta_from_residual(residual, A):
        """
        Estimate β from single-center fit residual.

        If single-center systematically underestimates Z, nucleus is likely deformed.

        Parameters
        ----------
        residual : float
            Z_obs - Z_pred_single_center
        A : float
            Mass number

        Returns
        -------
        beta : float
            Estimated deformation
        """
        # Heuristic: positive residual suggests prolate deformation
        # (single-center underestimates Z for elongated shape)
        beta_est = 0.01 * residual / A**(1/3) if residual > 0 else 0.0
        return np.clip(beta_est, 0.0, 0.45)

    # ========================================================================
    # Two-Center Formula
    # ========================================================================

    def Z_pred_single_center(self, A, N, family='A'):
        """Single-center prediction (baseline)"""
        if self.params_1c is None:
            raise ValueError("Single-center parameters not loaded")

        params = self.params_1c['families'][family]
        c1_0 = params['c1_0']
        dc1 = params['dc1']
        c2_0 = params['c2_0']
        dc2 = params['dc2']
        c3_0 = params['c3_0']
        dc3 = params['dc3']

        term1 = (c1_0 + N * dc1) * A**(2/3)
        term2 = (c2_0 + N * dc2) * A
        term3 = (c3_0 + N * dc3) * A**(4/3)

        return term1 + term2 + term3

    def Z_pred_two_center(self, A, N, beta, family='A'):
        """
        Two-center prediction with deformation correction.

        Approach: Effective radius correction
        R_eff = R₀(1 + β/3) for prolate ellipsoid

        Parameters
        ----------
        A : float
            Mass number
        N : int
            Harmonic mode index
        beta : float
            Deformation parameter
        family : str
            Harmonic family ('A', 'B', or 'C')

        Returns
        -------
        Z_pred : float
            Predicted proton number
        """
        if self.params_1c is None:
            raise ValueError("Single-center parameters not loaded")

        params = self.params_1c['families'][family]
        c1_0 = params['c1_0']
        dc1 = params['dc1']
        c2_0 = params['c2_0']
        dc2 = params['dc2']
        c3_0 = params['c3_0']
        dc3 = params['dc3']

        # Effective radius correction
        R_correction = (1 + beta/3)

        # Effective mass (volume conserved)
        A_eff_2_3 = A**(2/3) * R_correction**2
        A_eff_4_3 = A**(4/3) * R_correction**4

        term1 = (c1_0 + N * dc1) * A_eff_2_3
        term2 = (c2_0 + N * dc2) * A  # Bulk term unchanged
        term3 = (c3_0 + N * dc3) * A_eff_4_3

        return term1 + term2 + term3

    # ========================================================================
    # Epsilon Calculation
    # ========================================================================

    def epsilon_two_center(self, Z_obs, A, beta=None, family='A'):
        """
        Compute epsilon for deformed nucleus.

        Parameters
        ----------
        Z_obs : float
            Observed proton number
        A : float
            Mass number
        beta : float, optional
            Deformation parameter (if None, estimated from A, Z)
        family : str
            Harmonic family

        Returns
        -------
        epsilon : float
            Harmonic dissonance
        N_best : int
            Best harmonic mode
        beta_used : float
            Deformation used
        """
        # Estimate beta if not provided
        if beta is None:
            beta = self.beta_empirical(A, Z_obs)

        # Find best N by minimizing epsilon
        N_range = range(-5, 15)
        best_eps = 1.0
        best_N = 0

        for N in N_range:
            Z_pred = self.Z_pred_two_center(A, N, beta, family)
            residual = Z_obs - Z_pred

            # Compute N_hat (fractional mode number)
            params = self.params_1c['families'][family]
            dc3 = params['dc3']

            # Effective dc3 with deformation
            R_correction = (1 + beta/3)
            A_eff_4_3 = A**(4/3) * R_correction**4
            dc3_eff = dc3 * A_eff_4_3

            N_hat_frac = residual / dc3_eff if dc3_eff != 0 else 0
            eps = abs(N_hat_frac - round(N_hat_frac))

            if eps < best_eps:
                best_eps = eps
                best_N = N

        return best_eps, best_N, beta

    def score_all_families_two_center(self, Z_obs, A, beta=None):
        """
        Score nucleus across all three families, return best.

        Parameters
        ----------
        Z_obs : float
            Observed proton number
        A : float
            Mass number
        beta : float, optional
            Deformation parameter

        Returns
        -------
        results : dict
            Contains epsilon, N, family for best fit
        """
        families = ['A', 'B', 'C']
        results = {}

        for family in families:
            eps, N, beta_used = self.epsilon_two_center(Z_obs, A, beta, family)
            results[family] = {
                'epsilon': eps,
                'N': N,
                'beta': beta_used,
                'family': family
            }

        # Find best family (lowest epsilon)
        best_family = min(families, key=lambda f: results[f]['epsilon'])

        return {
            'epsilon_best': results[best_family]['epsilon'],
            'N_best': results[best_family]['N'],
            'best_family': best_family,
            'beta': results[best_family]['beta'],
            'epsilon_A': results['A']['epsilon'],
            'epsilon_B': results['B']['epsilon'],
            'epsilon_C': results['C']['epsilon'],
        }

    # ========================================================================
    # Bulk Scoring
    # ========================================================================

    def score_dataframe_two_center(self, df, A_threshold=161):
        """
        Score all nuclides in dataframe with two-center model.

        Uses single-center for A ≤ threshold, two-center for A > threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: A, Z, N
        A_threshold : float
            Mass number above which to use two-center

        Returns
        -------
        df_scored : pd.DataFrame
            Original df with additional columns:
            - epsilon_2c
            - N_best_2c
            - family_2c
            - beta_est
        """
        df_result = df.copy()

        # Initialize new columns
        df_result['epsilon_2c'] = np.nan
        df_result['N_best_2c'] = np.nan
        df_result['family_2c'] = ''
        df_result['beta_est'] = 0.0

        # Score heavy nuclides (A > threshold) with two-center
        mask_heavy = (df_result['A'] > A_threshold)

        for idx in df_result[mask_heavy].index:
            row = df_result.loc[idx]
            Z_obs = row['Z']
            A = row['A']

            result = self.score_all_families_two_center(Z_obs, A)

            df_result.loc[idx, 'epsilon_2c'] = result['epsilon_best']
            df_result.loc[idx, 'N_best_2c'] = result['N_best']
            df_result.loc[idx, 'family_2c'] = result['best_family']
            df_result.loc[idx, 'beta_est'] = result['beta']

        return df_result

    # ========================================================================
    # Validation
    # ========================================================================

    def validate_halflife_correlation(self, df_scored, A_range=(161, 250)):
        """
        Test if two-center epsilon correlates with half-life for heavy nuclides.

        Parameters
        ----------
        df_scored : pd.DataFrame
            Must contain: epsilon_best (single-center), epsilon_2c (two-center),
            half_life_s, is_stable, A
        A_range : tuple
            (A_min, A_max) for analysis

        Returns
        -------
        results : dict
            Correlation statistics for single-center vs two-center
        """
        # Filter to heavy unstable with known half-life
        mask = (
            (df_scored['A'] >= A_range[0]) &
            (df_scored['A'] <= A_range[1]) &
            (~df_scored['is_stable']) &
            (df_scored['half_life_s'].notna()) &
            (~np.isinf(df_scored['half_life_s']))
        )

        df_test = df_scored[mask].copy()

        if len(df_test) < 10:
            print(f"Warning: Only {len(df_test)} samples in range {A_range}")
            return None

        # Single-center correlation
        eps_1c = df_test['epsilon_best'].values
        log_t = np.log10(df_test['half_life_s'].values)

        r_1c, p_1c = stats.spearmanr(eps_1c, log_t)

        # Two-center correlation
        eps_2c = df_test['epsilon_2c'].values
        r_2c, p_2c = stats.spearmanr(eps_2c, log_t)

        print("="*80)
        print(f"VALIDATION: Half-Life Correlation (A ∈ [{A_range[0]}, {A_range[1]}])")
        print("="*80)
        print(f"Sample size: {len(df_test)} unstable nuclides\n")
        print(f"Single-center: r = {r_1c:+.4f}, p = {p_1c:.2e}")
        print(f"Two-center:    r = {r_2c:+.4f}, p = {p_2c:.2e}")
        print()

        # Improvement
        improvement = r_2c - r_1c
        print(f"Improvement: Δr = {improvement:+.4f}")

        if r_2c > r_1c and p_2c < 0.001:
            print("✓ TWO-CENTER MODEL RECOVERS CORRELATION!")
        elif r_2c > r_1c:
            print("? Two-center improves but not significantly")
        else:
            print("✗ Two-center does NOT improve correlation")

        print("="*80)

        return {
            'n_samples': len(df_test),
            'single_center': {'r': r_1c, 'p': p_1c},
            'two_center': {'r': r_2c, 'p': p_2c},
            'improvement': improvement,
            'A_range': A_range
        }

    def plot_comparison(self, df_scored, outpath='two_center_comparison.png'):
        """
        Plot single-center vs two-center epsilon comparison.

        Parameters
        ----------
        df_scored : pd.DataFrame
            Scored nuclides
        outpath : str
            Output file path
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Filter to heavy nuclides
        df_heavy = df_scored[(df_scored['A'] > 161) &
                              (df_scored['epsilon_2c'].notna())]

        # Panel 1: Epsilon comparison
        ax1.scatter(df_heavy['epsilon_best'], df_heavy['epsilon_2c'],
                    alpha=0.3, s=10, color='blue')
        ax1.plot([0, 0.5], [0, 0.5], 'r--', label='1:1 line')
        ax1.set_xlabel('Single-center ε', fontsize=11)
        ax1.set_ylabel('Two-center ε', fontsize=11)
        ax1.set_title('Epsilon Comparison (A > 161)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Beta distribution
        ax2.hist(df_heavy['beta_est'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Deformation β', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Deformation Distribution', fontsize=12, fontweight='bold')
        ax2.axvline(x=0.25, color='red', linestyle='--', label='Typical β ≈ 0.25')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: Half-life correlation (single-center)
        df_unstable = df_heavy[~df_heavy['is_stable'] &
                                df_heavy['half_life_s'].notna() &
                                ~np.isinf(df_heavy['half_life_s'])]

        if len(df_unstable) > 10:
            r_1c, p_1c = stats.spearmanr(df_unstable['epsilon_best'],
                                          np.log10(df_unstable['half_life_s']))
            ax3.scatter(df_unstable['epsilon_best'],
                        np.log10(df_unstable['half_life_s']),
                        alpha=0.3, s=10, color='blue')
            ax3.set_xlabel('Single-center ε', fontsize=11)
            ax3.set_ylabel('log₁₀(Half-life [s])', fontsize=11)
            ax3.set_title(f'Single-Center\nr = {r_1c:+.3f}, p = {p_1c:.2e}',
                          fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Panel 4: Half-life correlation (two-center)
            r_2c, p_2c = stats.spearmanr(df_unstable['epsilon_2c'],
                                          np.log10(df_unstable['half_life_s']))
            ax4.scatter(df_unstable['epsilon_2c'],
                        np.log10(df_unstable['half_life_s']),
                        alpha=0.3, s=10, color='red')
            ax4.set_xlabel('Two-center ε', fontsize=11)
            ax4.set_ylabel('log₁₀(Half-life [s])', fontsize=11)
            ax4.set_title(f'Two-Center\nr = {r_2c:+.3f}, p = {p_2c:.2e}',
                          fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        print(f"✓ Saved comparison plot to {outpath}")
        plt.close()


# ============================================================================
# Standalone Functions
# ============================================================================

def load_single_center_params(param_file):
    """Load single-center parameters from JSON"""
    with open(param_file, 'r') as f:
        params = json.load(f)
    return params


def test_two_center_model(nubase_parquet, params_json):
    """
    Test two-center model on heavy nuclides.

    Parameters
    ----------
    nubase_parquet : str
        Path to scored nuclides parquet file
    params_json : str
        Path to single-center parameters JSON

    Returns
    -------
    results : dict
        Validation results
    """
    # Load data
    df = pd.read_parquet(nubase_parquet)
    params = load_single_center_params(params_json)

    # Initialize model
    model = TwoCenterHarmonicModel(params)

    # Score heavy nuclides
    print("Scoring heavy nuclides with two-center model...")
    df_scored = model.score_dataframe_two_center(df, A_threshold=161)

    # Validate
    results = model.validate_halflife_correlation(df_scored, A_range=(161, 250))

    # Plot
    model.plot_comparison(df_scored, outpath='reports/tacoma_narrows/two_center_comparison.png')

    return results, df_scored


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Two-Center Harmonic Model')
    parser.add_argument('--scores', required=True, help='Scored nuclides parquet')
    parser.add_argument('--params', required=True, help='Single-center parameters JSON')
    parser.add_argument('--out', default='reports/tacoma_narrows/', help='Output directory')

    args = parser.parse_args()

    # Run test
    results, df_scored = test_two_center_model(args.scores, args.params)

    # Save results
    import os
    os.makedirs(args.out, exist_ok=True)

    output_file = os.path.join(args.out, 'two_center_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Save scored dataframe
    output_parquet = os.path.join(args.out, 'nuclides_two_center_scored.parquet')
    df_scored.to_parquet(output_parquet)
    print(f"✓ Scored nuclides saved to {output_parquet}")

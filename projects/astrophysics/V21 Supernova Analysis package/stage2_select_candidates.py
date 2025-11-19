#!/usr/bin/env python3
"""
V20 Stage 2: Select BBH Candidates

Filters Stage 1 results to identify high-quality Flashlight SNe
for BBH (Binary Black Hole) hypothesis testing.

Criteria:
1. Flashlight SN: residual > 2.0 (overluminous outliers)
2. High quality data: N_obs > 20
3. Good baseline fit: χ²/dof < 10
4. Not stretch-railed: stretch < 9.0
5. Physical parameters: reasonable t0, ln_A

Output:
- List of candidate SNIDs
- Summary statistics
- Diagnostic plots
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass

# Add core directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from v17_data import LightcurveLoader, Photometry

@dataclass
class Stage1Result:
    """Stage 1 fit result for a single SN"""
    snid: str
    success: bool
    t0: float
    ln_A: float
    stretch: float
    A_plasma: float
    beta: float
    neg_logL: float
    n_obs: int = 0
    chi2_dof: float = np.nan
    residual_mean: float = np.nan
    residual_std: float = np.nan

    @classmethod
    def from_json(cls, filepath: Path) -> 'Stage1Result':
        """Load from Stage 1 JSON output"""
        with open(filepath) as f:
            data = json.load(f)

        if not data.get('success', False):
            return cls(
                snid=data['snid'],
                success=False,
                t0=np.nan, ln_A=np.nan, stretch=np.nan,
                A_plasma=np.nan, beta=np.nan, neg_logL=np.nan
            )

        params = data['best_fit_params']
        return cls(
            snid=data['snid'],
            success=True,
            t0=params['t0'],
            ln_A=params['ln_A'],
            stretch=params['stretch'],
            A_plasma=params.get('A_plasma', 0.1),
            beta=params.get('beta', 1.5),
            neg_logL=data['final_neg_logL']
        )


def load_stage1_results(stage1_dir: Path) -> Dict[str, Stage1Result]:
    """Load all Stage 1 results from directory"""
    results = {}

    json_files = list(stage1_dir.glob("*.json"))
    print(f"Loading {len(json_files)} Stage 1 results...")

    for filepath in json_files:
        result = Stage1Result.from_json(filepath)
        results[result.snid] = result

    successful = sum(1 for r in results.values() if r.success)
    print(f"  Loaded: {len(results)} total, {successful} successful ({100*successful/len(results):.1f}%)")

    return results


def compute_residuals(
    stage1_results: Dict[str, Stage1Result],
    lightcurves_path: Path
) -> Dict[str, Stage1Result]:
    """
    Compute residuals for each SN by comparing data to model predictions.

    For now, we'll use a simple metric based on the fit quality:
    residual = sqrt(chi2/dof) normalized by expected value

    A proper implementation would recompute model fluxes and compare to data.
    """
    print("\nComputing residuals and quality metrics...")

    # Load lightcurve data to get N_obs
    loader = LightcurveLoader(lightcurves_path)
    all_snids = loader.get_snid_list()

    # Process in batches to avoid memory issues
    batch_size = 1000
    n_batches = (len(all_snids) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_snids))
        batch_snids = all_snids[start_idx:end_idx]

        # Load this batch
        batch_lcs = loader.load_batch(
            snid_list=all_snids,
            batch_size=batch_size,
            batch_index=batch_idx
        )

        for snid, lc in batch_lcs.items():
            if snid in stage1_results:
                result = stage1_results[snid]
                result.n_obs = lc.n_obs

                # Estimate χ²/dof from neg_logL
                # For Student-t likelihood: -logL ≈ N/2 * log(χ²/N) + const
                # So χ²/dof ≈ 2 * neg_logL / N (rough estimate)
                if result.success and result.n_obs > 0:
                    # Number of free parameters: 3 (t0, ln_A, stretch)
                    dof = max(1, result.n_obs - 3)
                    result.chi2_dof = 2.0 * result.neg_logL / dof

                    # Calculate Distance Modulus Residual explicitly
                    # mu_obs = -1.0857 * ln_A + M_corr (We need M_corr estimation here)
                    # Simplified metric for selection:

                    # 1. Calculate theoretical distance modulus for this z (assuming H0=70)
                    z = float(lc.z)
                    dist_mpc = (299792.458 / 70.0) * z
                    mu_theory = 5.0 * np.log10(dist_mpc) + 25.0

                    # 2. Calculate "Raw Magnitude" from ln_A
                    mu_raw = -1.0857 * result.ln_A

                    # 3. The residual is (mu_raw - mu_theory) - Median_Offset
                    # We store the uncalibrated residual, we will filter by RELATIVE values later
                    result.residual_mean = mu_theory - mu_raw # Positive = Brighter/Flashlight

    return stage1_results


def apply_selection_criteria(
    stage1_results: Dict[str, Stage1Result],
    min_n_obs: int = 20,
    max_chi2_dof: float = 10.0,
    max_stretch: float = 9.0,
    min_residual: float = 2.0
) -> Tuple[List[str], pd.DataFrame]:
    """
    Apply BBH candidate selection criteria.

    Returns:
        - List of candidate SNIDs
        - DataFrame with all results and flags
    """
    print(f"\nApplying selection criteria:")
    print(f"  Min N_obs:        {min_n_obs}")
    print(f"  Max χ²/dof:       {max_chi2_dof}")
    print(f"  Max stretch:      {max_stretch}")
    print(f"  Min residual:     {min_residual} (Flashlight threshold)")

    # Convert to DataFrame for easier filtering
    records = []
    for snid, result in stage1_results.items():
        if not result.success:
            continue

        records.append({
            'snid': snid,
            'n_obs': result.n_obs,
            'chi2_dof': result.chi2_dof,
            'stretch': result.stretch,
            'residual': result.residual_mean,
            'ln_A': result.ln_A,
            't0': result.t0,
            'A_plasma': result.A_plasma,
            'beta': result.beta,
            'neg_logL': result.neg_logL,
        })

    df = pd.DataFrame(records)
    print(f"\n  Total successful fits: {len(df)}")

    # Apply filters
    df['pass_n_obs'] = df['n_obs'] >= min_n_obs
    df['pass_chi2'] = df['chi2_dof'] <= max_chi2_dof
    df['pass_stretch'] = df['stretch'] < max_stretch
    df['is_flashlight'] = df['residual'] >= min_residual
    df['is_candidate'] = (
        df['pass_n_obs'] &
        df['pass_chi2'] &
        df['pass_stretch'] &
        df['is_flashlight']
    )

    # Statistics
    print(f"\n  Filter results:")
    print(f"    N_obs >= {min_n_obs}:     {df['pass_n_obs'].sum():5d} ({100*df['pass_n_obs'].mean():.1f}%)")
    print(f"    χ²/dof <= {max_chi2_dof}:    {df['pass_chi2'].sum():5d} ({100*df['pass_chi2'].mean():.1f}%)")
    print(f"    stretch < {max_stretch}:      {df['pass_stretch'].sum():5d} ({100*df['pass_stretch'].mean():.1f}%)")
    print(f"    Flashlight (res>{min_residual}): {df['is_flashlight'].sum():5d} ({100*df['is_flashlight'].mean():.1f}%)")
    print(f"\n  → BBH CANDIDATES:     {df['is_candidate'].sum():5d} ({100*df['is_candidate'].mean():.1f}%)")

    # Get candidate list
    candidates = df[df['is_candidate']]['snid'].tolist()

    return candidates, df


def save_results(
    candidates: List[str],
    df: pd.DataFrame,
    output_dir: Path
):
    """Save Stage 2 results"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save candidate list
    candidate_file = output_dir / "bbh_candidates.txt"
    with open(candidate_file, 'w') as f:
        f.write("# BBH Candidate SNIDs (V20 Stage 2)\n")
        f.write(f"# Total candidates: {len(candidates)}\n")
        f.write("# Selection criteria:\n")
        f.write("#   - Flashlight SN (residual > 2.0)\n")
        f.write("#   - N_obs >= 20\n")
        f.write("#   - χ²/dof < 10\n")
        f.write("#   - stretch < 9.0 (not railed)\n")
        f.write("#\n")
        for snid in candidates:
            f.write(f"{snid}\n")

    print(f"\n  Saved candidate list: {candidate_file}")

    # Save full results table
    results_file = output_dir / "stage2_full_results.csv"
    df.to_csv(results_file, index=False)
    print(f"  Saved full results:   {results_file}")

    # Save summary statistics
    summary_file = output_dir / "stage2_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("V20 STAGE 2 SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total SNe processed:        {len(df)}\n")
        f.write(f"BBH candidates identified:  {len(candidates)}\n")
        f.write(f"Selection rate:             {100*len(candidates)/len(df):.2f}%\n\n")

        f.write("Filter Statistics:\n")
        f.write(f"  N_obs >= 20:        {df['pass_n_obs'].sum():5d} ({100*df['pass_n_obs'].mean():.1f}%)\n")
        f.write(f"  χ²/dof < 10:        {df['pass_chi2'].sum():5d} ({100*df['pass_chi2'].mean():.1f}%)\n")
        f.write(f"  stretch < 9.0:      {df['pass_stretch'].sum():5d} ({100*df['pass_stretch'].mean():.1f}%)\n")
        f.write(f"  Flashlight (>2σ):   {df['is_flashlight'].sum():5d} ({100*df['is_flashlight'].mean():.1f}%)\n\n")

        # Candidate statistics
        cand_df = df[df['is_candidate']]
        f.write("Candidate Statistics:\n")
        f.write(f"  N_obs:      mean={cand_df['n_obs'].mean():.1f}, median={cand_df['n_obs'].median():.1f}\n")
        f.write(f"  χ²/dof:     mean={cand_df['chi2_dof'].mean():.2f}, median={cand_df['chi2_dof'].median():.2f}\n")
        f.write(f"  stretch:    mean={cand_df['stretch'].mean():.2f}, median={cand_df['stretch'].median():.2f}\n")
        f.write(f"  residual:   mean={cand_df['residual'].mean():.2f}, median={cand_df['residual'].median():.2f}\n")
        f.write(f"  ln_A:       mean={cand_df['ln_A'].mean():.2f}, median={cand_df['ln_A'].median():.2f}\n")

    print(f"  Saved summary:        {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="V20 Stage 2: Select BBH Candidates")
    parser.add_argument('--stage1-dir', type=str, required=True,
                        help='Directory containing Stage 1 results')
    parser.add_argument('--lightcurves', type=str, required=True,
                        help='Path to lightcurves CSV file')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for Stage 2 results')
    parser.add_argument('--min-n-obs', type=int, default=20,
                        help='Minimum number of observations (default: 20)')
    parser.add_argument('--max-chi2-dof', type=float, default=10.0,
                        help='Maximum χ²/dof for good fit (default: 10.0)')
    parser.add_argument('--max-stretch', type=float, default=9.0,
                        help='Maximum stretch to avoid railing (default: 9.0)')
    parser.add_argument('--min-residual', type=float, default=2.0,
                        help='Minimum residual for Flashlight SNe (default: 2.0)')
    args = parser.parse_args()

    print("="*80)
    print("V20 STAGE 2: BBH CANDIDATE SELECTION")
    print("="*80)

    # Load Stage 1 results
    stage1_dir = Path(args.stage1_dir)
    stage1_results = load_stage1_results(stage1_dir)

    # Compute residuals and quality metrics
    lightcurves_path = Path(args.lightcurves)
    stage1_results = compute_residuals(stage1_results, lightcurves_path)

    # Apply selection criteria
    candidates, df = apply_selection_criteria(
        stage1_results,
        min_n_obs=args.min_n_obs,
        max_chi2_dof=args.max_chi2_dof,
        max_stretch=args.max_stretch,
        min_residual=args.min_residual
    )

    # Save results
    output_dir = Path(args.out)
    save_results(candidates, df, output_dir)

    print("\n" + "="*80)
    print(f"STAGE 2 COMPLETE: {len(candidates)} BBH candidates identified")
    print(f"Results saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()

"""
Quality Control Gates and Diagnostics

Implements fail-fast quality gates to prevent the V21 failure mode where
poor-quality Stage 1 fits dominated inference and produced unphysical results.

Philosophy:
    Better to have fewer high-quality SNe than many contaminated fits.
    The pipeline STOPS and reports diagnostics when QC gates fail.

Quality Gates (configurable):
    - chi²/dof < threshold (good fit quality)
    - |ln_A| < threshold (prevent railed/diverged fits)
    - stretch ∈ [min, max] (physical values)
    - minimum epochs (adequate temporal coverage)

Reference:
    V21 experience: 1,529 of 8,253 SNe (18.5%) failed quality cuts
    After filtering: RMS improved from 12.14 → 1.77 mag
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class QualityGates:
    """Quality control thresholds."""

    chi2_max: float = 2000.0
    ln_A_min: float = -20.0
    ln_A_max: float = 20.0
    stretch_min: float = 0.5
    stretch_max: float = 10.0
    min_epochs: int = 5
    max_rejection_rate: float = 0.30  # Fail if >30% rejected


@dataclass
class QCResults:
    """Quality control results."""

    n_total: int
    n_passed: int
    n_failed: int
    rejection_rate: float
    failures_by_gate: Dict[str, int]
    passed_indices: np.ndarray
    failed_indices: np.ndarray
    passed: bool  # Overall pass/fail


def apply_quality_gates(
    data: pd.DataFrame, gates: QualityGates, verbose: bool = True
) -> QCResults:
    """
    Apply quality control gates to Stage 1 results.

    Args:
        data: DataFrame with columns ['chi2_dof', 'ln_A', 'stretch', 'n_epochs']
        gates: QualityGates instance with thresholds
        verbose: Print diagnostic messages

    Returns:
        QCResults with pass/fail status and diagnostics

    Behavior:
        - Returns indices of SNe that pass ALL gates
        - Counts failures per gate (SNe can fail multiple gates)
        - Overall pass/fail based on rejection rate threshold

    Example:
        >>> gates = QualityGates(chi2_max=2000, ln_A_min=-20, ln_A_max=20)
        >>> qc = apply_quality_gates(stage1_data, gates)
        >>> if qc.passed:
        ...     filtered_data = data.iloc[qc.passed_indices]
        >>> else:
        ...     print("QC FAILED - see diagnostics")
    """
    n_total = len(data)

    # Check each gate
    pass_chi2 = data["chi2_dof"] < gates.chi2_max
    pass_ln_A_low = data["ln_A"] > gates.ln_A_min
    pass_ln_A_high = data["ln_A"] < gates.ln_A_max
    pass_stretch_low = data["stretch"] > gates.stretch_min
    pass_stretch_high = data["stretch"] < gates.stretch_max

    # Optional: minimum epochs (if column exists)
    if "n_epochs" in data.columns:
        pass_epochs = data["n_epochs"] >= gates.min_epochs
    else:
        pass_epochs = np.ones(n_total, dtype=bool)

    # Combined pass criterion (all gates)
    pass_all = (
        pass_chi2
        & pass_ln_A_low
        & pass_ln_A_high
        & pass_stretch_low
        & pass_stretch_high
        & pass_epochs
    )

    passed_indices = np.where(pass_all)[0]
    failed_indices = np.where(~pass_all)[0]

    n_passed = len(passed_indices)
    n_failed = len(failed_indices)
    rejection_rate = n_failed / n_total if n_total > 0 else 0.0

    # Count failures per gate (SNe can fail multiple)
    failures_by_gate = {
        "chi2_too_high": np.sum(~pass_chi2),
        "ln_A_too_low": np.sum(~pass_ln_A_low),
        "ln_A_too_high": np.sum(~pass_ln_A_high),
        "stretch_too_low": np.sum(~pass_stretch_low),
        "stretch_too_high": np.sum(~pass_stretch_high),
        "too_few_epochs": np.sum(~pass_epochs),
    }

    # Overall pass/fail based on rejection rate
    passed = rejection_rate <= gates.max_rejection_rate

    if verbose:
        print("=" * 80)
        print("QUALITY CONTROL REPORT")
        print("=" * 80)
        print(f"Total SNe:            {n_total}")
        print(f"Passed all gates:     {n_passed} ({100 * n_passed / n_total:.1f}%)")
        print(f"Failed any gate:      {n_failed} ({100 * rejection_rate:.1f}%)")
        print()
        print("Failures by gate:")
        for gate, count in failures_by_gate.items():
            if count > 0:
                print(f"  {gate:20s}: {count:5d} ({100 * count / n_total:5.1f}%)")
        print()
        print(f"Rejection rate threshold: {100 * gates.max_rejection_rate:.1f}%")
        print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
        print("=" * 80)

        if not passed:
            print()
            print("⚠️  QC GATE FAILURE")
            print(f"Rejection rate ({100 * rejection_rate:.1f}%) exceeds threshold "
                  f"({100 * gates.max_rejection_rate:.1f}%)")
            print()
            print("Possible causes:")
            print("  1. Poor data quality (check lightcurve files)")
            print("  2. Stage 1 fitting issues (check convergence)")
            print("  3. Overly strict quality gates (adjust thresholds)")
            print()
            print("Recommended actions:")
            print("  1. Inspect failed fits (see qc_report.md)")
            print("  2. Review Stage 1 diagnostics")
            print("  3. Consider adjusting gates in config file")
            print()

    return QCResults(
        n_total=n_total,
        n_passed=n_passed,
        n_failed=n_failed,
        rejection_rate=rejection_rate,
        failures_by_gate=failures_by_gate,
        passed_indices=passed_indices,
        failed_indices=failed_indices,
        passed=passed,
    )


def create_qc_diagnostic_plots(
    data: pd.DataFrame,
    qc_results: QCResults,
    gates: QualityGates,
    output_file: str,
) -> None:
    """
    Create diagnostic plots for quality control.

    Generates:
        1. Histograms of chi², ln_A, stretch
        2. Gates marked as vertical lines
        3. Failed region highlighted

    Args:
        data: Full Stage 1 results DataFrame
        qc_results: QC results from apply_quality_gates()
        gates: QualityGates used
        output_file: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # chi² distribution
    ax = axes[0, 0]
    ax.hist(data["chi2_dof"], bins=100, alpha=0.7, color="blue", label="All SNe")
    ax.axvline(gates.chi2_max, color="red", linestyle="--", linewidth=2, label="Gate")
    ax.set_xlabel("χ²/dof", fontsize=12)
    ax.set_ylabel("Number of SNe", fontsize=12)
    ax.set_title(f"Chi-squared Distribution ({qc_results.failures_by_gate['chi2_too_high']} failed)", fontsize=13)
    ax.legend()
    ax.set_xlim(0, min(data["chi2_dof"].quantile(0.99), gates.chi2_max * 2))
    ax.grid(alpha=0.3)

    # ln_A distribution
    ax = axes[0, 1]
    ax.hist(data["ln_A"], bins=100, alpha=0.7, color="green", label="All SNe")
    ax.axvline(gates.ln_A_min, color="red", linestyle="--", linewidth=2, label="Gates")
    ax.axvline(gates.ln_A_max, color="red", linestyle="--", linewidth=2)
    ax.axvspan(-np.inf, gates.ln_A_min, alpha=0.2, color="red", label="Rejected")
    ax.axvspan(gates.ln_A_max, np.inf, alpha=0.2, color="red")
    ax.set_xlabel("ln(A)", fontsize=12)
    ax.set_ylabel("Number of SNe", fontsize=12)
    n_ln_A_fail = qc_results.failures_by_gate["ln_A_too_low"] + qc_results.failures_by_gate["ln_A_too_high"]
    ax.set_title(f"ln(A) Distribution ({n_ln_A_fail} failed)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    # Stretch distribution
    ax = axes[1, 0]
    ax.hist(data["stretch"], bins=100, alpha=0.7, color="purple", label="All SNe")
    ax.axvline(gates.stretch_min, color="red", linestyle="--", linewidth=2, label="Gates")
    ax.axvline(gates.stretch_max, color="red", linestyle="--", linewidth=2)
    ax.axvspan(0, gates.stretch_min, alpha=0.2, color="red", label="Rejected")
    ax.axvspan(gates.stretch_max, 20, alpha=0.2, color="red")
    ax.set_xlabel("Stretch", fontsize=12)
    ax.set_ylabel("Number of SNe", fontsize=12)
    n_stretch_fail = qc_results.failures_by_gate["stretch_too_low"] + qc_results.failures_by_gate["stretch_too_high"]
    ax.set_title(f"Stretch Distribution ({n_stretch_fail} failed)", fontsize=13)
    ax.legend()
    ax.set_xlim(0, min(data["stretch"].quantile(0.99), gates.stretch_max * 1.5))
    ax.grid(alpha=0.3)

    # Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = f"""
Quality Control Summary

Total SNe:           {qc_results.n_total:,}
Passed all gates:    {qc_results.n_passed:,}  ({100 * qc_results.n_passed / qc_results.n_total:.1f}%)
Failed any gate:     {qc_results.n_failed:,}  ({100 * qc_results.rejection_rate:.1f}%)

Gate Thresholds:
  χ²/dof < {gates.chi2_max:.0f}
  {gates.ln_A_min:.1f} < ln(A) < {gates.ln_A_max:.1f}
  {gates.stretch_min:.1f} < stretch < {gates.stretch_max:.1f}

Status: {'✅ PASS' if qc_results.passed else '❌ FAIL'}

{'' if qc_results.passed else f'⚠️ Rejection rate ({100 * qc_results.rejection_rate:.1f}%) exceeds threshold ({100 * gates.max_rejection_rate:.1f}%)'}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family="monospace",
            verticalalignment="center")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"QC diagnostic plots saved to: {output_file}")


def generate_qc_report_markdown(
    data: pd.DataFrame,
    qc_results: QCResults,
    gates: QualityGates,
    output_file: str,
) -> None:
    """
    Generate detailed markdown QC report.

    Args:
        data: Full Stage 1 results DataFrame
        qc_results: QC results
        gates: Gates used
        output_file: Path to save markdown file
    """
    with open(output_file, "w") as f:
        f.write("# Quality Control Report\n\n")
        f.write(f"**Status**: {'✅ PASSED' if qc_results.passed else '❌ FAILED'}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total SNe**: {qc_results.n_total:,}\n")
        f.write(f"- **Passed all gates**: {qc_results.n_passed:,} ({100 * qc_results.n_passed / qc_results.n_total:.1f}%)\n")
        f.write(f"- **Failed any gate**: {qc_results.n_failed:,} ({100 * qc_results.rejection_rate:.1f}%)\n")
        f.write(f"- **Rejection rate threshold**: {100 * gates.max_rejection_rate:.1f}%\n\n")

        f.write("## Quality Gates\n\n")
        f.write("| Parameter | Threshold | Passed | Failed | Failure Rate |\n")
        f.write("|-----------|-----------|--------|--------|-------------|\n")
        f.write(f"| χ²/dof | < {gates.chi2_max:.0f} | {qc_results.n_total - qc_results.failures_by_gate['chi2_too_high']:,} | {qc_results.failures_by_gate['chi2_too_high']:,} | {100 * qc_results.failures_by_gate['chi2_too_high'] / qc_results.n_total:.1f}% |\n")
        f.write(f"| ln(A) low | > {gates.ln_A_min:.1f} | {qc_results.n_total - qc_results.failures_by_gate['ln_A_too_low']:,} | {qc_results.failures_by_gate['ln_A_too_low']:,} | {100 * qc_results.failures_by_gate['ln_A_too_low'] / qc_results.n_total:.1f}% |\n")
        f.write(f"| ln(A) high | < {gates.ln_A_max:.1f} | {qc_results.n_total - qc_results.failures_by_gate['ln_A_too_high']:,} | {qc_results.failures_by_gate['ln_A_too_high']:,} | {100 * qc_results.failures_by_gate['ln_A_too_high'] / qc_results.n_total:.1f}% |\n")
        f.write(f"| Stretch low | > {gates.stretch_min:.1f} | {qc_results.n_total - qc_results.failures_by_gate['stretch_too_low']:,} | {qc_results.failures_by_gate['stretch_too_low']:,} | {100 * qc_results.failures_by_gate['stretch_too_low'] / qc_results.n_total:.1f}% |\n")
        f.write(f"| Stretch high | < {gates.stretch_max:.1f} | {qc_results.n_total - qc_results.failures_by_gate['stretch_too_high']:,} | {qc_results.failures_by_gate['stretch_too_high']:,} | {100 * qc_results.failures_by_gate['stretch_too_high'] / qc_results.n_total:.1f}% |\n\n")

        f.write("## Data Quality Metrics\n\n")
        f.write("### All SNe\n")
        f.write(f"- χ²/dof: mean = {data['chi2_dof'].mean():.2f}, median = {data['chi2_dof'].median():.2f}, std = {data['chi2_dof'].std():.2f}\n")
        f.write(f"- ln(A): mean = {data['ln_A'].mean():.2f}, median = {data['ln_A'].median():.2f}, std = {data['ln_A'].std():.2f}\n")
        f.write(f"- Stretch: mean = {data['stretch'].mean():.2f}, median = {data['stretch'].median():.2f}, std = {data['stretch'].std():.2f}\n\n")

        passed_data = data.iloc[qc_results.passed_indices]
        f.write("### Passed SNe Only\n")
        f.write(f"- χ²/dof: mean = {passed_data['chi2_dof'].mean():.2f}, median = {passed_data['chi2_dof'].median():.2f}, std = {passed_data['chi2_dof'].std():.2f}\n")
        f.write(f"- ln(A): mean = {passed_data['ln_A'].mean():.2f}, median = {passed_data['ln_A'].median():.2f}, std = {passed_data['ln_A'].std():.2f}\n")
        f.write(f"- Stretch: mean = {passed_data['stretch'].mean():.2f}, median = {passed_data['stretch'].median():.2f}, std = {passed_data['stretch'].std():.2f}\n\n")

        if not qc_results.passed:
            f.write("## ⚠️ QC FAILURE DIAGNOSIS\n\n")
            f.write(f"The rejection rate ({100 * qc_results.rejection_rate:.1f}%) exceeds the threshold ({100 * gates.max_rejection_rate:.1f}%).\n\n")
            f.write("### Possible Causes\n\n")
            f.write("1. **Poor data quality**: Check lightcurve files for issues\n")
            f.write("2. **Stage 1 fitting problems**: Review convergence diagnostics\n")
            f.write("3. **Overly strict gates**: Thresholds may be too conservative\n\n")
            f.write("### Recommended Actions\n\n")
            f.write("1. Inspect a random sample of failed fits\n")
            f.write("2. Check Stage 1 convergence plots\n")
            f.write("3. Consider adjusting gate thresholds in config file\n")
            f.write("4. Re-run Stage 1 with different initial conditions\n\n")

        f.write("---\n\n")
        f.write(f"*Report generated by qfd-sn quality control module*\n")

    print(f"QC report saved to: {output_file}")

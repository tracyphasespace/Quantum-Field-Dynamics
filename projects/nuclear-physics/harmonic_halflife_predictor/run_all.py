#!/usr/bin/env python3
"""
Harmonic Halflife Predictor - Complete Pipeline Runner

One-stop script to reproduce all results from the paper:
  1. Download AME2020 data (if needed)
  2. Run all 4 validation engines (A, B, C, D)
  3. Generate all figures
  4. Create summary reports

Usage:
    python run_all.py                    # Run everything
    python run_all.py --engines-only     # Skip data download and figures
    python run_all.py --figures-only     # Only regenerate figures
    python run_all.py --help             # Show all options

Output:
    results/    - CSV files and analysis results
    figures/    - Publication-quality plots
    logs/       - Execution logs
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Color output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_step(number, text):
    """Print step number and description."""
    print(f"{Colors.BOLD}{Colors.OKBLUE}[{number}] {text}{Colors.ENDC}")


def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def run_script(script_path, description, log_dir=None):
    """
    Run a Python script and capture output.

    Args:
        script_path: Path to the script
        description: Human-readable description
        log_dir: Optional directory to save logs

    Returns:
        True if successful, False otherwise
    """
    print(f"  Running: {script_path.name}")

    # Create log file if log directory provided
    log_file = None
    if log_dir:
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{script_path.stem}_{timestamp}.log"

    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)
            print(f"    Log: {log_file}")
        else:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True
            )

        print_success(description)
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        if log_file and log_file.exists():
            print(f"    Check log: {log_file}")
        return False


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Run complete Harmonic Halflife Predictor pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Full pipeline
  python run_all.py --engines-only     # Skip data download and figures
  python run_all.py --figures-only     # Only generate figures
  python run_all.py --skip-download    # Use existing data
        """
    )

    parser.add_argument('--engines-only', action='store_true',
                       help='Run only the 4 validation engines')
    parser.add_argument('--figures-only', action='store_true',
                       help='Generate only the figures')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download (use existing data)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output from each script')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / 'scripts'
    results_dir = base_dir / 'results'
    figures_dir = base_dir / 'figures'
    logs_dir = base_dir / 'logs'
    data_dir = base_dir / 'data'

    # Create directories
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Print banner
    print_header("Harmonic Halflife Predictor - Complete Pipeline")
    print("A unified geometric framework for exotic nuclear decay")
    print("Authors: Tracy McSheery")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Track success/failure
    steps_passed = 0
    steps_failed = 0

    # Step 1: Download data (unless skipped)
    if not args.figures_only and not args.skip_download:
        print_step(1, "Downloading AME2020 nuclear data")

        data_file = data_dir / 'ame2020_system_energies.csv'
        if data_file.exists():
            print(f"  Data already exists: {data_file}")
            print_success("AME2020 data available")
            steps_passed += 1
        else:
            if run_script(scripts_dir / 'download_ame2020.py',
                         'AME2020 data downloaded', logs_dir):
                steps_passed += 1
            else:
                steps_failed += 1
                print_error("Cannot proceed without data. Exiting.")
                return 1

    # Step 2: Run validation engines (unless figures-only)
    if not args.figures_only:
        print_step(2, "Running 4 Validation Engines")

        engines = [
            ('cluster_decay_scanner.py', 'Engine C: Cluster Decay (Pythagorean)'),
            ('neutron_drip_scanner.py', 'Engine A: Neutron Drip Line'),
            ('fission_neck_scan.py', 'Engine B: Fission Asymmetry'),
            ('validate_proton_engine.py', 'Engine D: Proton Drip Line'),
        ]

        for script_name, description in engines:
            script_path = scripts_dir / script_name
            if script_path.exists():
                if run_script(script_path, description, logs_dir):
                    steps_passed += 1
                else:
                    steps_failed += 1
            else:
                print(f"  Script not found: {script_name}")
                steps_failed += 1

    # Step 3: Generate figures (unless engines-only)
    if not args.engines_only:
        print_step(3, "Generating Publication Figures")

        figure_scripts = [
            ('generate_yrast_plots.py', 'Yrast spectroscopy plots'),
            ('plot_n_conservation.py', 'N-conservation plots'),
            # Add more figure generation scripts as needed
        ]

        for script_name, description in figure_scripts:
            script_path = scripts_dir / script_name
            if script_path.exists():
                if run_script(script_path, description, logs_dir):
                    steps_passed += 1
                else:
                    steps_failed += 1
            else:
                print(f"  Script not found: {script_name} (skipping)")

    # Final summary
    print_header("Pipeline Execution Summary")

    total_steps = steps_passed + steps_failed
    success_rate = (steps_passed / total_steps * 100) if total_steps > 0 else 0

    print(f"Total steps: {total_steps}")
    print(f"{Colors.OKGREEN}Passed: {steps_passed}{Colors.ENDC}")
    if steps_failed > 0:
        print(f"{Colors.FAIL}Failed: {steps_failed}{Colors.ENDC}")
    print(f"Success rate: {success_rate:.1f}%")

    print(f"\nOutput directories:")
    print(f"  Results: {results_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Logs: {logs_dir}")

    if steps_failed == 0:
        print_success("All steps completed successfully!")
        print("\nNext steps:")
        print("  1. Check results/ for CSV files and analysis")
        print("  2. Check figures/ for publication-quality plots")
        print("  3. Read QUICK_START.txt for interpretation guide")
        return 0
    else:
        print_error(f"{steps_failed} step(s) failed. Check logs for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

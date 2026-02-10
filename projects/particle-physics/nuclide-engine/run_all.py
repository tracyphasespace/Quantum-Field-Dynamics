#!/usr/bin/env python3
"""
run_all.py — Orchestrator for QFD Nuclide Engine analysis suite.

Runs the core engine and test scripts in dependency order.
All stdout is captured to results/.

Usage: python run_all.py
"""

import os
import sys
import time
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(RESULTS_DIR, 'run_all.log')


def run_script(name, log_file, timeout=300):
    """Run a script and return (success, elapsed_seconds)."""
    path = os.path.join(SCRIPTS_DIR, name)
    print(f"  Running {name}...", end='', flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SCRIPTS_DIR,
        )
        elapsed = time.time() - t0
        log_file.write(f"\n{'='*72}\n{name} (exit={result.returncode}, {elapsed:.1f}s)\n{'='*72}\n")
        log_file.write(result.stdout)
        if result.stderr:
            log_file.write(f"\n--- stderr ---\n{result.stderr}")

        # Save individual output
        out_name = name.replace('.py', '_output.txt')
        with open(os.path.join(RESULTS_DIR, out_name), 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n--- stderr ---\n{result.stderr}")

        if result.returncode == 0:
            print(f" OK ({elapsed:.1f}s)")
            return True, elapsed
        else:
            print(f" FAILED (exit={result.returncode}, {elapsed:.1f}s)")
            err_lines = result.stderr.strip().split('\n')
            for line in err_lines[-5:]:
                print(f"    {line}")
            return False, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f" TIMEOUT ({elapsed:.1f}s)")
        log_file.write(f"\n{'='*72}\n{name} TIMEOUT after {elapsed:.1f}s\n{'='*72}\n")
        return False, elapsed


def main():
    print("=" * 60)
    print("QFD Nuclide Engine — Full Reproduction Run")
    print("=" * 60)
    print(f"\nScripts: {SCRIPTS_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Log:     {LOG_PATH}\n")

    t_start = time.time()
    passed = 0
    failed = 0

    with open(LOG_PATH, 'w') as log:
        log.write("QFD Nuclide Engine — run_all.py log\n")
        log.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Stage 1: Core engine (must run first to verify imports work)
        print("Stage 1: Core engine")
        ok, _ = run_script('model_nuclide_topology.py', log, timeout=300)
        if ok:
            passed += 1
        else:
            failed += 1

        # Stage 2: Channel analysis (imports from model_nuclide_topology)
        print("\nStage 2: Channel analysis")
        ok, _ = run_script('channel_analysis.py', log, timeout=300)
        if ok:
            passed += 1
        else:
            failed += 1

        # Stage 3: Isomer clock (imports from model_nuclide_topology)
        print("\nStage 3: Isomer clock analysis")
        ok, _ = run_script('isomer_clock_analysis.py', log, timeout=300)
        if ok:
            passed += 1
        else:
            failed += 1

        # Stage 4: Independent test scripts
        print("\nStage 4: Test scripts")
        tests = [
            'test_177_algebraic.py',
            'test_density_shells.py',
            'test_diameter_ceiling.py',
            'test_lyapunov_clock.py',
            'test_overflow_unified.py',
            'test_resonance_spacing.py',
            'test_tennis_racket.py',
        ]
        for script in tests:
            ok, _ = run_script(script, log, timeout=120)
            if ok:
                passed += 1
            else:
                failed += 1

        # Stage 5: Heatmap (optional — requires matplotlib)
        print("\nStage 5: Visualization (optional)")
        try:
            import matplotlib
            ok, _ = run_script('qfd_6d_heatmap.py', log, timeout=120)
            if ok:
                passed += 1
            else:
                failed += 1
        except ImportError:
            print("  Skipping qfd_6d_heatmap.py (matplotlib not installed)")
            log.write("\nqfd_6d_heatmap.py SKIPPED (no matplotlib)\n")

        log.write(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    t_total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Total time: {t_total:.1f}s")
    print(f"Log saved: {LOG_PATH}")
    print(f"Individual outputs: {RESULTS_DIR}/")

    if failed == 0:
        print("\nAll scripts passed.")
    else:
        print(f"\n{failed} script(s) failed. Check the log for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

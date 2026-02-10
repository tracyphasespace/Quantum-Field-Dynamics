#!/usr/bin/env python3
"""
run_all.py — Master orchestrator for Three-Layer LaGrangian reviewer package.

Runs all analysis scripts in dependency order and verifies outputs.
Usage: python run_all.py
"""

import os
import sys
import time
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Environment for child processes
env = os.environ.copy()
env['TLG_DATA_DIR'] = DATA_DIR
env['TLG_RESULTS_DIR'] = RESULTS_DIR

LOG_PATH = os.path.join(RESULTS_DIR, 'run_all.log')


def run_script(name, log_file):
    """Run a script and return (success, elapsed_seconds)."""
    path = os.path.join(SCRIPTS_DIR, name)
    print(f"  Running {name}...", end='', flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, path],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.time() - t0
        log_file.write(f"\n{'='*72}\n{name} (exit={result.returncode}, {elapsed:.1f}s)\n{'='*72}\n")
        log_file.write(result.stdout)
        if result.stderr:
            log_file.write(f"\n--- stderr ---\n{result.stderr}")
        if result.returncode == 0:
            print(f" OK ({elapsed:.1f}s)")
            return True, elapsed
        else:
            print(f" FAILED (exit={result.returncode}, {elapsed:.1f}s)")
            # Print last few lines of stderr for diagnosis
            err_lines = result.stderr.strip().split('\n')
            for line in err_lines[-5:]:
                print(f"    {line}")
            return False, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f" TIMEOUT ({elapsed:.1f}s)")
        log_file.write(f"\n{'='*72}\n{name} TIMEOUT after {elapsed:.1f}s\n{'='*72}\n")
        return False, elapsed


def check_csv(name, min_rows=1):
    """Check that a CSV exists in RESULTS_DIR with at least min_rows rows."""
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return False, 0
    with open(path) as f:
        lines = sum(1 for _ in f) - 1  # subtract header
    return lines >= min_rows, lines


def main():
    print("=" * 60)
    print("Three-Layer LaGrangian — Full Reproduction Run")
    print("=" * 60)
    print(f"\nData:    {DATA_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Log:     {LOG_PATH}\n")

    t_start = time.time()
    all_ok = True

    with open(LOG_PATH, 'w') as log:
        log.write("Three-Layer LaGrangian — run_all.py log\n")
        log.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Stage 1: independent scripts (no inter-dependencies)
        print("Stage 1: Independent scripts")
        stage1 = [
            'model_comparison.py',
            'lagrangian_layer_a_orthogonal.py',
            'tracked_channel_fits.py',
            'zero_param_clock.py',
            'lagrangian_decomposition.py',
            'rate_competition.py',
        ]
        for script in stage1:
            ok, _ = run_script(script, log)
            if not ok:
                all_ok = False

        # Stage 2: depends on lagrangian_decomposition.csv
        print("\nStage 2: Depends on lagrangian_decomposition.csv")
        ok, _ = run_script('layer_c_investigation.py', log)
        if not ok:
            all_ok = False

        log.write(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    t_total = time.time() - t_start

    # Verify outputs
    print(f"\n{'='*60}")
    print("Output Verification")
    print(f"{'='*60}")

    expected = [
        ('model_comparison_results.csv', 10),
        ('lagrangian_decomposition.csv', 3000),
        ('tracked_channel_scores.csv', 10),
        ('rate_competition_results.csv', 3),
    ]

    for name, min_rows in expected:
        ok, rows = check_csv(name, min_rows)
        status = f"OK ({rows} rows)" if ok else "MISSING or too few rows"
        print(f"  {name:40s}  {status}")
        if not ok:
            all_ok = False

    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Log saved: {LOG_PATH}")

    if all_ok:
        print("\nAll scripts passed. Results are in results/.")
    else:
        print("\nSome scripts failed. Check the log for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

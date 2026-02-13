#!/usr/bin/env python3
"""
QFD Core Validation Runner
===========================

Runs the essential validation scripts sequentially and reports results.
Designed for CI and Docker environments.

Exit code: 0 if all critical tests pass, 1 otherwise.
"""

import sys
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Critical validations (must all pass)
CRITICAL = [
    ("Shared Constants", [sys.executable, "-m", "qfd.shared_constants"], PROJECT_ROOT),
    ("Golden Loop Error Propagation",
     [sys.executable, str(SCRIPT_DIR / "golden_loop_error_propagation.py")], PROJECT_ROOT),
    ("k_geom Derivation from Integrals",
     [sys.executable, str(SCRIPT_DIR / "derive_k_geom_from_integrals.py")], PROJECT_ROOT),
]

# Non-critical validations (informational, allowed to fail)
INFORMATIONAL = [
    ("Verify Golden Loop",
     [sys.executable, str(SCRIPT_DIR / "verify_golden_loop.py")], PROJECT_ROOT),
    ("Derive Beta from Alpha",
     [sys.executable, str(SCRIPT_DIR / "derive_beta_from_alpha.py")], PROJECT_ROOT),
    ("g-2 Corrected",
     [sys.executable, str(SCRIPT_DIR / "validate_g2_corrected.py")], PROJECT_ROOT),
    ("Lepton Stability",
     [sys.executable, str(SCRIPT_DIR / "lepton_stability.py")], PROJECT_ROOT),
    ("Tau Anomaly Test",
     [sys.executable, str(PROJECT_ROOT / "projects/particle-physics/lepton-isomer-ladder/tau_anomaly_test.py")],
     PROJECT_ROOT),
]


def run_validation(name, cmd, cwd):
    """Run a single validation and return (name, success, output)."""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(cwd)
        )
        success = result.returncode == 0
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if success:
            # Print last 5 lines of output as summary
            lines = output.strip().split('\n')
            for line in lines[-5:]:
                print(f"  {line}")
        else:
            print(f"  FAILED (exit code {result.returncode})")
            for line in output.strip().split('\n')[-10:]:
                print(f"  {line}")
        return success
    except FileNotFoundError:
        print(f"  SKIPPED (script not found)")
        return None  # Not a failure, just missing
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>120s)")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("  QFD CORE VALIDATION SUITE")
    print("=" * 60)

    critical_pass = 0
    critical_fail = 0
    info_pass = 0
    info_fail = 0
    info_skip = 0

    # Run critical tests
    print("\n--- CRITICAL TESTS (must pass) ---")
    for name, cmd, cwd in CRITICAL:
        result = run_validation(name, cmd, cwd)
        if result is True:
            critical_pass += 1
        elif result is False:
            critical_fail += 1

    # Run informational tests
    print("\n--- INFORMATIONAL TESTS (may fail) ---")
    for name, cmd, cwd in INFORMATIONAL:
        result = run_validation(name, cmd, cwd)
        if result is True:
            info_pass += 1
        elif result is False:
            info_fail += 1
        else:
            info_skip += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Critical: {critical_pass}/{critical_pass + critical_fail} passed")
    print(f"  Informational: {info_pass}/{info_pass + info_fail + info_skip} passed"
          f" ({info_skip} skipped)")

    if critical_fail > 0:
        print(f"\n  *** {critical_fail} CRITICAL TEST(S) FAILED ***")
        return 1
    else:
        print(f"\n  *** ALL CRITICAL TESTS PASSED ***")
        return 0


if __name__ == "__main__":
    sys.exit(main())

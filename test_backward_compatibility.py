#!/usr/bin/env python3
"""
Test backward compatibility between legacy and enhanced nuclear adapters.

Verifies that the enhanced version produces IDENTICAL results to the original.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Import both versions
sys.path.insert(0, str(Path(__file__).parent))

from qfd.adapters.nuclear.charge_prediction_legacy import predict_charge as predict_legacy
from qfd.adapters.nuclear.charge_prediction import predict_charge as predict_enhanced

def test_identical_results():
    """Test that both versions produce identical predictions."""

    print("=" * 70)
    print("Backward Compatibility Test: Legacy vs Enhanced")
    print("=" * 70)

    # Test data
    test_cases = [
        {
            "name": "Simple test",
            "df": pd.DataFrame({"A": [4, 12, 16, 56, 208]}),
            "params": {"c1": 1.0, "c2": 0.4}
        },
        {
            "name": "Phase 1 validated params",
            "df": pd.DataFrame({"A": [1, 2, 3, 4, 10, 20, 50, 100, 200]}),
            "params": {"c1": 0.496296, "c2": 0.323671}
        },
        {
            "name": "Nuclide-prediction fit",
            "df": pd.DataFrame({"A": [12, 56, 120, 208]}),
            "params": {"c1": 0.5292508558990585, "c2": 0.31674263258172686}
        },
        {
            "name": "Edge case (A=1)",
            "df": pd.DataFrame({"A": [1]}),
            "params": {"c1": 0.5, "c2": 0.3}
        },
        {
            "name": "Large A",
            "df": pd.DataFrame({"A": [238, 250, 300]}),
            "params": {"c1": 0.496, "c2": 0.324}
        }
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"  A: {test['df']['A'].values}")
        print(f"  Params: c1={test['params']['c1']:.6f}, c2={test['params']['c2']:.6f}")

        # Run both versions
        Q_legacy = predict_legacy(test['df'], test['params'])
        Q_enhanced = predict_enhanced(test['df'], test['params'], config={"validate_constraints": False})

        # Compare
        diff = np.abs(Q_legacy - Q_enhanced)
        max_diff = np.max(diff)
        rel_diff = np.max(diff / (np.abs(Q_legacy) + 1e-10))

        print(f"  Legacy:   {Q_legacy}")
        print(f"  Enhanced: {Q_enhanced}")
        print(f"  Max abs diff: {max_diff:.2e}")
        print(f"  Max rel diff: {rel_diff:.2e}")

        # Check if identical (within floating point precision)
        if max_diff < 1e-10:
            print("  ✓ PASS: Results identical")
        else:
            print(f"  ✗ FAIL: Results differ by {max_diff}")
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("✅ ALL TESTS PASSED: Enhanced version is 100% backward compatible")
        return True
    else:
        print("❌ SOME TESTS FAILED: Results differ between versions")
        return False


def test_with_validation():
    """Test that enhanced version with validation still gives same results."""

    print("\n" + "=" * 70)
    print("Validation Test: Enhanced with constraints enabled")
    print("=" * 70)

    # Valid parameters (should pass constraints)
    df = pd.DataFrame({"A": [12, 56, 208]})
    params_valid = {"c1": 0.496, "c2": 0.324}

    print("\n[Valid parameters]")
    print(f"  c1={params_valid['c1']}, c2={params_valid['c2']}")

    Q_legacy = predict_legacy(df, params_valid)
    Q_enhanced = predict_enhanced(df, params_valid)  # Validation enabled by default

    diff = np.abs(Q_legacy - Q_enhanced)
    max_diff = np.max(diff)

    print(f"  Legacy:   {Q_legacy}")
    print(f"  Enhanced: {Q_enhanced}")
    print(f"  Max diff: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("  ✓ PASS: Validation doesn't change results")
        return True
    else:
        print(f"  ✗ FAIL: Validation changed results by {max_diff}")
        return False


if __name__ == "__main__":
    passed1 = test_identical_results()
    passed2 = test_with_validation()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if passed1 and passed2:
        print("✅ Enhanced adapter is 100% backward compatible")
        print("   - Same results as legacy version")
        print("   - Validation doesn't change output")
        print("   - Safe to use in production")
        sys.exit(0)
    else:
        print("❌ Compatibility issues detected")
        sys.exit(1)

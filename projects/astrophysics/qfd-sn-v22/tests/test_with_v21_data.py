"""
Test V22 Core Modules with Existing V21 Filtered Data

This test script validates that the new V22 modules work correctly
by running them on the validated V21 filtered results.

Expected Results:
    - QFD parameters: k_J=121.34, η'=-0.04, ξ=-6.45, σ=1.64
    - All Lean constraints: PASS
    - RMS ≈ 1.77 mag
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from qfd_sn import cosmology
from qfd_sn.lean_validation import constraints, schema_interface
from qfd_sn import qc


def test_lean_validation_with_v21_results():
    """Test Lean validation using V21 filtered results."""

    print("=" * 80)
    print("TEST 1: Lean Validation with V21 Results")
    print("=" * 80)

    # V21 filtered best-fit parameters
    k_J_total = 121.34067381227166
    eta_prime = -0.038744051334258844
    xi = -6.448166102943284
    sigma_ln_A = 1.6362959528378436

    # Validate
    passed, results = constraints.validate_parameters(
        k_J_total=k_J_total,
        eta_prime=eta_prime,
        xi=xi,
        sigma_ln_A=sigma_ln_A
    )

    print(f"\\nParameters:")
    print(f"  k_J_total = {k_J_total:.4f} km/s/Mpc")
    print(f"  η' = {eta_prime:.4f}")
    print(f"  ξ  = {xi:.4f}")
    print(f"  σ_ln_A = {sigma_ln_A:.4f}")
    print(f"\\nValidation Results:")

    for param, (ok, msg) in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {msg}")

    print(f"\\nOverall: {'✅ ALL PASS' if passed else '❌ FAILED'}")

    assert passed, "V21 parameters should pass Lean validation"
    print("\\n✅ Test PASSED\\n")


def test_cosmology_calculations():
    """Test cosmology module calculations."""

    print("=" * 80)
    print("TEST 2: Cosmology Module Calculations")
    print("=" * 80)

    # Test with typical supernova
    z = 0.5
    k_J = 121.34
    eta_prime = -0.04
    xi = -6.45

    # Distance calculation
    D_mpc = cosmology.qfd_distance_mpc(z, k_J)
    mu_th = cosmology.qfd_distance_modulus(z, k_J)
    ln_A_pred = cosmology.ln_amplitude_predicted(z, eta_prime, xi)
    mu_qfd = cosmology.qfd_predicted_distance_modulus(z, k_J, eta_prime, xi)

    print(f"\\nFor z = {z}, k_J = {k_J:.2f} km/s/Mpc:")
    print(f"  Distance: {D_mpc:.2f} Mpc")
    print(f"  μ_th (distance-only): {mu_th:.4f} mag")
    print(f"  ln_A_predicted: {ln_A_pred:.4f}")
    print(f"  μ_QFD (with corrections): {mu_qfd:.4f} mag")

    # Sanity checks
    assert D_mpc > 0, "Distance should be positive"
    assert mu_th > 0, "Distance modulus should be positive"
    assert abs(ln_A_pred) < 10, "ln_A should be reasonable"

    print("\\n✅ Test PASSED\\n")


def test_schema_interface():
    """Test QFD schema interface."""

    print("=" * 80)
    print("TEST 3: QFD Schema Interface")
    print("=" * 80)

    # Create parameters object
    params = schema_interface.QFDParameters(
        k_J_correction=51.34,
        eta_prime=-0.04,
        xi=-6.45,
        sigma_ln_A=1.64
    )

    print(f"\\nQFD Parameters Object:")
    print(params)

    # Test conversions
    params_dict = params.to_dict()
    params_json = params.to_json()
    params_roundtrip = schema_interface.QFDParameters.from_json(params_json)

    print(f"\\nRoundtrip test:")
    print(f"  Original k_J_correction: {params.k_J_correction:.4f}")
    print(f"  Roundtrip k_J_correction: {params_roundtrip.k_J_correction:.4f}")

    # Schema compliance
    is_compliant = schema_interface.validate_schema_compliance(params)
    print(f"\\nSchema compliance: {'✅ PASS' if is_compliant else '❌ FAIL'}")

    assert is_compliant, "Parameters should be schema-compliant"
    assert abs(params.k_J_correction - params_roundtrip.k_J_correction) < 1e-6

    print("\\n✅ Test PASSED\\n")


def test_qc_gates():
    """Test quality control gates."""

    print("=" * 80)
    print("TEST 4: Quality Control Gates")
    print("=" * 80)

    # Create synthetic test data
    np.random.seed(42)
    n_sne = 1000

    data = pd.DataFrame({
        'chi2_dof': np.random.exponential(100, n_sne),
        'ln_A': np.random.normal(0, 2, n_sne),
        'stretch': np.random.normal(1, 0.3, n_sne),
        'n_epochs': np.random.randint(3, 20, n_sne)
    })

    # Add some failures
    data.loc[:50, 'chi2_dof'] = 5000  # Chi2 too high
    data.loc[51:100, 'ln_A'] = 25     # ln_A railed high
    data.loc[101:150, 'ln_A'] = -25   # ln_A railed low

    # Apply gates
    gates = qc.QualityGates(
        chi2_max=2000.0,
        ln_A_min=-20.0,
        ln_A_max=20.0,
        stretch_min=0.5,
        stretch_max=10.0,
        max_rejection_rate=0.30
    )

    qc_results = qc.apply_quality_gates(data, gates, verbose=True)

    print(f"\\nExpected ~150 failures (chi2 + ln_A railed)")
    print(f"Actual failures: {qc_results.n_failed}")
    print(f"\\nQC Status: {'✅ PASS' if qc_results.passed else '❌ FAIL'}")

    assert qc_results.n_failed >= 150, "Should catch synthetic failures"
    assert qc_results.passed, "Should pass with <30% rejection rate"

    print("\\n✅ Test PASSED\\n")


if __name__ == "__main__":
    print("\\n" + "=" * 80)
    print("V22 CORE MODULES TEST SUITE")
    print("Testing with V21 Validated Results")
    print("=" * 80 + "\\n")

    try:
        test_lean_validation_with_v21_results()
        test_cosmology_calculations()
        test_schema_interface()
        test_qc_gates()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\\nV22 core modules are working correctly!")
        print("Ready to integrate Stage 1-3 pipeline scripts.\\n")

    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}\\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

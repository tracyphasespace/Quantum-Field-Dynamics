"""
Regression Test: Ensure November 5, 2024 Results Can Be Reproduced

This test locks in the "golden" results from November 5, 2024 that match the papers.
If this test fails, something has broken the code!

Usage:
    pytest tests/test_regression_nov5.py -v

Or run directly:
    python tests/test_regression_nov5.py
"""

import json
import pytest
from pathlib import Path

# Golden reference results from November 5, 2024
# These match the paper values and should be reproducible
GOLDEN_RESULTS = {
    "k_J": 10.770038588319618,
    "k_J_std": 4.566720120697059,
    "eta_prime": -7.987900510670775,
    "eta_prime_std": 1.4390699801244529,
    "xi": -6.907618767280434,
    "xi_std": 3.745818404624118,
}

# Acceptable ranges (±30% for point estimates due to sampling variation)
ACCEPTABLE_RANGES = {
    "k_J": (7.5, 13.9),          # 10.7 ± 30%
    "eta_prime": (-10.4, -5.6),  # -8.0 ± 30%
    "xi": (-9.0, -4.8),          # -7.0 ± 30%
}

# Minimum uncertainty thresholds (to detect overfitting)
MIN_UNCERTAINTIES = {
    "k_J_std": 1.0,
    "eta_prime_std": 0.5,
    "xi_std": 1.0,
}


def load_latest_stage2_results():
    """Load the most recent Stage 2 results."""
    # Check common result locations
    search_paths = [
        Path("../results/v15_clean/stage2_fullscale/best_fit.json"),
        Path("../results/v15_clean/stage2_production/best_fit.json"),
        Path("../results/v15_clean/stage2_recovery_test/best_fit.json"),
        Path("../results/abc_comparison_20251105_165123/A_unconstrained/best_fit.json"),
    ]

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f), str(path)

    pytest.skip("No Stage 2 results found. Run pipeline first.")


def test_golden_reference_exists():
    """Verify the golden reference file exists."""
    golden_path = Path("../results/abc_comparison_20251105_165123/A_unconstrained/best_fit.json")
    assert golden_path.exists(), f"Golden reference not found: {golden_path}"

    with open(golden_path) as f:
        data = json.load(f)

    # Check it has the expected structure
    assert "k_J" in data
    assert "eta_prime" in data
    assert "xi" in data

    print(f"✓ Golden reference found: {golden_path}")


def test_k_J_in_range():
    """Test k_J is in acceptable range."""
    results, path = load_latest_stage2_results()
    k_J = results.get("k_J", 0)

    min_val, max_val = ACCEPTABLE_RANGES["k_J"]
    assert min_val < k_J < max_val, \
        f"k_J = {k_J:.2f} is OUT OF RANGE (expected {min_val} to {max_val})\n" \
        f"Results from: {path}"

    print(f"✓ k_J = {k_J:.2f} (expected ~10.7)")


def test_eta_prime_in_range():
    """Test eta_prime is in acceptable range."""
    results, path = load_latest_stage2_results()
    eta_prime = results.get("eta_prime", 0)

    min_val, max_val = ACCEPTABLE_RANGES["eta_prime"]
    assert min_val < eta_prime < max_val, \
        f"eta_prime = {eta_prime:.2f} is OUT OF RANGE (expected {min_val} to {max_val})\n" \
        f"Results from: {path}"

    print(f"✓ eta_prime = {eta_prime:.2f} (expected ~-8.0)")


def test_xi_in_range():
    """Test xi is in acceptable range."""
    results, path = load_latest_stage2_results()
    xi = results.get("xi", 0)

    min_val, max_val = ACCEPTABLE_RANGES["xi"]
    assert min_val < xi < max_val, \
        f"xi = {xi:.2f} is OUT OF RANGE (expected {min_val} to {max_val})\n" \
        f"Results from: {path}"

    print(f"✓ xi = {xi:.2f} (expected ~-7.0)")


def test_uncertainties_realistic():
    """Test that uncertainties are realistic (not overfitting)."""
    results, path = load_latest_stage2_results()

    for param, min_std in MIN_UNCERTAINTIES.items():
        std_val = results.get(param, 0)
        assert std_val > min_std, \
            f"{param} = {std_val:.4f} is too small (overfitting!)\n" \
            f"Expected > {min_std}\n" \
            f"Results from: {path}"

    print(f"✓ Uncertainties are realistic (no overfitting detected)")


def test_no_negative_sign_bug():
    """Test that the January 2025 negative sign bug is not present."""
    stage2_file = Path("stages/stage2_mcmc_numpyro.py")

    with open(stage2_file) as f:
        content = f.read()

    # Check for the incorrect negative sign
    # It should NOT appear in lines with "c = -jnp.array([k_J, eta_prime, xi])"
    # in the 'informed' or 'physics' constraint modes

    # This is a heuristic check - if this fails, investigate manually
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'c = -jnp.array([k_J, eta_prime, xi]) * scales' in line:
            # Check surrounding context to see if this is in informed/physics mode
            context = '\n'.join(lines[max(0, i-20):i+5])
            if "constrain_signs == 'informed'" in context or "constrain_signs == 'physics'" in context:
                pytest.fail(
                    f"REGRESSION DETECTED! Incorrect negative sign found at line {i}\n"
                    f"This is the January 2025 bug that breaks the code.\n"
                    f"Run: ./fix_regression.sh"
                )

    print("✓ No negative sign bug detected (code is correct)")


def test_stage2_used_enough_sne():
    """Test that Stage 2 used enough supernovae (not the 548 bug)."""
    results, path = load_latest_stage2_results()

    # Check if results include SNe count
    # This might be in a separate summary file
    summary_path = Path(path).parent / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            n_sne = summary.get("n_sne", 0)

            assert n_sne > 4000, \
                f"Only {n_sne} SNe used! Expected ~4,727\n" \
                f"This indicates the regression bug is present.\n" \
                f"Results from: {path}"

            print(f"✓ Using {n_sne} SNe (expected ~4,727)")
    else:
        # No summary file, can't check this
        pytest.skip("No summary.json found, can't verify SNe count")


def test_results_match_golden_roughly():
    """Test that results are within 50% of golden values (loose check)."""
    results, path = load_latest_stage2_results()

    for param, golden_val in GOLDEN_RESULTS.items():
        if param.endswith("_std"):
            continue  # Skip uncertainties for this test

        observed_val = results.get(param, 0)

        # Allow 50% deviation (very loose!)
        tolerance = abs(golden_val) * 0.5
        min_val = golden_val - tolerance
        max_val = golden_val + tolerance

        assert min_val < observed_val < max_val, \
            f"{param} = {observed_val:.2f} differs too much from golden {golden_val:.2f}\n" \
            f"Expected within {min_val:.2f} to {max_val:.2f}\n" \
            f"Results from: {path}"

    print("✓ Results are within 50% of golden values")


# Allow running as standalone script
if __name__ == "__main__":
    print("="*80)
    print("REGRESSION TEST: November 5, 2024 Golden Results")
    print("="*80)
    print()

    tests = [
        ("Golden reference exists", test_golden_reference_exists),
        ("k_J in range", test_k_J_in_range),
        ("eta_prime in range", test_eta_prime_in_range),
        ("xi in range", test_xi_in_range),
        ("Uncertainties realistic", test_uncertainties_realistic),
        ("No negative sign bug", test_no_negative_sign_bug),
        ("Enough SNe used", test_stage2_used_enough_sne),
        ("Results match golden (loose)", test_results_match_golden_roughly),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            print(f"Testing: {name}...", end=" ")
            test_func()
            print("PASS")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"SKIP - {e}")
            skipped += 1
        except AssertionError as e:
            print(f"FAIL")
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR")
            print(f"  {e}")
            failed += 1

    print()
    print("="*80)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*80)

    if failed > 0:
        print()
        print("⚠️  REGRESSION DETECTED!")
        print("Some tests failed. Your results may not match the papers.")
        print()
        print("Common issues:")
        print("  1. Negative sign bug (run ./fix_regression.sh)")
        print("  2. Wrong constraint mode (use --constrain-signs informed)")
        print("  3. Insufficient samples (use --nsamples 2000)")
        print()
        exit(1)
    elif passed > 0:
        print()
        print("✅ ALL TESTS PASSED!")
        print("Your results should match the paper values.")
        print()
        exit(0)
    else:
        print()
        print("⚠️  ALL TESTS SKIPPED")
        print("Run the pipeline first to generate results.")
        print()
        exit(0)

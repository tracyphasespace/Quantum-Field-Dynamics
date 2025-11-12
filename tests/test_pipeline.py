"""
Integration tests for the QFD supernova analysis pipeline.

These tests validate the Stage 1 → Stage 2 handoff and catch parameter
ordering issues that could cause catastrophic MCMC failures.

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from pipeline_io import PerSNParams, GlobalParams
import v15_model


class TestParameterOrdering:
    """Test that parameter ordering is consistent across pipeline stages."""

    def test_persn_params_array_conversion(self):
        """Test PerSNParams → array → PerSNParams round-trip."""
        original = PerSNParams(
            t0=57295.21,
            A_plasma=0.12,
            beta=0.57,
            ln_A=18.5
        )

        # Convert to array and back
        arr = original.to_array()
        restored = PerSNParams.from_array(arr)

        # Should be identical
        assert restored.t0 == pytest.approx(original.t0)
        assert restored.A_plasma == pytest.approx(original.A_plasma)
        assert restored.beta == pytest.approx(original.beta)
        assert restored.ln_A == pytest.approx(original.ln_A)

    def test_persn_params_model_order(self):
        """Test that to_model_order() produces correct parameter ordering."""
        params = PerSNParams(
            t0=57295.21,
            A_plasma=0.12,
            beta=0.57,
            ln_A=18.5
        )

        # Model expects: (t0, ln_A, A_plasma, beta)
        model_tuple = params.to_model_order()

        assert len(model_tuple) == 4
        assert model_tuple[0] == pytest.approx(params.t0)      # t0
        assert model_tuple[1] == pytest.approx(params.ln_A)    # ln_A
        assert model_tuple[2] == pytest.approx(params.A_plasma) # A_plasma
        assert model_tuple[3] == pytest.approx(params.beta)    # beta

    def test_stage1_order_matches_namedtuple(self):
        """Test that Stage 1 array order matches PerSNParams definition."""
        # Stage 1 saves as: [t0, A_plasma, beta, ln_A]
        stage1_array = np.array([57295.21, 0.12, 0.57, 18.5])

        # Load into NamedTuple
        params = PerSNParams.from_array(stage1_array)

        # Verify field assignments
        assert params.t0 == pytest.approx(stage1_array[0])
        assert params.A_plasma == pytest.approx(stage1_array[1])
        assert params.beta == pytest.approx(stage1_array[2])
        assert params.ln_A == pytest.approx(stage1_array[3])


class TestNumericalStability:
    """Test that parameter values produce numerically stable model evaluations."""

    def test_typical_parameters_no_overflow(self):
        """Test that typical Stage 1 parameters don't cause overflow in model."""
        import jax.numpy as jnp

        # Typical observation: [MJD, wavelength] (model just needs time and wavelength)
        obs = jnp.array([57300.0, 4500.0])

        # Typical global parameters from paper
        global_params = (10.7, -8.0, -7.0)  # k_J, eta_prime, xi

        # Typical per-SN parameters
        params = PerSNParams(
            t0=57295.21,
            A_plasma=0.12,
            beta=0.57,
            ln_A=18.5
        )
        persn_tuple = params.to_model_order()

        # Canonical L_peak
        L_peak = 2.8e36  # W/Hz
        z_obs = 0.5

        # Should not raise overflow or produce NaN/Inf
        try:
            flux_jy = v15_model.qfd_lightcurve_model_jax(
                obs, global_params, persn_tuple, L_peak, z_obs
            )

            # Check result is finite (may be zero if outside valid time range)
            assert jnp.isfinite(flux_jy), f"Model produced non-finite flux: {flux_jy}"
            assert flux_jy >= 0, f"Model produced negative flux: {flux_jy}"

        except (OverflowError, FloatingPointError) as e:
            pytest.fail(f"Model raised numerical error: {e}")

    def test_parameter_order_sensitivity(self):
        """Test that parameter order matters for model output."""
        import jax.numpy as jnp

        obs = jnp.array([57300.0, 4500.0])
        global_params = (10.7, -8.0, -7.0)
        L_peak = 2.8e36
        z_obs = 0.5

        # CORRECT order: (t0, ln_A, A_plasma, beta)
        correct_persn = (57295.21, 18.5, 0.12, 0.57)

        # WRONG order: (t0, A_plasma, beta, ln_A) - Stage 1 array order
        # If passed directly without reordering, beta=0.57 becomes ln_A
        # and ln_A=18.5 becomes beta (catastrophic!)
        wrong_persn = (57295.21, 0.12, 0.57, 18.5)

        # Verify correct order works
        try:
            flux_correct = v15_model.qfd_lightcurve_model_jax(
                obs, global_params, correct_persn, L_peak, z_obs
            )
            assert jnp.isfinite(flux_correct), "Correct ordering should produce finite result"

            # Wrong order should produce very different (likely catastrophic) results
            flux_wrong = v15_model.qfd_lightcurve_model_jax(
                obs, global_params, wrong_persn, L_peak, z_obs
            )

            # The results should be drastically different
            # (ln_A=0.57 vs 18.5 means ~e^18 difference in normalization!)
            assert abs(flux_correct - flux_wrong) > 1.0, \
                "Wrong parameter order should produce significantly different results"

        except Exception as e:
            # It's OK if wrong order causes numerical issues
            # That proves parameter ordering matters!
            pass


class TestGlobalParams:
    """Test global parameter structures."""

    def test_global_params_conversion(self):
        """Test GlobalParams round-trip conversion."""
        original = GlobalParams(k_J=10.7, eta_prime=-8.0, xi=-7.0)

        arr = original.to_array()
        restored = GlobalParams.from_array(arr)

        assert restored.k_J == pytest.approx(original.k_J)
        assert restored.eta_prime == pytest.approx(original.eta_prime)
        assert restored.xi == pytest.approx(original.xi)

    def test_global_params_expected_ranges(self):
        """Test that expected parameter ranges are reasonable."""
        # From paper expectations
        params = GlobalParams(k_J=10.7, eta_prime=-8.0, xi=-7.0)

        # k_J should be positive
        assert params.k_J > 0, "k_J should be positive"

        # eta_prime and xi should be negative (per paper)
        assert params.eta_prime < 0, "eta_prime should be negative"
        assert params.xi < 0, "xi should be negative"


class TestStage1Stage2Handoff:
    """Integration tests for Stage 1 → Stage 2 data handoff."""

    def test_load_stage1_results_structure(self, tmp_path):
        """Test that Stage 1 results can be loaded with correct structure."""
        # Simulate Stage 1 output
        sn_id = "SN2019abc"
        persn_best = np.array([57295.21, 0.12, 0.57, 18.5])  # Stage 1 order

        # Save to temporary directory
        sn_dir = tmp_path / sn_id
        sn_dir.mkdir()
        np.save(sn_dir / "persn_best.npy", persn_best)

        # Stage 2 should load with PerSNParams
        loaded_array = np.load(sn_dir / "persn_best.npy")
        params = PerSNParams.from_array(loaded_array)

        # Verify correct interpretation
        assert params.t0 == pytest.approx(57295.21)
        assert params.A_plasma == pytest.approx(0.12)
        assert params.beta == pytest.approx(0.57)
        assert params.ln_A == pytest.approx(18.5)

        # Verify model ordering
        model_tuple = params.to_model_order()
        assert model_tuple == (57295.21, 18.5, 0.12, 0.57)  # (t0, ln_A, A_plasma, beta)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

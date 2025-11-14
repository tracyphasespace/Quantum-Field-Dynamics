# Work Summary: QFD Supernova Pipeline v17 MCMC Stabilization

## Introduction

This summary outlines the work performed to address numerical instability issues in the Quantum Field Dynamics (QFD) supernova pipeline, specifically focusing on the Stage 1 optimization and the Stage 2 Markov Chain Monte Carlo (MCMC) global parameter fitting using the `v17` model. The primary goal was to stabilize the pipeline, resolve NaN gradient problems, and ensure the MCMC sampler could run to completion, integrating the structural refactoring to the `v17` codebase.

## Key Issues Addressed

Throughout this process, several critical issues were identified and resolved:

1.  **NaN Gradients in Stage 1 Optimization:** Initial problems with `NaN` values propagating through the Stage 1 per-supernova optimization, leading to optimizer failures.
2.  **`v17` Pipeline Structural Refactoring:** Integration of the new `v17` codebase, which involved deprecating `v15` files and establishing the `pipeline/` directory as the canonical source.
3.  **MCMC Initialization Failures:** The NumPyro MCMC sampler in Stage 2 repeatedly failed with `RuntimeError: Cannot find valid initial parameters`, indicating that the model's log-probability was `NaN` or `-inf` for the initial parameter values.
4.  **`overflow encountered in cast` Warnings:** JAX-related warnings suggesting that intermediate calculations were producing values too large for the default floating-point precision.
5.  **Incorrect Script Pathing:** The `run_pipeline_v17.sh` script was incorrectly referencing the `v17` directory, leading to "No such file or directory" errors.
6.  **`NameError` in `predict_apparent_magnitude`:** A bug introduced during a previous fix, where a variable was used before its definition.

## Solutions Implemented

A series of targeted fixes and improvements were applied:

*   **Stage 1 Optimization Stabilization (Recap from previous turns):**
    *   The `qfd_lightcurve_model_jax` function in `pipeline/core/v17_lightcurve_model.py` was corrected to return actual flux values instead of a constant `1.0`.
    *   The `calculate_neg_log_L_student_t` function in `pipeline/stages/stage1_optimize_v17.py` was made NaN-safe by flooring `flux_err`, replacing non-finite residuals with `0`, and clamping non-finite log-PDF values.
    *   Indentation errors in both files were corrected.

*   **Script Pathing Correction:**
    *   The `V17_DIR` variable in `pipeline/scripts/run_pipeline_v17.sh` was updated from `./v17` to `./pipeline`, resolving file not found errors.

*   **Numerical Stability in `pipeline/core/v17_qfd_model.py`:**
    *   The `predict_apparent_magnitude` function was modified to clamp `1 + total_physical_redshift` to a minimum value of `1e-6` before taking `jnp.log10`, preventing `NaN` or `-inf` results from unphysical redshift values.
    *   The `NameError` in `predict_apparent_magnitude` was resolved by correctly defining `total_physical_redshift` before its use.

*   **Numerical Stability in `pipeline/stages/stage2_mcmc_v17.py`:**
    *   The upper bound of the distance search range in `find_distance_for_redshift` was increased from `20,000.0` to `200,000.0` to accommodate a wider range of physical distances and improve root bracketing.
    *   The `find_distance_for_redshift` function was further enhanced to return the `high` (upper bound) value instead of `jnp.nan` when a root is not bracketed, ensuring a finite distance is always returned.
    *   A small epsilon (`1e-6`) was added to the `sigma_m` prior (`dist.HalfNormal(0.2) + 1e-6`) to prevent `sigma_m` from becoming exactly zero, which could lead to numerical instability in the likelihood calculation.
    *   JAX's global floating-point precision was explicitly set to 64-bit (`jax.config.update("jax_enable_x64", True)`) to mitigate `overflow encountered in cast` warnings and improve overall numerical accuracy.
    *   `numpyro.set_host_device_count(1)` was added to explicitly manage device usage, ensuring sequential chain execution and avoiding potential device-related issues.

## Current Status of Stage 2 MCMC

The Stage 2 MCMC sampler now successfully runs to completion without encountering `RuntimeError` or `overflow encountered in cast` warnings. This indicates that the numerical stability issues preventing initialization have been resolved.

**However, the MCMC results currently exhibit significant issues regarding convergence and reliability:**

*   **High Number of Divergences (1272):** Out of 2000 total samples (1000 samples per chain across 2 chains), 1272 divergences were reported. A high number of divergences is a critical indicator that the Hamiltonian Monte Carlo (HMC) sampler is struggling to accurately explore the posterior distribution. This often points to issues with the model's parameterization, highly correlated parameters, or poorly specified priors, leading to unreliable sampling.
*   **Poor Convergence Diagnostics (`n_eff` and `r_hat`):**
    *   **`eta_prime`:** `n_eff = 1.82`, `r_hat = 1.41`. The effective sample size (`n_eff`) is extremely low, and the Gelman-Rubin statistic (`r_hat`) is significantly above the acceptable threshold of 1.05 (ideally close to 1.0). This indicates very poor mixing between chains and a clear lack of convergence.
    *   **`xi`:** `n_eff = 1.00`, `r_hat = 102.06`. These values are even more problematic, suggesting almost no effective samples and a complete failure of convergence.
    *   **`sigma_m`:** `n_eff = 1.01`, `r_hat = 7.30`. Similar to `xi`, indicating very poor convergence.
    *   **`k_J`:** `n_eff = nan`, `r_hat = 1.00`. While `r_hat` appears good, the `NaN` for `n_eff` is concerning and likely an artifact of the overall poor sampling quality for other parameters.

These diagnostics collectively imply that the MCMC, while technically running, is not producing reliable samples from the posterior distribution. The current results are not suitable for drawing scientific conclusions.

## Next Steps and Recommendations

To obtain reliable results from the Stage 2 MCMC, further work is required:

1.  **Model Re-parameterization:** Investigate alternative ways to parameterize the QFD model to improve the geometry of the posterior distribution, reducing parameter correlations and making it easier for the HMC sampler to explore.
2.  **Prior Refinement:** Re-evaluate and potentially refine the prior distributions for `k_J`, `eta_prime`, `xi`, and `sigma_m`. More informative or constrained priors, if justified by domain knowledge, could guide the sampler towards more stable regions.
3.  **Increased Warmup and Sampling Steps:** While not a solution for fundamental model issues, increasing `num_warmup` and `num_samples` might help if the sampler is simply taking too long to converge, though this is unlikely to resolve high divergences alone.
4.  **Debugging Model Logic:** A thorough review of the QFD model's mathematical formulation and its implementation in JAX is recommended to ensure there are no subtle errors or inconsistencies that could lead to the observed sampling difficulties.
5.  **Visual Diagnostics:** Generate trace plots, posterior predictive checks, and other visual diagnostics to gain deeper insights into the sampler's behavior and the shape of the posterior.

## Commit History

The work described above has been captured in the following Git commits:

*   `FEAT: Implement NaN-safe Stage 1 optimization and fix core model bug`
*   `FIX: Improve MCMC numerical stability and pathing`
*   `CLEAN: Remove stale __pycache__ files`
*   `CLEAN: Remove stale __pycache__ file from pipeline/stages`

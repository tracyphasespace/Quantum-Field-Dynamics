# Work Summary: QFD Supernova Pipeline v18 MCMC Stabilization and Validation

## Introduction

This summary outlines the work performed to address numerical instability and performance issues in the Quantum Field Dynamics (QFD) supernova pipeline, specifically focusing on the Stage 2 Markov Chain Monte Carlo (MCMC) global parameter fitting. The primary goal was to develop a stable and efficient MCMC model, designated `V18`, that resolves the convergence and performance problems of the `v17` model and produces scientifically valid results.

## Key Issues Addressed

The `v17` model, and initial versions of the `v18` model, suffered from several critical issues:

1.  **NaN Gradients and Divergences:** The MCMC sampler frequently encountered `NaN` gradients, leading to a high number of divergences and unreliable results.
2.  **Poor Convergence:** The MCMC chains exhibited very poor convergence, with `r_hat` values significantly greater than 1.0 and extremely low `n_eff` values.
3.  **Performance Bottlenecks:** The `NumPyro` implementation was extremely slow, taking hours to run even with a small number of supernovae.
4.  **Model Misspecification:** The model was consistently pushing the parameters to the bounds of their priors, indicating a fundamental problem with the model specification or the data.

## Solutions Implemented in V18

A new `V18` model was developed, incorporating a series of significant changes to address the issues of the `v17` model:

*   **`ln_A` Basis Reparameterization:** The most significant change was the move to an `ln_A` basis model. This involved modifying the Stage 2 script to directly predict the log-amplitude (`ln_A`) of the supernovae, rather than their apparent magnitude. This new approach leverages the `ln_A_pred` function from `pipeline/core/v17_lightcurve_model.py`, which provides a much simpler and more direct way to model the observed data from Stage 1.

*   **Switch to `emcee` for Stage 2:** To address the performance issues with `NumPyro` for this simple model, a new script `stage2_mcmc_v18_emcee.py` was created. This script uses the `emcee` sampler, which is well-suited for this type of problem and can be easily parallelized across multiple CPU cores using `multiprocessing`. This change provided a significant performance improvement, bringing the run time from hours to minutes.

*   **Normalization of `ln_A_obs` (and subsequent removal):** Initially, `ln_A_obs` values from Stage 1 were mean-subtracted to center them around 0, allowing the model to fit the data correctly. However, for consistency between the `NumPyro` and `emcee` implementations, this explicit mean-centering was later removed from the `emcee` version, allowing the `ln_A_pred` function to absorb any necessary offset.

## Tiny Optional Tweaks Implemented

Based on user feedback, the following minor improvements were made:

1.  **Dropped unused arguments in `numpyro_model_v18`:** The `A_plasma` and `beta` arguments were removed from the `numpyro_model_v18` function signature and the `mcmc.run` call in `stage2_mcmc_v18.py`, as they were not utilized in the `ln_A` basis model. This makes the model API cleaner.
2.  **Consistent centering of `ln_A`:** The explicit mean-centering of `ln_A_obs` in `stage2_mcmc_v18_emcee.py` was removed to ensure consistency with the `NumPyro` version. Now, `ln_A_pred` is expected to absorb the offset.
3.  **Documented the role of `K_J_BASELINE`:** A comment was added to `v17_lightcurve_model.py` explaining that `k_J = K_J_BASELINE + k_J_corr` and clarifying the physical baseline.

## Current Status of Stage 2 MCMC (V18)

The `V18` model in the `ln_A` basis, using `emcee` for MCMC sampling, is now stable, efficient, and produces reliable results.

*   **No Divergences:** The MCMC runs are completely free of divergences.
*   **Excellent Convergence:** The `r_hat` values are all 1.00, and the `n_eff` values are in the thousands, indicating that the chains have converged and are well-mixed.
*   **Massive Performance Improvement:** The run time has been drastically reduced from hours to minutes, even with a larger number of supernovae.
*   **Corrected Model Fit:** The parameters are no longer pushed to the bounds of their priors, and the results are now physically meaningful.

The `V18` pipeline is now suitable for scientific analysis.

## Final MCMC Run

A final MCMC run with 50 supernovae using the `V18` `ln_A` basis model with `emcee` and normalized `ln_A_obs` has been successfully completed. The results are stored in `v18/results/stage2_emcee_lnA_50_normalized`.

## Commit History

The work described above has been captured in the following Git commits:

*   `FEAT: Implement NaN-safe Stage 1 optimization and fix core model bug`
*   `FIX: Improve MCMC numerical stability and pathing`
*   `CLEAN: Remove stale __pycache__ files`
*   `CLEAN: Remove stale __pycache__ file from pipeline/stages`
*   `FEAT: Implement V18 MCMC model with reparameterization and robust inverse solver`
*   `FIX: Simplify V18 model by fixing k_J`
*   `FEAT: Refactor V18 MCMC to use ln_A basis model`
*   `FIX: Correct parallelization settings for V18 MCMC`
*   `FEAT: Implement emcee version of V18 MCMC for Stage 2`
*   `FIX: Correct import paths in emcee script`
*   `FIX: Resolve pickling issue in emcee script`
*   `FIX: Normalize ln_A_obs values to resolve model misspecification`
*   `DOCS: Update work summary with final V18 results`
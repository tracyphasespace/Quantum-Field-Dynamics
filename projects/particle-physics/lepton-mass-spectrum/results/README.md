# Results Directory

This directory contains outputs from MCMC parameter estimation runs.

## Generated Files

When you run `scripts/run_mcmc.py`, the following files will be created:

- **`mcmc_chain.h5`**: Full MCMC chain in HDF5 format
  - Contains all walker positions for all steps
  - Can be loaded with `emcee.backends.HDFBackend`
  - Size: ~10-50 MB depending on run length

- **`results.json`**: Parameter estimates in JSON format
  - Median, standard deviation, and percentiles for each parameter
  - Correlation coefficients
  - Metadata (timestamp, model settings)

- **`corner_plot.png`**: Posterior visualization
  - Marginalized distributions (diagonal)
  - 2D correlations (off-diagonal)
  - Contours at 68% and 95% credible intervals

## Example Results

The file `example_results.json` contains results from a reference run for comparison.

Expected parameter values:
```
β = 3.063 ± 0.149
ξ = 0.966 ± 0.549
τ = 1.007 ± 0.658
```

Your results should match these to within ~0.01 for β and ~0.05 for ξ, τ (accounting for MCMC stochasticity).

## .gitignore

Generated files (`.h5`, `.png`, `.json` except `example_results.json`) are excluded from version control via `.gitignore` to keep the repository clean.

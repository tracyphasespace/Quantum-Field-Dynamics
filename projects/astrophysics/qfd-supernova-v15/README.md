# QFD Supernova Analysis V15

GPU-accelerated 3-stage pipeline for fitting Type Ia supernova lightcurves with Quantum Field Dynamics (QFD) cosmology.

## Overview

V15 implements a hierarchical Bayesian fitting pipeline that optimizes per-supernova nuisance parameters (Stage 1), then fits global cosmological parameters via MCMC (Stage 2), and finally generates Hubble diagrams comparing QFD vs ΛCDM (Stage 3).

## Key Results (5,124 Quality SNe)

- **QFD RMS Residual**: 1.20 mag
- **ΛCDM RMS Residual**: 3.48 mag
- **Improvement**: 65.4% better fit with QFD
- **Best-fit Parameters**: k_J = 70.0, η' = 0.0102, ξ = 30.0

## Architecture

### Stage 1: Per-SN Nuisance Parameter Optimization
- **Input**: Lightcurve photometry (5,468 SNe)
- **Method**: JAX gradients + L-BFGS-B optimizer on GPU
- **Optimizes**: t₀ (explosion epoch), A_plasma (plasma veil), β (wavelength dependence), α (distance lever)
- **Output**: Per-SN best-fit parameters + uncertainties
- **Runtime**: ~3 hours on GPU

**Critical Fix**: L_peak frozen at 1.5×10⁴³ erg/s to break degeneracy with α

### Stage 2: Global Parameter MCMC
- **Input**: Stage 1 results (quality-filtered)
- **Method**: NumPyro NUTS sampler (GPU-accelerated)
- **Samples**: 4 chains × 2,000 steps = 8,000 posterior samples
- **Fits**: Global QFD parameters (k_J, η', ξ)
- **Output**: Posterior distributions + best-fit values
- **Runtime**: ~10-15 minutes

### Stage 3: Hubble Diagram Generation
- **Input**: Stages 1 & 2 results
- **Method**: Parallel distance modulus calculation (multiprocessing)
- **Compares**: QFD vs ΛCDM predictions
- **Output**: Hubble diagram, residual plots, statistics
- **Runtime**: ~5 minutes

## Quick Start

### Prerequisites
```bash
# Python 3.9+
pip install jax jaxlib numpyro pandas numpy scipy matplotlib
```

### Run Full Pipeline
```bash
# Stage 1: Optimize per-SN parameters (parallel)
./scripts/run_stage1_parallel.sh \
    path/to/lightcurves.csv \
    results/stage1 \
    70,0.01,30 \
    7  # workers

# Stage 2: MCMC for global parameters
./scripts/run_stage2_numpyro_production.sh

# Stage 3: Generate Hubble diagram
python src/stage3_hubble_optimized.py \
    --stage1-results results/stage1 \
    --stage2-results results/stage2 \
    --lightcurves path/to/lightcurves.csv \
    --out results/stage3 \
    --ncores 7
```

## Data Format

Lightcurves CSV must contain:
- `snid`: Supernova ID
- `mjd`: Modified Julian Date
- `flux_[band]`: Flux in each band (e.g., `flux_g`, `flux_r`)
- `fluxerr_[band]`: Flux uncertainty
- `z`: Redshift

## Project Structure

```
qfd-supernova-v15/
├── src/
│   ├── stage1_optimize.py          # Stage 1: per-SN optimization
│   ├── stage2_mcmc_numpyro.py      # Stage 2: MCMC sampling
│   ├── stage3_hubble_optimized.py  # Stage 3: Hubble diagram
│   ├── v15_model.py                # QFD lightcurve model
│   ├── v15_data.py                 # Data loading utilities
│   ├── v15_config.py               # Configuration
│   └── v15_*.py                    # Supporting modules
├── scripts/
│   ├── run_full_pipeline.sh        # Automated 3-stage runner
│   ├── run_stage1_parallel.sh      # Parallel Stage 1 execution
│   └── check_pipeline_status.sh    # Progress monitoring
├── docs/
│   ├── V15_Architecture.md         # Detailed architecture
│   ├── V15_FINAL_VERDICT.md        # Validation & results
│   └── FINAL_RESULTS_SUMMARY.md    # Summary statistics
└── results/                        # Output directory (gitignored)
```

## Key Fixes in V15

1. **L_peak/α Degeneracy**: Frozen L_peak at canonical value to allow α to encode distance variations
2. **Dynamic t₀ Bounds**: Per-SN bounds based on observed MJD range (fixes χ² = 66B failures)
3. **Multiprocessing Optimization**: Configurable worker count to avoid OOM on limited RAM systems

## Performance

- **Stage 1**: 5,468 SNe in ~3 hours (0.5 SNe/sec with GPU)
- **Stage 2**: 8,000 MCMC samples in ~12 minutes
- **Stage 3**: 5,124 distance moduli in ~5 minutes (16 cores)
- **Total**: ~3.5 hours for full pipeline

## References

See `docs/` for detailed technical documentation and validation results.

## License

Part of the Quantum Field Dynamics research project.

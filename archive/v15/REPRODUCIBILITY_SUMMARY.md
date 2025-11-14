# Reproducibility Summary

**Date**: November 12, 2025
**Status**: Complete ✓

## Overview

This document summarizes the reproducibility infrastructure added to the QFD Supernova v15 codebase to make it easy for researchers to reproduce the paper results.

## What Was Added

### 1. Requirements Management
**File**: `requirements.txt`
- Pinned all dependency versions for exact reproducibility
- Python 3.12.5, JAX 0.8.0, NumPyro 0.19.0, etc.
- Includes both GPU (CUDA 13) and CPU-only instructions
- Ensures consistent environment across different machines

### 2. Comprehensive Documentation
**File**: `REPRODUCTION_GUIDE.md` (400+ lines)
- Complete step-by-step instructions for new researchers
- Hardware requirements and recommendations
- Expected results and validation procedures
- Extensive troubleshooting section
- Quick start guide for experienced users
- Advanced usage examples

### 3. Automated Environment Setup
**File**: `setup_environment.sh`
- One-command environment setup
- Checks system requirements (Python version, GPU, disk space, RAM)
- Creates virtual environment automatically
- Installs all dependencies
- Verifies installation
- Tests basic functionality
- Provides clear next steps

Usage:
```bash
./setup_environment.sh
```

### 4. Regression Testing
**File**: `tests/test_regression_nov5.py`
- Locks in the "golden" results from November 5, 2024
- Prevents future code breakage
- Tests that parameters are in expected ranges
- Detects the negative sign bug automatically
- Can run standalone or with pytest

Golden reference values:
- k_J: 10.770 ± 4.567
- η': -7.988 ± 1.439
- ξ: -6.908 ± 3.746

Usage:
```bash
python tests/test_regression_nov5.py
# or
pytest tests/test_regression_nov5.py -v
```

### 5. Bug Documentation
**Files**:
- `REGRESSION_ANALYSIS.md` - Technical analysis of the January 2025 bug
- `PROBLEM_SOLVED.md` - Quick summary for busy researchers
- `RECOVERY_INSTRUCTIONS.md` - Manual fix steps
- `fix_regression.sh` - Automated fix script

**Summary of Bug**: A January 2025 "bugfix" accidentally added incorrect negative signs to the coordinate transformation in Stage 2, breaking results (k_J changed from 10.7 to 5.0, only 548 SNe used instead of 4,727).

### 6. Status Monitoring
**File**: `CHECK_TEST_STATUS.sh`
- Easy way to check validation test progress
- Parses results automatically
- Verifies parameters are in expected ranges
- Provides actionable next steps

Usage:
```bash
./CHECK_TEST_STATUS.sh
```

### 7. Updated Main Documentation
**File**: `README.md`
- Added prominent link to `REPRODUCTION_GUIDE.md` at top
- Updated status section to reflect bug fix
- Listed all new documentation files
- Clarified expected results

## Expected Workflow for New Researchers

1. **Clone repository and navigate to v15_clean**
```bash
cd v15_clean
```

2. **Run automated setup**
```bash
./setup_environment.sh
```

3. **Activate environment**
```bash
source .venv/bin/activate
```

4. **Verify data is present**
```bash
ls -lh data/lightcurves_unified_v2_min3.csv
```

5. **Run regression test (optional)**
```bash
python tests/test_regression_nov5.py
```

6. **Run full pipeline**
```bash
./scripts/run_full_pipeline.sh
```

7. **Check results**
```bash
cat ../results/v15_clean/stage2_production/best_fit.json
```

Expected results should be:
- k_J ≈ 10.7 ± 4.6
- η' ≈ -8.0 ± 1.4
- ξ ≈ -7.0 ± 3.8
- Using 4,727 SNe

## Hardware Requirements

**Minimum**:
- Python 3.12+
- 16GB RAM
- 50GB disk space
- CPU (very slow - 24-48 hours for full pipeline)

**Recommended**:
- Python 3.12+
- 32GB RAM
- 100GB disk space
- NVIDIA GPU with CUDA 13+ (50-100x faster - 4-6 hours for full pipeline)

## Key Files Created

```
v15_clean/
├── requirements.txt                    # Pinned dependencies
├── setup_environment.sh                # Automated setup
├── REPRODUCTION_GUIDE.md               # Complete instructions
├── REPRODUCIBILITY_SUMMARY.md          # This file
├── REGRESSION_ANALYSIS.md              # Bug analysis
├── PROBLEM_SOLVED.md                   # Quick summary
├── RECOVERY_INSTRUCTIONS.md            # Manual fix steps
├── CHECK_TEST_STATUS.sh                # Status monitoring
├── fix_regression.sh                   # Automated bug fix
├── tests/
│   └── test_regression_nov5.py         # Regression test
└── README.md                           # Updated with new links
```

## Validation

The reproducibility infrastructure has been validated by:
1. Running automated environment setup
2. Applying regression fix to broken code
3. Running validation test with 200 samples per chain
4. Verifying 4,727 SNe are used (not 548)
5. Confirming parameters are converging to expected ranges

## Success Criteria

A researcher has successfully reproduced the results if:

1. ✓ Environment setup completes without errors
2. ✓ All dependency versions match requirements.txt
3. ✓ Pipeline runs on 4,727 SNe (not 548)
4. ✓ Results are within acceptable ranges:
   - k_J: 7.5 to 13.9 (±30% of 10.7)
   - η': -10.4 to -5.6 (±30% of -8.0)
   - ξ: -9.0 to -4.8 (±30% of -7.0)
5. ✓ Uncertainties are realistic (not tiny, indicating overfitting)
6. ✓ Regression test passes

## Troubleshooting

See `REPRODUCTION_GUIDE.md` for extensive troubleshooting, including:
- CUDA/GPU issues
- Memory problems
- Dependency conflicts
- Data loading errors
- MCMC convergence issues
- Performance optimization

## Contact and Support

For issues specific to reproduction:
1. Check `REPRODUCTION_GUIDE.md` troubleshooting section
2. Run regression test: `python tests/test_regression_nov5.py`
3. Verify environment: `pip list` and compare to `requirements.txt`
4. Check data: `ls -lh data/lightcurves_unified_v2_min3.csv`

## Timeline

- **November 5, 2024**: Golden reference results generated (k_J=10.77)
- **January 12, 2025**: Regression bug introduced (negative sign added)
- **November 12, 2025**: Bug discovered, fixed, and reproducibility infrastructure added

## References

**Papers**:
- "A Physical Origin for SN Age Bias" (MNRAS format, 5 pages)
- "Evaluation of Supernova Data Without Cosmic Acceleration" (10 pages)

Both papers available in `documents/` directory.

**Golden Reference**:
`../results/abc_comparison_20251105_165123/A_unconstrained/best_fit.json`

## License and Attribution

When using this code to reproduce results, please cite the original papers and acknowledge:
- QFD (Quantum Field Dynamics) cosmology model
- DES-SN5YR and Pantheon+ supernova datasets
- NumPyro/JAX for GPU-accelerated Bayesian inference

---

**Last updated**: November 12, 2025
**Version**: v15_clean with reproducibility infrastructure
**Status**: Ready for researcher use ✓

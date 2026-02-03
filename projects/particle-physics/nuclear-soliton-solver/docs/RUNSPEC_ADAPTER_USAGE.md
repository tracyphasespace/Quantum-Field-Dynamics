# RunSpec Adapter Usage Guide

## Overview

The RunSpec adapter (`src/runspec_adapter.py`) bridges the QFD RunSpec v0 schema system with the Phase 9 SCF nuclear soliton solver. It enables:

- **Schema-compliant experiment specifications**: JSON-based configuration with validation
- **Reproducible experiments**: Complete provenance tracking
- **Parameter optimization**: Integration with scipy optimizers
- **Grand Solver compatibility**: Standardized interface for multi-domain optimization

## Quick Start

### 1. Evaluation Mode (Fixed Parameters)

Evaluate Trial 32 parameters on test isotopes:

```bash
python src/runspec_adapter.py experiments/test_trial32_eval.runspec.json
```

### 2. Optimization Mode (Parameter Search)

Optimize parameters for heavy isotopes:

```bash
python src/runspec_adapter.py experiments/nuclear_heavy_region.runspec.json
```

## RunSpec File Structure

A minimal RunSpec file contains:

```json
{
  "schema_version": "v0.1",
  "experiment_id": "my_experiment",
  "description": "Brief description",

  "model": {
    "id": "qfd.nuclear.binding.soliton",
    "variant": "phase9_scf"
  },

  "parameters": [
    {
      "name": "nuclear.c_v2_base",
      "value": 2.201711,
      "role": "coupling",
      "bounds": [2.0, 3.0],
      "frozen": false
    }
  ],

  "datasets": [
    {
      "id": "ame2020",
      "source": "../data/ame2020_system_energies.csv",
      "cuts": {"A_range": [1, 250]}
    }
  ],

  "solver": {
    "method": "scipy.differential_evolution",
    "options": {"maxiter": 100}
  }
}
```

## Parameter Specification

### Required Parameters

The solver requires these parameters (defaults provided if not specified):

- `nuclear.c_v2_base`: Cohesion baseline (Trial 32: 2.201711)
- `nuclear.c_v2_iso`: Isospin cohesion (Trial 32: 0.027035)
- `nuclear.c_v2_mass`: Mass compounding (Trial 32: -0.000205)
- `nuclear.c_v4_base`: Repulsion baseline (Trial 32: 5.282364)
- `nuclear.c_v4_size`: Size repulsion (Trial 32: -0.085018)
- `nuclear.c_sym`: Asymmetry coefficient (Trial 32: 25.0)

### Additional Parameters (Auto-Defaulted)

- `nuclear.c_v4_iso`: Isospin repulsion (default: 0.005164)
- `nuclear.alpha_e_scale`: Electron scaling (default: 1.007419)
- `nuclear.beta_e_scale`: Beta scaling (default: 0.504312)
- `nuclear.kappa_rho`: Density coupling (default: 0.029816)
- `nuclear.c_coul`: Coulomb coefficient (default: 0.801463)
- `nuclear.c_surf`: Surface energy (default: 18.5)
- `nuclear.c_pair_even`: Even pairing (default: 12.0)
- `nuclear.c_pair_odd`: Odd pairing (default: -12.0)
- `nuclear.c_shell`: Shell correction (default: 0.0)

### Parameter Roles

- `"coupling"`: Physical coupling constants to optimize
- `"nuisance"`: Systematic parameters
- `"fixed"`: Constants (e.g., speed of light)
- `"derived"`: Computed from other parameters

### Frozen vs Optimizable

```json
{
  "name": "nuclear.c_v2_base",
  "value": 2.201711,
  "frozen": false,          // Will be optimized
  "bounds": [2.0, 3.0]      // Search range
}
```

```json
{
  "name": "nuclear.c_sym",
  "value": 25.0,
  "frozen": true,           // Fixed at initial value
  "bounds": null            // No optimization
}
```

## Dataset Configuration

### Basic Dataset

```json
{
  "id": "ame2020_full",
  "source": "../data/ame2020_system_energies.csv",
  "cuts": {
    "A_range": [1, 250],
    "stable_only": false
  }
}
```

### Regional Selection

For heavy isotope calibration:

```json
{
  "id": "heavy_nuclei",
  "source": "../data/ame2020_system_energies.csv",
  "cuts": {
    "A_range": [120, 250]
  },
  "description": "Heavy isotopes A ≥ 120"
}
```

### Column Mapping

If your dataset uses different column names:

```json
{
  "columns": {
    "mass_number": "A",
    "atomic_number": "Z",
    "binding_energy": "BE_MeV"
  }
}
```

## Solver Configuration

### Evaluation Only

```json
{
  "solver": {
    "method": "evaluation"
  }
}
```

All parameters must have `"frozen": true`.

### Differential Evolution (Global Search)

```json
{
  "solver": {
    "method": "scipy.differential_evolution",
    "options": {
      "maxiter": 300,
      "popsize": 15,
      "atol": 0.01,
      "seed": 42
    }
  }
}
```

Good for rough global optimization with many parameters.

### Local Minimization

```json
{
  "solver": {
    "method": "scipy.minimize",
    "options": {
      "method": "L-BFGS-B",
      "maxiter": 100
    }
  }
}
```

Fast refinement when starting near optimum.

## Output Format

### Results JSON

Saved to `results/<experiment_id>/results_YYYYMMDD_HHMMSS.json`:

```json
{
  "status": "success",
  "mode": "optimization",
  "parameters": {
    "c_v2_base": 2.35,
    "c_v4_base": 4.89,
    ...
  },
  "predictions": [
    {
      "Z": 82,
      "A": 208,
      "pred_E_MeV": 193728.4,
      "exp_E_MeV": 193729.1,
      "error_pct": -0.00036,
      "virial": 0.12
    }
  ],
  "metrics": {
    "mean_error_pct": -0.42,
    "std_error_pct": 2.1,
    "max_abs_error_pct": 5.3,
    "mean_virial": 0.14,
    "n_converged": 8
  },
  "optimization": {
    "success": true,
    "n_iterations": 47,
    "final_loss": 0.023
  },
  "provenance": {
    "runspec": "experiments/nuclear_heavy_region.runspec.json",
    "experiment_id": "exp_2025_nuclear_heavy_region_v1",
    "timestamp": "2025-12-29T10:35:42.123456",
    "schema_version": "v0.1"
  }
}
```

## Provenance Tracking

Every RunSpec execution records:

- **RunSpec file path**: Full path to configuration
- **Experiment ID**: User-defined identifier
- **Timestamp**: ISO 8601 format
- **Schema version**: RunSpec schema version used
- **Git commit** (if in repo): Code version
- **Parameter values**: Complete parameter set used
- **Solver settings**: Method and options
- **Dataset sources**: Data files and filters

This enables full reproducibility and traceability for publications.

## Validation Against Schema

Before running, validate your RunSpec:

```bash
python validate_runspec.py experiments/my_experiment.runspec.json
```

Common validation errors:

- **Unknown model ID**: Add to `RunSpec.schema.json` allowed models
- **Invalid bounds**: Must be `[lower, upper]` or `null`
- **Missing required fields**: `schema_version`, `model`, `parameters`, `datasets`, `solver`

## Examples

### Example 1: Trial 32 Validation

Test Trial 32 on representative isotopes (evaluation only):

```bash
python src/runspec_adapter.py experiments/test_trial32_eval.runspec.json
```

Expected output:
```
Mode: Evaluation (all parameters frozen)
Target isotopes: 10
mean_error_pct: -0.68
mean_virial: 0.12
```

### Example 2: Heavy Region Optimization

Optimize cohesion/repulsion for A ≥ 120:

```bash
python src/runspec_adapter.py experiments/nuclear_heavy_region.runspec.json
```

Target: Reduce -8.4% systematic error to < -2.0%.

## Performance Tips

### Grid Resolution

The solver uses two modes:

- **Fast mode** (32 grid, 150 iterations): For parameter search
- **Full mode** (48 grid, 360 iterations): For final validation

The adapter uses **full mode** for all evaluations to ensure accurate convergence.

### Virial Constraint

Physical solutions require `|virial| < 0.18`. The objective function automatically penalizes virial violations:

```python
penalty = 4.0 * max(0, |virial| - 0.18)^2
```

### Isotope Selection

For optimization, use 10-20 representative isotopes per region:

- Too few (< 5): Overfitting risk
- Too many (> 50): Slow, diminishing returns

The adapter samples evenly across A-range if not explicitly specified.

## Troubleshooting

### Error: "KeyError: 'c_v2_base'"

**Cause**: Required parameter missing from RunSpec.

**Fix**: Add parameter with Trial 32 default:

```json
{
  "name": "nuclear.c_v2_base",
  "value": 2.201711,
  "role": "coupling",
  "frozen": true
}
```

### Error: "No successful predictions"

**Cause**: Solver failed to converge for all isotopes.

**Possible causes**:
- Parameters outside physical range
- Isotopes too heavy/exotic for current model
- Grid resolution too low

**Fix**: Check virial values in logs, adjust initial parameters, or use simpler isotopes for testing.

### Warning: High virial values

**Example**: `virial = 5.2` (should be < 0.18)

**Cause**: Non-physical field configuration (not converged).

**Fix**:
- Increase `iters_outer` in solver
- Adjust initial parameters closer to Trial 32
- Check for numerical instabilities in logs

## Integration with Grand Solver

The RunSpec adapter is designed for eventual integration with the Grand Solver (multi-domain optimization across nuclear, electronic, cosmological sectors).

### Cross-Sector Parameter Sharing

Future feature: Share vacuum stiffness β across sectors:

```json
{
  "name": "vacuum.beta",
  "value": 3.043233053,
  "role": "coupling",
  "shared_with": ["nuclear", "electronic", "cosmology"]
}
```

### Multi-Objective Optimization

Future feature: Optimize across multiple observables:

```json
{
  "objective": {
    "type": "multi_objective",
    "components": [
      {"observable": "nuclear.binding_energy", "weight": 1.0},
      {"observable": "electronic.fine_structure", "weight": 0.5},
      {"observable": "cosmology.cmb_dipole", "weight": 0.3}
    ]
  }
}
```

## See Also

- **SCHEMA_MIGRATION.md**: Migration guide from ad-hoc scripts
- **RunSpec.schema.json**: Complete schema specification
- **ParameterSpec.schema.json**: Parameter format details
- **qfd_nuclear_soliton_phase9.model.json**: Model specification

## Citation

When using the RunSpec adapter in publications, cite:

```bibtex
@software{qfd_runspec_adapter,
  title = {QFD Nuclear Soliton Solver - RunSpec Adapter},
  year = {2025},
  version = {v0.1},
  url = {https://github.com/qfd/nuclear-soliton-solver}
}
```

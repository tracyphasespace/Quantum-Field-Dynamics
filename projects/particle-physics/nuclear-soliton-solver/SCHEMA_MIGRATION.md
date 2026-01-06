# Schema Migration Plan

**Date**: December 29, 2025
**Status**: In Progress
**Goal**: Migrate nuclear soliton solver to QFD RunSpec v0 schema

---

## Current State

The soliton solver uses ad-hoc Python scripts:
- `qfd_solver.py` - hardcoded CLI arguments
- `qfd_metaopt_ame2020.py` - parameters as dict
- `qfd_regional_calibration.py` - parameters as dict

**Issues**:
- No schema validation
- No provenance tracking
- Parameters not standardized
- Not compatible with Grand Solver architecture

---

## Target State (RunSpec v0 Compliant)

### Model Specification

```json
{
  "model_id": "qfd.nuclear.binding.soliton",
  "variant": "phase9_scf",
  "equations": [
    {"term": "kinetic_nucleon", "implementation": "qfd.solvers.nuclear.terms.kinetic_3d"},
    {"term": "kinetic_electron", "implementation": "qfd.solvers.nuclear.terms.kinetic_3d"},
    {"term": "V2_cohesion", "implementation": "qfd.solvers.nuclear.terms.phi2_attractive"},
    {"term": "V4_repulsion", "implementation": "qfd.solvers.nuclear.terms.phi4_repulsive"},
    {"term": "V6_saturation", "implementation": "qfd.solvers.nuclear.terms.phi6_saturation"},
    {"term": "coulomb_cross", "implementation": "qfd.solvers.nuclear.terms.coulomb_spectral"},
    {"term": "symmetry_energy", "implementation": "qfd.solvers.nuclear.terms.charge_asymmetry"},
    {"term": "surface_tension", "implementation": "qfd.solvers.nuclear.terms.gradient_squared"},
    {"term": "rotor", "implementation": "qfd.solvers.nuclear.terms.angular_momentum"}
  ],
  "grid": {
    "dimension": 3,
    "points": 48,
    "extent": {"min": -20.0, "max": 20.0, "units": "fm"},
    "boundary": "periodic"
  },
  "backend": {
    "framework": "torch",
    "device": "cuda",
    "precision": "float32",
    "deterministic": true
  }
}
```

### Parameter Specifications

**Trial 32 Parameters** (mapped to ParameterSpec):

```json
{
  "parameters": [
    {
      "name": "nuclear.c_v2_base",
      "value": 2.201711,
      "role": "coupling",
      "bounds": [0.0, 10.0],
      "units": "MeV",
      "description": "Baseline cohesion strength (φ² attractive term)"
    },
    {
      "name": "nuclear.c_v2_iso",
      "value": 0.027035,
      "role": "coupling",
      "bounds": [0.0, 1.0],
      "units": "MeV",
      "description": "Isospin-dependent cohesion correction"
    },
    {
      "name": "nuclear.c_v2_mass",
      "value": -0.000205,
      "role": "coupling",
      "bounds": [-0.01, 0.01],
      "units": "1/A",
      "description": "Mass-dependent compounding (≈0 for Trial 32)"
    },
    {
      "name": "nuclear.c_v4_base",
      "value": 5.282364,
      "role": "coupling",
      "bounds": [0.0, 20.0],
      "units": "MeV",
      "description": "Baseline quartic repulsion (φ⁴ term)"
    },
    {
      "name": "nuclear.c_v4_size",
      "value": -0.085018,
      "role": "coupling",
      "bounds": [-1.0, 1.0],
      "units": "MeV",
      "description": "Size-dependent quartic correction"
    },
    {
      "name": "nuclear.alpha_e_scale",
      "value": 1.007419,
      "role": "coupling",
      "bounds": [0.0, 5.0],
      "units": "dimensionless",
      "description": "Electron cohesion scale factor"
    },
    {
      "name": "nuclear.beta_e_scale",
      "value": 0.504312,
      "role": "coupling",
      "bounds": [0.0, 5.0],
      "units": "dimensionless",
      "description": "Electron quartic scale factor"
    },
    {
      "name": "nuclear.c_sym",
      "value": 25.0,
      "role": "coupling",
      "bounds": [0.0, 100.0],
      "units": "MeV",
      "description": "QFD charge asymmetry coefficient (not SEMF!)"
    },
    {
      "name": "nuclear.kappa_rho",
      "value": 0.029816,
      "role": "coupling",
      "bounds": [0.0, 1.0],
      "units": "dimensionless",
      "description": "Density-dependent coupling (φ⁶ term)"
    }
  ]
}
```

### Regional RunSpec Files

**Three separate RunSpec files**:

1. **`experiments/nuclear_light_region.runspec.json`**
   - A range: 1-59
   - Parameters: Trial 32 (frozen, validation only)
   - Dataset: AME2020 light isotopes + magic numbers

2. **`experiments/nuclear_medium_region.runspec.json`**
   - A range: 60-119
   - Parameters: Trial 32 baseline, fine-tune bounds ±10%
   - Dataset: AME2020 medium isotopes

3. **`experiments/nuclear_heavy_region.runspec.json`**
   - A range: 120-250
   - Parameters: Trial 32 baseline, wide bounds for cohesion (+15%)
   - Dataset: AME2020 heavy isotopes (Pb-208, Au-197, U-238)

---

## Migration Steps

### Phase 1: Create Model Specification ✓ (in progress)

- [x] Review QFD schema structure
- [ ] Create `models/qfd_nuclear_soliton_phase9.model.json`
- [ ] Define all equation terms with implementations
- [ ] Specify grid and backend configuration

### Phase 2: Convert Parameters

- [ ] Create `parameters/trial32_universal.params.json`
- [ ] Create `parameters/heavy_region_bounds.params.json`
- [ ] Validate against ParameterSpec.schema.json

### Phase 3: Create RunSpec Files

- [ ] `experiments/nuclear_heavy_region.runspec.json` (priority)
- [ ] `experiments/nuclear_medium_region.runspec.json`
- [ ] `experiments/nuclear_light_validation.runspec.json`
- [ ] Validate against RunSpec.schema.json

### Phase 4: Update Solver

- [ ] Create `qfd_solver_runspec.py` (RunSpec-compatible wrapper)
- [ ] Adapt `qfd_solver.py` to consume RunSpec
- [ ] Add provenance tracking (git commit, timestamps)
- [ ] Test with example RunSpec

### Phase 5: Integrate with Schema Validator

- [ ] Use `/home/tracy/development/QFD_SpectralGap/schema/v0/validate_runspec.py`
- [ ] Run validation on all three RunSpec files
- [ ] Fix any schema violations

### Phase 6: Results Schema

- [ ] Define ResultSpec for regional calibration
- [ ] Include: best parameters, loss, convergence metrics, isotope-by-isotope errors
- [ ] Link back to RunSpec provenance

---

## Benefits of Schema Compliance

### Immediate

1. **Validation**: Catch errors before expensive computation
2. **Documentation**: Self-documenting JSON files
3. **Provenance**: Git commit + RunSpec = full reproducibility

### Integration

4. **Grand Solver compatibility**: Can be called from unified parameter optimization
5. **Cross-domain constraints**: Share parameters (e.g., V4, g_c) with lepton sector
6. **Observable adapters**: Standard interface for prediction functions

### Publication

7. **Reproducible research**: Archive RunSpec + git tag in supplementary material
8. **Peer review**: Reviewers can reproduce exact fits
9. **Comparison**: A/B test different parameter sets with identical setup

---

## Schema File Structure

```
nuclear-soliton-solver/
├── models/
│   └── qfd_nuclear_soliton_phase9.model.json   # ModelSpec
├── parameters/
│   ├── trial32_universal.params.json           # Trial 32 baseline
│   ├── heavy_region_bounds.params.json         # Heavy region search space
│   └── medium_region_bounds.params.json        # Medium region search space
├── experiments/
│   ├── nuclear_heavy_region.runspec.json       # Full RunSpec for heavy
│   ├── nuclear_medium_region.runspec.json      # Full RunSpec for medium
│   └── nuclear_light_validation.runspec.json   # Validation-only RunSpec
├── results/
│   ├── heavy_region_fit_001.result.json        # ResultSpec output
│   └── ...
└── src/
    ├── qfd_solver_runspec.py                   # RunSpec-compatible solver
    └── ...
```

---

## Example: Heavy Region RunSpec (Skeleton)

```json
{
  "schema_version": "v0.1",
  "experiment_id": "exp_2025_nuclear_heavy_region_v1",
  "description": "Regional calibration for heavy isotopes (A ≥ 120). Target: reduce -8.4% systematic underbinding to -2%.",

  "model": {
    "$ref": "models/qfd_nuclear_soliton_phase9.model.json"
  },

  "parameters": [
    {
      "name": "nuclear.c_v2_base",
      "value": 2.201711,
      "role": "coupling",
      "bounds": [2.31, 2.64],
      "units": "MeV",
      "description": "Increased cohesion for heavy nuclei (+5% to +20%)"
    },
    {
      "name": "nuclear.c_v4_base",
      "value": 5.282364,
      "role": "coupling",
      "bounds": [4.49, 5.28],
      "units": "MeV",
      "description": "Decreased repulsion for heavy nuclei (-15% to 0%)"
    },
    // ... other parameters with tighter bounds around Trial 32
  ],

  "datasets": [
    {
      "id": "ame2020_heavy",
      "source": "data/ame2020_system_energies.csv",
      "columns": {
        "A": "A",
        "Z": "Z",
        "target": "E_exp_MeV",
        "sigma": "E_uncertainty_MeV"
      },
      "cuts": {
        "A_min": 120,
        "A_max": 250,
        "Z_list": [50, 79, 80, 82, 92]
      }
    }
  ],

  "objective": {
    "type": "chi_squared",
    "components": [
      {
        "dataset_id": "ame2020_heavy",
        "observable_adapter": "qfd.adapters.nuclear.predict_total_energy",
        "weight": 1.0
      }
    ],
    "regularization": [
      {
        "type": "virial_hinge",
        "threshold": 0.18,
        "weight": 4.0
      }
    ]
  },

  "solver": {
    "method": "scipy.differential_evolution",
    "options": {
      "maxiter": 20,
      "popsize": 8,
      "seed": 42,
      "atol": 0.01,
      "tol": 0.01
    }
  },

  "provenance": {
    "git": {
      "repo": "QFD_SpectralGap",
      "commit": "<to be filled>",
      "dirty": false
    },
    "created": "2025-12-29T00:00:00Z",
    "author": "Tracy McPherson"
  }
}
```

---

## Compatibility Notes

### Backward Compatibility

The old scripts (`qfd_regional_calibration.py`) will remain for quick prototyping, but production runs should use RunSpec.

### Forward Compatibility

Schema v0 → v0.2 will add:
- DependencyGraph for parameter constraints
- SensitivitySpec for uncertainty propagation
- ComparisonSpec for multi-run A/B tests

Design RunSpec files to be forward-compatible by using standard field names.

---

## Next Actions

1. **Create model spec**: `models/qfd_nuclear_soliton_phase9.model.json`
2. **Create parameter files**: Trial 32 baseline + regional bounds
3. **Create heavy region RunSpec**: Priority target for fixing -8% errors
4. **Test validation**: Use schema validator to catch errors
5. **Run optimization**: Execute heavy region fit with RunSpec
6. **Document results**: Save ResultSpec with provenance

---

**Status**: Ready to begin Phase 1 (Model Specification)

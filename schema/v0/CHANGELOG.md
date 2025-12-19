# QFD RunSpec Schema Changelog

## v0.2 - Hardened Implementation (2025-12-19)

### Critical Fixes

**Objective Function**:
- ✅ Implemented true chi-squared: χ² = Σ((y - ŷ)/σ)² when σ is provided
- ✅ Falls back to SSE when σ is absent (treats as σ=1)
- ✅ Explicit error for unsupported objective types (no silent failures)
- ✅ Added `sse` as explicit objective type in schema

**Dataset Cuts**:
- ✅ Strict validation: Fails loudly if `stable_only=true` but no stability column exists
- ✅ Strict validation: Fails loudly if `mass_min` specified but no A column exists
- ✅ No more silent ignoring of requested cuts

**Parameter Validation**:
- ✅ Explicit check: Raises KeyError if required parameters (c1, c2) are missing
- ✅ Clear error messages indicating which parameters are missing

**Schema Strictness**:
- ✅ Added `$id` to all schemas for robust reference resolution
- ✅ Made `prior.type` required when `prior` is specified
- ✅ Type-constrained `objective.weights` (must be numbers)
- ✅ Added `columns.stable` to schema for explicit stability column mapping
- ✅ All schemas now use `additionalProperties: false` (prevents typos)

**Provenance Tracking**:
- ✅ SHA256 hash of each dataset file
- ✅ Row counts: raw vs. final (after cuts)
- ✅ Dataset provenance in `results_summary.json`
- ✅ Algorithm tracking: Records which scipy method was used

**Solver Configuration**:
- ✅ Respects `solver.options.algo` (e.g., "L-BFGS-B", "BFGS", "Nelder-Mead")
- ✅ No more hard-coded algorithm selection

### New Features

**LoadedDataset Dataclass**:
- Tracks dataset spec, dataframe, file hash, row counts
- Enables comprehensive provenance

**Artifact Generation**:
- `predictions.csv` - A, y_obs, y_pred, residual for every dataset
- `runspec_resolved.json` - Exact configuration as-run
- `results_summary.json` - Best-fit params, loss, provenance

**Error Messages**:
- All errors now include context (dataset ID, column names, parameter names)
- Clear instructions for fixing issues (e.g., "Specify datasets[*].columns.stable")

### Schema Changes

**ParameterSpec.schema.json**:
```diff
+ "$id": "ParameterSpec.schema.json"
+ "prior.required": ["type"]
+ "prior.additionalProperties": false
```

**RunSpec.schema.json**:
```diff
+ "$id": "RunSpec.schema.json"
+ "objective.type.enum": ["chi_squared", "sse"]
+ "objective.weights.additionalProperties": {"type": "number"}
+ "datasets[*].columns.stable": {"type": "string"}
+ "datasets[*].additionalProperties": false
```

### Breaking Changes

None - v0.2 is backward compatible with v0.1 RunSpecs, but:
- Will now fail on invalid configurations that previously failed silently
- Requires explicit column mapping if using cuts (recommended, not breaking)

### Migration Guide

If you have v0.1 RunSpecs:

1. **Add explicit column mapping** (recommended):
```json
"columns": {
  "A": "mass_number",
  "target": "binding_energy_mev",
  "stable": "is_stable"
}
```

2. **Specify algorithm** if you want non-default:
```json
"solver": {
  "method": "scipy.minimize",
  "options": {
    "algo": "L-BFGS-B",  // Add this
    "maxiter": 1000,
    "tol": 1e-6
  }
}
```

3. **Add sigma column** for true chi-squared (optional):
```json
"columns": {
  "A": "mass_number",
  "target": "binding_energy_mev",
  "sigma": "binding_energy_uncertainty_mev"  // Add this
}
```

### Testing

All v0.2 improvements tested with `experiments/ccl_fit_v1.json`:
- ✅ Schema validation passes
- ✅ Strict cut handling works
- ✅ True chi-squared implemented
- ✅ Provenance tracking complete
- ✅ Artifacts generated correctly

## v0.1 - Initial Release (2025-12-19)

### Features

- JSON Schema-based configuration system
- Parameter hydration with bounds and priors
- Dataset loading with basic cuts
- SSE objective function
- scipy.optimize.minimize integration
- Basic provenance (git commit)
- Artifact generation (predictions.csv, results.json)

### Known Issues (Fixed in v0.2)

- Objective always computed SSE regardless of schema
- Cuts silently ignored if columns missing
- No dataset file hashing
- Hard-coded optimization algorithm
- Missing row count tracking

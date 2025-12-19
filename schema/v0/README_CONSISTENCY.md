# Lean4 ↔ JSON Consistency Checker

## Quick Start

```bash
# Check a single config
python check_lean_json_consistency.py experiments/ccl_fit_v1.json

# Check all configs
python check_lean_json_consistency.py --all

# Strict mode (bounds mismatches are errors)
python check_lean_json_consistency.py experiments/ccl_fit_v1.json --strict
```

## What It Checks

### 1. Parameter Existence
- ✅ All JSON `coupling` parameters exist in Lean4 schema
- ℹ️  Nuisance/calibration parameters can be JSON-only (dataset-specific)

### 2. Bounds Consistency
- ✅ JSON bounds contain Lean4 bounds: `[json_lo, json_hi] ⊇ (lean_lo, lean_hi)`
- ⚠️  Warning if JSON bounds don't contain Lean bounds
- ❌ Error in `--strict` mode

**Example**:
```
Lean: 0.5 < c1 < 1.5  → (0.5, 1.5)
JSON: [0.5, 1.5]      → ✅ Contains Lean bounds
JSON: [0.1, 2.0]      → ⚠️  Looser (exploration range)
JSON: [0.6, 1.4]      → ❌ Doesn't contain Lean bounds
```

### 3. Units Consistency
- ✅ JSON units match Lean4 type
- ℹ️  Info if "unitless" vs "dimensionless" (recommend "dimensionless")
- ⚠️  Warning if units don't match expected type

**Mappings**:
| Lean4 Type | Expected JSON Units |
|------------|---------------------|
| `Unitless` | dimensionless, unitless |
| `Energy` | eV, MeV, GeV, J |
| `Mass` | kg, eV, MeV |
| `Length` | m, fm, km |
| `Velocity` | m/s, km/s, km/s/Mpc |

### 4. Coverage
- ℹ️  Info if JSON doesn't include all Lean parameters
- This is expected for domain-specific configs

## Exit Codes

- `0`: Success (warnings/info allowed)
- `1`: Failure (errors found)
- `2`: Usage error

## Example Output

```bash
$ python check_lean_json_consistency.py experiments/ccl_fit_v1.json

============================================================
Checking: ccl_fit_v1.json
============================================================

ℹ️  INFO (2):

  [json_only_nuisance] JSON parameter 'calibration.offset' is nuisance/calibration only (not in Lean)
    JSON:  role=nuisance, units=

  [incomplete_json] JSON config does not include all Lean parameters
    Lean4: Missing: V4, k_c2, alpha_n, beta_n, gamma_e, ...

============================================================
Summary: 0 errors, 0 warnings, 2 info

✅ PASSED: All checks passed
```

## Design Philosophy

### Lean4: Physics Correctness
- Type-safe dimensions (can't add Length + Energy)
- Proven bounds from physical constraints
- Theorems about parameter relationships

### JSON: Practical Exploration
- Looser bounds for optimization
- Runtime validation
- Easy to modify for experiments

### Consistency Checker: Bridge
- Ensures JSON respects Lean4 physics
- Allows JSON flexibility where appropriate
- Automated checking prevents drift

## Integration with CI

Add to `.github/workflows/ci.yml`:

```yaml
- name: Build Lean4 schema
  run: |
    cd projects/Lean4
    lake build QFD.Schema.DimensionalAnalysis \
               QFD.Schema.Couplings \
               QFD.Schema.Constraints

- name: Check schema consistency
  run: |
    cd schema/v0
    python check_lean_json_consistency.py --all
```

## Common Issues

### ❌ Bounds Mismatch
```
Parameter 'nuclear.c2': JSON bounds do not contain Lean bounds
  Lean: (0.0, 0.1)
  JSON: [0.1, 1.0]
```

**Fix**: Adjust JSON bounds to contain Lean bounds:
```json
{
  "name": "nuclear.c2",
  "bounds": [0.0, 0.1]
}
```

### ℹ️  Units Naming
```
Parameter 'nuclear.c1': Units naming inconsistency
  Lean: Unitless
  JSON: 'unitless' (recommend 'dimensionless')
```

**Fix**: Use "dimensionless" consistently:
```json
{
  "name": "nuclear.c1",
  "units": "dimensionless"
}
```

### ℹ️  Incomplete JSON
```
JSON config does not include all Lean parameters
  Missing: V4, g_c, k_J, ...
```

**This is OK**: Domain-specific configs only use relevant parameters.
Nuclear fit doesn't need cosmology parameters (k_J, eta_prime).

## Future Enhancements

- [ ] Parse dimensional annotations from Lean (⟨1,0,-1,0⟩ → "m/s")
- [ ] Generate JSON Schema from Lean4 definitions
- [ ] Round-trip validation (JSON → Lean parse → validate)
- [ ] Constraint translation (Lean Prop → JSON validator)
- [ ] Fisher information consistency

## References

- Lean4 Schema: `projects/Lean4/QFD/Schema/`
- JSON Schema: `schema/v0/*.schema.json`
- Consistency Analysis: `schema/v0/LEAN_JSON_CONSISTENCY.md`

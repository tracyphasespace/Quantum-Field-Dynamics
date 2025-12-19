# Lean4 ↔ JSON Schema Consistency Analysis

**Date**: 2025-12-19
**Lean4 Build**: ✅ All modules compile (749/749 jobs)
**JSON Schema**: ✅ v0.2 validated

---

## Overview

We have **two parallel schema systems**:

1. **Lean4 Schema** (`projects/Lean4/QFD/Schema/`) - Type-safe, proof-carrying definitions
2. **JSON Schema** (`schema/v0/`) - Runtime validation for Python solver

These must remain consistent to ensure:
- Lean4 proofs about parameters apply to JSON configs
- JSON configs satisfy Lean4 constraints
- Parameter bounds match across systems

---

## Build Status

### Lean4 Modules ✅

```bash
$ lake build QFD.Schema.DimensionalAnalysis \
             QFD.Schema.Couplings \
             QFD.Schema.Constraints

⚠ [749/749] Replayed QFD.Schema.Constraints
Build completed successfully (749 jobs).
```

**Fixed Issues**:
1. ✅ Removed `deriving Repr` from `Quantity` (Real.instRepr is unsafe)
2. ✅ Removed invalid Mul/Div instances (dimensions are heterogeneous)
3. ✅ Fixed `.abs` notation → `|x|` (abs is a function, not a field)

**Remaining Warnings** (non-blocking):
- Style linter: Extra spaces in source (cosmetic)
- 3 intentional `sorry`s in Constraints.lean (placeholders for future proofs)

---

## Parameter Consistency Check

### Nuclear Parameters

**Lean4** (`QFD/Schema/Couplings.lean`):
```lean
structure NuclearParams where
  c1       : Unitless      -- Surface term
  c2       : Unitless      -- Volume term
  V4       : Energy        -- Potential depth
  k_c2     : Mass          -- Mass scale for binding
  alpha_n  : Unitless      -- Nuclear fine structure
  beta_n   : Unitless      -- Asymmetry coupling
  gamma_e  : Unitless      -- Geometric shielding
```

**JSON** (`experiments/ccl_fit_v1.json`):
```json
{
  "name": "nuclear.c1",
  "value": 0.5,
  "role": "coupling",
  "bounds": [0.1, 2.0],
  "units": "unitless"
}
```

**Consistency**:
- ✅ Parameter names match: `c1`, `c2`, `V4`, etc.
- ✅ Units match: Lean `Unitless` ↔ JSON `"unitless"`
- ✅ Roles match: Lean schema comment "coupling" ↔ JSON `"role": "coupling"`
- ⚠️ **Bounds not enforced**: Lean has constraints, JSON has bounds, but no automated check

### Cosmology Parameters

**Lean4**:
```lean
structure CosmoParams where
  k_J       : Quantity ⟨1, 0, -1, 0⟩  -- km/s/Mpc
  eta_prime : Unitless
  A_plasma  : Unitless
  rho_vac   : Density
  w_dark    : Unitless
```

**JSON** (not yet in v0 solver, but in schema):
```json
{
  "name": "cosmo.k_J",
  "units": "km/s/Mpc",
  "role": "coupling"
}
```

**Consistency**:
- ✅ Parameter names match
- ✅ Units match: Lean `⟨1, 0, -1, 0⟩` = L¹T⁻¹ ↔ JSON `"km/s/Mpc"`
- ⚠️ **Not yet in Python solver**: Cosmology parameters defined but not implemented

### Particle Parameters

**Lean4**:
```lean
structure ParticleParams where
  g_c       : Unitless      -- Geometric charge (0 ≤ g_c ≤ 1)
  V2        : Energy
  lambda_R  : Unitless
  mu_e      : Mass          -- Electron mass seed
  mu_nu     : Mass          -- Neutrino mass seed
```

**JSON** (in core_compression_fit.runspec.json):
```json
{
  "name": "g_c",
  "value": 0.985,
  "bounds": [0.9, 1.0],
  "units": "dimensionless"
}
```

**Consistency**:
- ✅ Parameter names match
- ⚠️ **Units naming**: Lean uses `Unitless`, JSON uses `"dimensionless"` or `"unitless"`
  - Should standardize on one term
- ✅ g_c constraint matches: Lean `0 ≤ g_c ≤ 1` ↔ JSON `[0.9, 1.0]` (stricter)

---

## Constraint Consistency

### Nuclear Constraints

**Lean4** (`QFD/Schema/Constraints.lean`):
```lean
structure NuclearConstraints (p : NuclearParams) : Prop where
  c1_range : 0.5 < p.c1.val ∧ p.c1.val < 1.5
  c2_range : 0.0 < p.c2.val ∧ p.c2.val < 0.1
  V4_range : 1e6 < p.V4.val ∧ p.V4.val < 1e9  -- eV scale

  genesis_compatible :
    |p.alpha_n.val - 3.5| < 1.0 ∧
    |p.beta_n.val - 3.9| < 1.0 ∧
    |p.gamma_e.val - 5.5| < 2.0
```

**JSON** (`experiments/ccl_fit_v1.json`):
```json
{
  "name": "nuclear.c1",
  "bounds": [0.1, 2.0]
}
```

**Consistency Issues**:
- ⚠️ **Bounds mismatch**: Lean `(0.5, 1.5)` vs JSON `[0.1, 2.0]`
  - JSON bounds are looser (exploration range)
  - Lean bounds are tighter (proven physical range)
  - This is acceptable: JSON allows exploration, Lean enforces physics

- ✅ **genesis_compatible**: Not in JSON (derived constraint, checked post-fit)

### Cosmology Constraints

**Lean4**:
```lean
structure CosmoConstraints (p : CosmoParams) : Prop where
  k_J_range : 50.0 < p.k_J.val ∧ p.k_J.val < 100.0  -- km/s/Mpc
  eta_prime_range : 0.0 ≤ p.eta_prime.val ∧ p.eta_prime.val < 0.1
  w_dark_range : -2.0 < p.w_dark.val ∧ p.w_dark.val < 0.0
```

**JSON** (from original v0 example, not in ccl_fit_v1):
```json
{
  "name": "k_J",
  "bounds": [50.0, 100.0]
}
```

**Consistency**:
- ✅ **k_J bounds match exactly**: Lean `(50, 100)` ↔ JSON `[50, 100]`

---

## Dimensional Analysis Comparison

### Lean4 Type Safety

Lean4 enforces dimensions at **compile time**:
```lean
def Length := Quantity ⟨1, 0, 0, 0⟩
def Energy := Quantity ⟨2, 1, -2, 0⟩
def Velocity := Quantity ⟨1, 0, -1, 0⟩

def Quantity.add {d : Dimensions} (a b : Quantity d) : Quantity d :=
  ⟨a.val + b.val⟩  -- Can only add quantities with same dimensions!
```

**Prevented errors**:
```lean
def bad := Length.add (⟨5⟩ : Length) (⟨10⟩ : Energy)
-- ERROR: Type mismatch - cannot add Length and Energy
```

### JSON Runtime Validation

JSON Schema checks units at **runtime**:
```json
{
  "name": "k_J",
  "units": "km/s/Mpc"
}
```

**Limitations**:
- ❌ No dimensional consistency check across parameters
- ❌ Cannot prevent `c1 + V4` (unitless + energy) in Python code
- ✅ Can validate units match expected string

### Gap Analysis

**Problem**: Lean4 prevents dimensional errors at compile time, JSON only documents units.

**Solutions**:

1. **Add dimensional validation to Python solver**:
```python
from dataclasses import dataclass

@dataclass
class Dimensions:
    length: int
    mass: int
    time: int
    charge: int

@dataclass
class Quantity:
    value: float
    dims: Dimensions

    def __add__(self, other):
        if self.dims != other.dims:
            raise DimensionalError(f"Cannot add {self.dims} and {other.dims}")
        return Quantity(self.value + other.value, self.dims)
```

2. **Export Lean4 constraints to JSON**:
```lean
-- Generate JSON Schema bounds from Lean4 constraints
def export_constraint (name : String) (lo hi : Float) : Json := ...
```

3. **Bidirectional validation**:
   - Parse JSON config in Lean4, verify constraints
   - Generate JSON Schema from Lean4 definitions

---

## Units Naming Standardization

**Inconsistencies**:
| Lean4 | JSON (ccl_fit_v1) | JSON (core_compression) | Recommendation |
|-------|-------------------|-------------------------|----------------|
| `Unitless` | `"unitless"` | `"dimensionless"` | **Use `"dimensionless"`** |
| `Energy` | `"eV"` | `"eV"` | ✅ Consistent |
| `Velocity ⟨1,0,-1,0⟩` | `"km/s/Mpc"` | - | ✅ Correct |

**Action**: Update JSON schemas to use `"dimensionless"` consistently.

---

## Missing Cross-Checks

### What's NOT automated:

1. ❌ **Bounds consistency**: Lean constraints vs JSON bounds
   - Lean: `0.5 < c1 < 1.5`
   - JSON: `[0.1, 2.0]`
   - No check that JSON bounds contain Lean bounds

2. ❌ **Parameter completeness**: All Lean params have JSON equivalents
   - Easy to add param to Lean, forget JSON
   - No automated check

3. ❌ **Units parsing**: JSON `"km/s/Mpc"` → Lean `⟨1, 0, -1, 0⟩`
   - Manual mapping, error-prone

4. ❌ **Constraint equivalence**: genesis_compatible in Lean but not JSON
   - Some constraints are proof-level (Lean only)
   - Others should be runtime-checked (JSON)

### Recommended Automation

**Priority 1: Bounds Checker**
```python
# schema/v0/check_lean_json_consistency.py
def check_bounds_consistency(lean_constraints, json_config):
    for param in lean_constraints:
        lean_lo, lean_hi = param.bounds
        json_lo, json_hi = json_config[param.name]["bounds"]
        assert json_lo <= lean_lo < lean_hi <= json_hi, \
            f"{param.name}: JSON bounds must contain Lean bounds"
```

**Priority 2: Parameter Inventory**
```bash
# Extract all Lean params
grep "^  [a-z_].*:" QFD/Schema/Couplings.lean

# Extract all JSON params
jq '.parameters[].name' experiments/ccl_fit_v1.json

# Diff and report missing
```

**Priority 3: Units Parser**
```lean
-- Convert Lean Dimensions to JSON string
def dimensions_to_string : Dimensions → String
  | ⟨1, 0, 0, 0⟩ => "m"
  | ⟨0, 1, 0, 0⟩ => "kg"
  | ⟨1, 0, -1, 0⟩ => "m/s"
  | ⟨2, 1, -2, 0⟩ => "J" -- or "eV" for particle physics
  | _ => "(custom)"
```

---

## Recommended Workflow

### When Adding a New Parameter

1. **Define in Lean4**:
```lean
-- QFD/Schema/Couplings.lean
structure NewDomainParams where
  new_param : Energy  -- Add with type
```

2. **Add constraint in Lean4**:
```lean
-- QFD/Schema/Constraints.lean
structure NewDomainConstraints (p : NewDomainParams) : Prop where
  new_param_range : 1.0 < p.new_param.val ∧ p.new_param.val < 100.0
```

3. **Export to JSON**:
```json
{
  "name": "new_domain.new_param",
  "value": 10.0,
  "role": "coupling",
  "bounds": [0.1, 200.0],
  "units": "eV"
}
```

4. **Verify consistency**:
```bash
# Check Lean bounds [1, 100] ⊂ JSON bounds [0.1, 200]
python schema/v0/check_lean_json_consistency.py
```

5. **Build Lean4**:
```bash
lake build QFD.Schema.NewDomain
```

6. **Validate JSON**:
```bash
python validate_runspec.py experiments/new_domain_fit.json
```

---

## Current Status

### ✅ What Works

- Lean4 schema builds successfully (749 jobs)
- JSON schema validates correctly
- Parameter names consistent across systems
- Units mostly consistent (modulo unitless/dimensionless)
- Example configs validate

### ⚠️ What's Manual

- Bounds consistency checking (Lean ↔ JSON)
- Parameter inventory (all Lean params in JSON?)
- Units naming (unitless vs dimensionless)
- Constraint translation (Lean Prop → JSON validation)

### ❌ What's Missing

- Automated consistency checker
- Dimensional validation in Python solver
- Units parser (JSON string → Lean Dimensions)
- Round-trip validation (JSON → Lean → JSON)

---

## Next Steps

### Immediate (Week 1)

1. ✅ Build Lean4 schema (DONE)
2. ⏳ Standardize units naming: `"dimensionless"` everywhere
3. ⏳ Create `check_lean_json_consistency.py`
4. ⏳ Run consistency check on ccl_fit_v1.json

### Short-term (Weeks 2-3)

5. Add dimensional validation to Python solver
6. Export Lean4 constraints to JSON automatically
7. Test round-trip: JSON → Lean parse → validate

### Medium-term (Month 1-2)

8. Bidirectional schema generation
9. Automated testing: every JSON config must pass Lean4 constraints
10. CI check: Lean4 build + JSON validation on every commit

---

## Conclusion

The Lean4 and JSON schemas are **manually consistent** for current parameters (c1, c2, g_c, V4). However:

- **No automated checking** - consistency is manual effort
- **Bounds mismatch** - Lean tighter, JSON looser (acceptable for exploration)
- **Units naming** - needs standardization
- **Dimensional safety** - only in Lean4, not in Python solver

**Recommendation**: Implement `check_lean_json_consistency.py` as the minimum viable automation to prevent drift as the schema grows from ~5 parameters to 15-30.

**Status**: ✅ Lean4 builds, ⚠️ Manual consistency, ❌ No automation

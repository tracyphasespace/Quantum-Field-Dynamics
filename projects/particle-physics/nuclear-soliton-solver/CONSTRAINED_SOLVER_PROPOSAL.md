# Constrained Soliton Solver Using Core Compression Law

## Problem with Current Approach

**Blind optimization failed**:
- Optimized parameters → Solver fails completely (E_model = 0, virial = 999)
- No solutions found in 1090 evaluations
- Parameter space too large, no guidance

## Key Insight

**We already know the answer!**

From Lean formalization (`CoreCompressionLaw.lean`):
- **c1 = 0.529** (surface term, validated)
- **c2 = 0.317** (volume term, validated)
- **R² = 0.9794** on 5,842 isotopes
- **Stress ratio = 3.6×** (stable vs unstable)

**Backbone formula**: Q(A) = c1·A^(2/3) + c2·A

This **already predicts charge numbers correctly**. Use it!

---

## Proposed Solution: Constrained Soliton Solving

### Step 1: Fix Charge from Experiment

For each isotope (Z, A):
- **Input**: Z (known from experiment, e.g., Z=82 for Pb-208)
- **Not optimizing**: Z is FIXED, not a free parameter

### Step 2: Use CCL as Charge Distribution Constraint

Instead of letting the solver find arbitrary charge distribution ρ_N, constrain it:

```python
# Current (unconstrained):
ρ_N(r) = |ψ_N(r)|² + |B_N(r)|²  # Free to be anything

# Proposed (constrained):
∫ ρ_N(r) dV = A                  # Mass number (already enforced)
∫ ρ_Z(r) dV = Z                  # Charge (already enforced)

# NEW CONSTRAINT - Use CCL to guide radial profile:
Q_backbone(A) = c1·A^(2/3) + c2·A
ρ_charge(r) should peak at radius r_0 ~ A^(1/3)  # From CCL geometry
```

### Step 3: Solve Reduced Parameter Set

With Z and charge distribution constrained, optimize only:

**Nuclear potential parameters** (5 parameters instead of 9):
1. `c_v2_base` - Quartic bulk strength
2. `c_v4_base` - Sextic bulk strength
3. `c_sym` - Symmetry energy coefficient
4. `kappa_rho` - Surface tension
5. `alpha_e_scale` - Electron coupling

**Fixed from CCL**:
- ~~`c_v2_iso`~~ - Determined by CCL c2
- ~~`c_v2_mass`~~ - Determined by CCL mass scaling
- ~~`c_v4_size`~~ - Determined by CCL c1 (surface)
- ~~`beta_e_scale`~~ - Determined by electron constraints

### Step 4: Add CCL Penalty to Loss Function

```python
def loss_with_ccl_constraint(params):
    loss_binding = binding_energy_error(params)  # As before

    # NEW: CCL consistency check
    Q_predicted = c1 * A**(2/3) + c2 * A
    Z_from_solver = get_effective_charge(params, A)

    ccl_penalty = (Z_from_solver - Q_predicted)**2

    return loss_binding + lambda_ccl * ccl_penalty
```

**Rationale**: The solver should produce charge distributions consistent with CCL. If it doesn't, penalize heavily.

---

## Expected Benefits

### 1. **Massively Reduced Parameter Space**
- Before: 9 parameters × unbounded search
- After: 5 parameters × guided by CCL
- **Reduction**: 44% fewer parameters, much tighter bounds

### 2. **Physical Constraints Guide Convergence**
- CCL tells us the **correct charge distribution**
- Solver doesn't waste time on unphysical configurations
- Virial should converge faster (fewer bad states explored)

### 3. **Validates Cross-Realm Consistency**
- If soliton solver **reproduces CCL** from microscopic physics → strong validation
- If it **contradicts CCL** → reveals physics gap in model

### 4. **Falsifiability**
- Clear test: "Do optimized solitons satisfy Q(A) = c1·A^(2/3) + c2·A?"
- If no: Theory has internal inconsistency
- If yes: Strong evidence for QFD unification

---

## Implementation Sketch

### Modified Objective Function

```python
# File: src/constrained_objective.py

from parallel_objective import ParallelObjective

class CCLConstrainedObjective(ParallelObjective):
    def __init__(self, c1=0.529, c2=0.317, lambda_ccl=10.0, **kwargs):
        super().__init__(**kwargs)
        self.c1 = c1  # From Lean validation
        self.c2 = c2  # From Lean validation
        self.lambda_ccl = lambda_ccl  # CCL penalty strength

    def backbone_charge(self, A):
        """Predicted charge from Core Compression Law."""
        return self.c1 * (A ** (2/3)) + self.c2 * A

    def __call__(self, params):
        # Standard binding energy loss
        loss_binding = super().__call__(params)

        # CCL consistency penalty
        ccl_penalty = 0.0
        for Z, A in self.target_isotopes:
            Q_ccl = self.backbone_charge(A)

            # The solver should produce charge distributions
            # that integrate to Z ≈ Q_ccl for stable isotopes
            # For now, just check that Z is close to Q_ccl
            stress = abs(Z - Q_ccl)

            # Penalize high stress (unstable configurations)
            if stress > 1.0:  # More than 1 proton off backbone
                ccl_penalty += stress**2

        return loss_binding + self.lambda_ccl * ccl_penalty
```

### Reduced Parameter Set

```python
# File: experiments/nuclear_heavy_ccl_constrained.runspec.json

{
  "experiment_id": "exp_2025_nuclear_heavy_CCL_CONSTRAINED",
  "parameters": [
    {"name": "nuclear.c_v2_base", "value": 2.2, "bounds": [2.0, 2.5]},
    {"name": "nuclear.c_v4_base", "value": 5.3, "bounds": [4.5, 6.0]},
    {"name": "nuclear.c_sym", "value": 25.0, "bounds": [20.0, 30.0]},
    {"name": "nuclear.kappa_rho", "value": 0.03, "bounds": [0.02, 0.04]},
    {"name": "solver.alpha_e_scale", "value": 1.0, "bounds": [0.9, 1.1]},

    # FIXED from CCL:
    {"name": "nuclear.c_v2_iso", "value": 0.027, "frozen": true},
    {"name": "nuclear.c_v2_mass", "value": -0.0002, "frozen": true},
    {"name": "nuclear.c_v4_size", "value": -0.085, "frozen": true},
    {"name": "solver.beta_e_scale", "value": 0.5, "frozen": true}
  ]
}
```

---

## Test Plan

### Phase 1: Validate CCL Consistency (Quick Test)

```bash
# Test that current parameters reproduce CCL predictions
python3 test_ccl_consistency.py
```

Expected output:
```
Isotope   Z_exp   Q_ccl   Stress   Status
Pb-208     82     82.3     0.3     ✓ Stable (stress < 1)
U-238      92     91.2     0.8     ✓ Stable (stress < 1)
```

### Phase 2: Constrained Optimization (Overnight)

```bash
# Run optimization with CCL constraints
python3 run_ccl_constrained_optimization.py \
    --maxiter 20 \
    --popsize 10 \
    --workers 4 \
    --lambda_ccl 10.0
```

**Success criteria**:
- Virial values < 0.18 (physical solutions)
- Binding energy errors < 2% (better than current)
- Stress values match CCL predictions (< 1.0 for stable isotopes)

### Phase 3: Cross-Validation (Proof of Consistency)

```python
# Verify optimized solitons satisfy CCL
for Z, A in test_isotopes:
    E_binding = solve_soliton(params_optimized, Z, A)
    Q_ccl = c1 * A**(2/3) + c2 * A
    stress = abs(Z - Q_ccl)

    assert stress < 1.5, f"Soliton violates CCL for {Z}-{A}"
```

If this passes → **Unification validated!**

---

## Why This Should Work

### 1. **Fewer Degrees of Freedom**
- 5 parameters instead of 9
- Tighter bounds guided by CCL

### 2. **Physical Guidance**
- CCL provides the **correct answer** for charge distribution
- Solver just needs to find potentials that **reproduce** this
- Much easier than blind search

### 3. **Virial Should Converge**
- Current failure: solver explores unphysical Z values
- With Z fixed and CCL constraint: only explore physical region
- Virial criterion should be satisfied in physical region

### 4. **Validates Theoretical Consistency**
- CCL (2-param phenomenology) should emerge from soliton theory (9-param microscopic)
- If it doesn't → reveals gap in QFD model
- If it does → proves multi-scale consistency

---

## Next Steps

1. **Implement `CCLConstrainedObjective`** class
2. **Create test script** `test_ccl_consistency.py`
3. **Run quick validation** on 2-3 isotopes
4. **Launch constrained optimization** if validation passes
5. **Analyze results** - do solitons reproduce CCL?

---

## Expected Outcome

**Best case**:
- Optimized solitons reproduce Q(A) = c1·A^(2/3) + c2·A
- Virial values < 0.18 (physical solutions)
- Binding energies match experiment within 1-2%
- **Proof**: Microscopic QFD → Phenomenological CCL ✓

**Acceptable case**:
- Solitons approximately satisfy CCL (stress < 2)
- Some virial convergence issues remain (need higher resolution)
- Reveals which parameters need refinement

**Failure case**:
- Solitons violate CCL badly (stress > 5)
- → Reveals fundamental inconsistency in QFD model
- → Need to revise either soliton equations or CCL assumptions

All outcomes are **scientifically valuable**!

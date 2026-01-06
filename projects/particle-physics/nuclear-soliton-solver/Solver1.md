# QFD Nuclear Soliton Solver - Complete Code Review Documentation

**Date**: 2025-12-31 (Updated after stability energy diagnosis)  
**Purpose**: External review of solver physics and implementation  
**Status**: Sign flip diagnosed - V4 too weak, parameters completely wrong

---

## Executive Summary

### Current Status (UPDATED)
- **Physics bugs**: All 3 "Flat Earth" bugs already fixed in code ✓
- **Comparison logic**: CORRECTED to use stability energy (not total mass) ✓
- **Critical discovery**: Solver finds correct **magnitude** (~82 MeV) but **wrong sign** (+82 instead of -81)
- **Root cause**: V4 attractive potential (-46 MeV) is **TOO WEAK** to overcome kinetic energy (+124 MeV)
- **Impact**: Parameters from C-12 optimization are completely wrong for QFD physics

### Key Finding: The Sign Flip

**Target (C-12)**: Stability energy = -81.33 MeV (stable, in potential well)  
**Actual (C-12)**: E_model = +82.46 MeV (unstable, on repulsive branch)  
**Magnitude ratio**: 1.014 (essentially perfect!)  
**Conclusion**: Solver geometry is correct, but found repulsive solution instead of attractive

### Energy Component Breakdown (C-12)

```
T_kinetic:   +123.67 MeV  (gradient energy - TOO HIGH)
V4 (attract): -46.06 MeV  (attractive but TOO WEAK)
V6 (repel):   +24.92 MeV  (quartic stiffness)
V_coul:       -20.88 MeV  (Coulomb attraction)
V_surf:        +0.80 MeV  (surface tension)
──────────────────────────
TOTAL:        +82.46 MeV  (should be -81.33 MeV)
```

**Problem**: V4 needs to be ~-250 MeV (not -46 MeV) to reach target of -81 MeV

---

## 1. Physics Framework: QFD Soliton Theory (CORRECTED TERMINOLOGY)

### Core Hypothesis
Nuclear structure arises from **self-confined scalar field solitons** (Q-balls), NOT from particle-based nuclear force.

**Key claims**:
1. Mass = Vacuum Displacement (no constituent particles)
2. Stability Energy = Energy saved/spent by soliton geometry vs scattered protons
3. Total Mass = Baseline + Stability Energy
4. Derrick virial: T + 3V = 0 (soliton equilibrium, NOT orbital 2T + V)

### CRITICAL TERMINOLOGY CHANGE

**OLD (contaminated)**: "Binding Energy" → implies gluing pre-existing particles  
**NEW (QFD)**: "Stability Energy" or "Topological Defect Energy"

### Mass Framework

**Baseline (scattered vacuum)**:
```
M_baseline = A × M_proton

where A = mass number (# of displaced vacuum cells)
      M_proton ≈ 938.27 MeV (fundamental soliton unit from Chapter 12)
```

**Stability Energy (field geometry)**:
```
E_stability = T_gradients + V_potential

where this represents energy saved (negative) or spent (positive)
      by arranging A units into coherent soliton vs scattered
```

**Total Mass (observable)**:
```
M_total = M_baseline + E_stability

For stable isotopes: E_stability < 0 (saves energy)
For unstable:        E_stability > 0 (costs energy)
```

**Example (C-12)**:
```
M_baseline   = 12 × 938.27 = 11,259.26 MeV
M_experiment = 11,177.93 MeV (AME2020)
E_stability  = 11,177.93 - 11,259.26 = -81.33 MeV ✓ stable
```

### Lagrangian Structure

```
L = ½(∂μψ)² - V(ψ)

V(ψ) = -½α·ρ² + ⅙β·ρ³  (α > 0 attractive, β > 0 stiffness)

where ρ = ψ²
```

**Energy functional**:
```
E_stability = ∫[ ½|∇ψ|² + V(ρ) ] dV

            = T_kinetic + V_potential
```

**Expected**: For C-12, E_stability ≈ -81 MeV (stable configuration)

---

## 2. Three "Flat Earth" Bugs - All Fixed ✓

### Bug #1: Neutron Star Collapse (c_sym) - FIXED ✓

**Physics Error**: Symmetry energy forces electron cloud to collapse into nucleus

**Code location**: `qfd_solver.py` line 261-295

```python
def symmetry_energy(self) -> torch.Tensor:
    if self.c_sym == 0.0:
        return torch.tensor(0.0, device=self.device)  # ← FIXED: Returns 0
    
    # (code that forces electron/nucleus density matching - disabled when c_sym=0)
```

**Status**: ✓ Code correctly returns 0 when c_sym=0

---

### Bug #2: Double Mass Counting (M_constituents) - FIXED ✓

**Physics Error**: Adding constituent masses double-counts mass

**Code location**: `parallel_objective.py` lines 266-283 (CORRECTED 2025-12-31)

```python
# The Experimental Truth
exp_mass_total = self.exp_data[(Z, A)]['E_exp'] 

# The QFD Vacuum Baseline (A * Unit Cell)
vacuum_baseline = A * M_PROTON

# The Target Stability Energy (This will be negative for stable atoms)
target_stability_energy = exp_mass_total - vacuum_baseline

# The Solver Output
# E_model represents the "shape energy" relative to the baseline
solved_stability_energy = result['E_model']

# The Loss
# We want the solver to find the specific geometry that provides
# exactly the required stability deficit.
loss = (solved_stability_energy - target_stability_energy)**2
```

**Status**: ✓ Corrected to compare stability energies (no M_constituents addition)

---

### Bug #3: Repulsive Potential (V4 sign) - FIXED ✓

**Physics Error**: Positive V4 creates repulsion, can't form bound states

**Code location**: `qfd_solver.py` line 196-199

```python
def potential_from_density(self, rho: torch.Tensor, alpha: float, beta: float):
    V4 = -0.5 * alpha * (rho * rho).sum() * self.dV  # NEGATIVE (attractive!)
    V6 = (1.0/6.0) * beta * (rho * rho * rho).sum() * self.dV
    return V4, V6
```

**Verification**: With c_v2_base=3.643 → alpha_eff=3.84 → V4 ≈ -46 MeV ✓ negative

**Status**: ✓ V4 is correctly negative (attractive)

---

## 3. Core Solver Code

### 3.1 Energy Calculation (`qfd_solver.py` lines 297-331)

```python
def energies(self) -> Dict[str, torch.Tensor]:
    psiN = self.psi_N
    psiE = self.psi_e
    rhoN = self.nucleon_density()  # ψ_N²
    rhoE = self.electron_density()  # ψ_e²
    
    # Kinetic energies (gradient terms)
    T_N = self.kinetic_scalar(psiN, self.kN)
    T_e = self.kinetic_scalar(psiE, self.ke)
    T_rotor = self.rotor_terms.T_rotor(self.B_N)
    
    # Potentials (V4 attractive, V6 repulsive stiffness)
    V4_N, V6_N = self.potential_from_density(rhoN, self.alpha_eff, self.beta_eff)
    V4_e, V6_e = self.potential_from_density(rhoE, self.alpha_e, self.beta_e)
    
    # Interactions
    V_surf = self.surface_energy(rho_tot)
    V_coul = self.coulomb_cross_energy(rhoN, rhoE)
    E_sym = self.symmetry_energy()  # Returns 0 if c_sym=0
    
    # Constraints
    V_mass_N = self.mass_penalty(rhoN, float(self.A), self.mass_penalty_N)
    V_mass_e = self.mass_penalty(rhoE, float(self.Z), self.mass_penalty_e)
    V_rotor = self.rotor_terms.V_rotor(self.B_N)
    
    return dict(
        T_N=T_N, T_e=T_e, T_rotor=T_rotor,
        V4_N=V4_N, V6_N=V6_N, V4_e=V4_e, V6_e=V6_e,
        V_iso=torch.tensor(0.0, device=self.device),
        V_rotor=V_rotor,
        V_surf=V_surf,
        V_coul_cross=V_coul,
        V_mass_N=V_mass_N, V_mass_e=V_mass_e,
        V_sym=E_sym,
    )
```

### 3.2 SCF Minimization (`qfd_solver.py` lines 423-461)

```python
def scf_minimize(model: Phase8Model, iters_outer:int=360, lr_psi:float=1e-2, 
                 lr_B:float=1e-2, early_stop_vir:float=0.2, verbose:bool=False):
    """
    Self-consistent field optimization using Adam optimizer.
    
    Minimizes: loss = total_energy + 10.0 * virial²
    
    Returns: (best_result, virial, energies)
    """
    optim = torch.optim.Adam([
        {"params": [model.psi_N], "lr": lr_psi},
        {"params": [model.psi_e], "lr": lr_psi},
        {"params": [model.B_N], "lr": lr_B},
    ])
    
    best = dict(E=float("inf"), vir=float("inf"))
    best_state = None
    
    for it in range(1, iters_outer+1):
        optim.zero_grad()
        energies = model.energies()
        total = sum(energies.values())  # ← Stability energy (should be negative)
        vir = model.virial(energies)
        loss = total + 10.0 * vir*vir    # ← Minimizes this
        loss.backward()
        optim.step()
        model.projections()
        
        e_val = float(total.detach())
        vir_val = float(vir.detach())
        
        # Track best by lowest virial
        if abs(vir_val) < abs(best.get("vir", float("inf"))):
            best = dict(E=e_val, vir=vir_val)
            best_state = [...] # Save state
        
        if abs(vir_val) <= early_stop_vir:
            break
    
    # Restore best state
    [restore fields...]
    
    return best, best.get("vir", float("nan")), best_energies or energies
```

**CRITICAL ISSUE**: 
- Line 438 minimizes `loss = total + 10*vir²`
- If total starts positive, minimization drives toward +0 (not through 0 to negative)
- Solver gets stuck on repulsive branch (+82 MeV) instead of attractive (-81 MeV)

### 3.3 Virial Check (`qfd_solver.py` lines 333-355)

```python
def virial(self, energies: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Derrick scaling for 3D solitons: T + 3V = 0
    
    NOT orbital virial (2T + V = 0)
    """
    kinetic = energies["T_N"] + energies["T_e"] + energies["T_rotor"]
    total = sum(energies.values())
    potential = total - kinetic  # V = E_total - T
    return kinetic + 3.0 * potential  # T + 3V
```

**Status**: ✓ Correct Derrick scaling (verified virial ≈ 0.033 for C-12)

---

## 4. Corrected Optimization Framework

### 4.1 Parallel Objective (`parallel_objective.py` lines 254-290) - UPDATED

```python
def __call__(self, params: Dict[str, float]) -> float:
    """
    Evaluate objective function with parallel isotope solving.
    
    CORRECTED (2025-12-31): Compares stability energies, not total masses
    """
    # [Submit solves to thread pool...]
    # [Collect results...]
    
    from qfd_metaopt_ame2020 import M_PROTON
    
    errors = []
    for result in results:
        if result['status'] != 'success':
            continue
        
        Z, A = result['Z'], result['A']
        if (Z, A) not in self.exp_data:
            continue
        
        # ═══════════════════════════════════════════════════════════════
        # PURE QFD SOLITON PHYSICS (CORRECTED)
        # ═══════════════════════════════════════════════════════════════
        
        # The Experimental Truth
        exp_mass_total = self.exp_data[(Z, A)]['E_exp'] 
        
        # The QFD Vacuum Baseline (A * Unit Cell)
        vacuum_baseline = A * M_PROTON
        
        # The Target Stability Energy (negative for stable isotopes)
        target_stability_energy = exp_mass_total - vacuum_baseline
        
        # The Solver Output (shape energy)
        solved_stability_energy = result['E_model']
        
        # The Loss
        # Find geometry that provides exact required stability deficit
        loss = (solved_stability_energy - target_stability_energy)**2
        errors.append(loss)
    
    if not errors:
        return 1e9  # Catastrophic penalty
    
    return sum(errors) / len(errors)
```

**KEY CHANGE**: 
- OLD: Compared E_model (~82) to M_total (~11,177) → 99% error
- NEW: Compares E_stability (+82) to E_target (-81) → reveals sign flip

---

## 5. Diagnostic Test Results (UPDATED)

### Test: C-12 with Corrected Comparison

**Parameters** (from C-12 golden probe):
```python
params = {
    'c_v2_base': 3.643,  # → alpha_eff = 3.84
    'c_v2_iso': 0.0135,
    'c_v2_mass': 0.0005,
    'c_v4_base': 9.33,   # → beta_eff = 9.03
    'c_v4_size': -0.129,
    'alpha_e_scale': 1.181,
    'beta_e_scale': 0.523,
    'c_sym': 0.0,        # ✓ Pure soliton
    'kappa_rho': 0.044
}
```

**Results**:
```
Target stability:  -81.33 MeV (C-12 should save 81 MeV vs 12 protons)
Solved stability:  +82.46 MeV (solver found repulsive configuration)
Magnitude ratio:   1.014 (essentially perfect!)
Sign:              FLIPPED ✗
Virial:            0.0333 (excellent convergence)
```

### Energy Component Breakdown

| Component | Value (MeV) | Type | Expected |
|-----------|-------------|------|----------|
| T_N | +22.65 | Nucleon kinetic | Positive |
| T_e | +101.01 | Electron kinetic | Positive |
| T_rotor | +0.01 | Rotor kinetic | Small |
| **T_total** | **+123.67** | **Total kinetic** | **~50 MeV?** |
| V4_N | -30.53 | Nucleon attraction | More negative! |
| V4_e | -15.54 | Electron attraction | More negative! |
| **V4_total** | **-46.06** | **Total V4** | **~-250 MeV?** |
| V6_N | +24.50 | Nucleon stiffness | Positive |
| V6_e | +0.42 | Electron stiffness | Small |
| V_coul | -20.88 | Coulomb | Negative |
| V_surf | +0.80 | Surface | Small |
| V_sym | 0.00 | Symmetry (disabled) | Zero ✓ |
| **TOTAL** | **+82.46** | **Stability energy** | **-81.33 MeV** |

**Analysis**:
```
Kinetic + V4 = +123.67 - 46.06 = +77.61 MeV

→ Kinetic DOMINATES over V4 (repulsive regime)
→ Need V4 ≈ -250 MeV to reach -81 MeV target
→ c_v2_base needs to be ~5-6x larger (≈15-20) OR grid too coarse
```

---

## 6. Root Cause Analysis (UPDATED)

### Why E_model = +82.46 Instead of -81.33?

**Three possible causes**:

#### Cause 1: V4 Too Weak (c_v2_base too small)

Current: c_v2_base = 3.643 → alpha_eff = 3.84 → V4 = -46 MeV

To get -81 MeV total with ~124 MeV kinetic:
```
Target: T + V4 + V6 + V_coul ≈ -81
        124 + V4 + 25 + (-21) ≈ -81
        V4 ≈ -209 MeV (need 4.5x stronger!)
```

**Hypothesis**: c_v2_base should be ~15-18 (not 3.6)

#### Cause 2: Kinetic Energy Too High (grid too coarse)

Grid = 32 points → coarse gradients → inflated |∇ψ|²

Expected kinetic for C-12: ~40-60 MeV?  
Actual kinetic: 124 MeV (2-3x too high!)

**Hypothesis**: Grid 32 → 48 or 64 might reduce kinetic to realistic levels

#### Cause 3: Normalization/Units Issue

Field units or dV normalization might be off by constant factor

**Hypothesis**: All energies scaled wrong, need to check dimension analysis

---

## 7. Why Previous Optimization "Succeeded"

### The False Success (Loss = 0.000479)

**What happened**:
```python
# OLD COMPARISON (wrong)
soliton_mass = E_model = 82 MeV
exp_mass = 11,177 MeV
rel_error = (82 - 11,177) / 11,177 = -0.9927 (99% error!)
loss = 0.9927² = 0.986 (TERRIBLE!)

# But solver was comparing to DIFFERENT baseline:
# It was minimizing virial, not energy error
# So low loss didn't mean good energy match!
```

**Why it seemed good**:
- Virial converged to 0.033 ✓ (geometric stability achieved)
- Differential evolution declared "convergence" based on parameter variance
- Loss value of 0.000479 was actually measuring virial convergence, not energy accuracy

**The truth**:
- Geometric stability: CORRECT (virial ≈ 0)
- Energy magnitude: CORRECT (|82| ≈ |81|)
- Energy sign: WRONG (+82 instead of -81)
- Parameters: COMPLETELY WRONG (c_v2_base off by factor of ~5)

---

## 8. Proposed Fixes

### Option A: Increase c_v2_base (Strengthen V4)

**Hypothesis**: c_v2_base ≈ 3.6 is too small

**Test**: Try c_v2_base ∈ [10, 20] to get V4 ≈ -200 to -300 MeV

**Implementation**:
```python
# In runspec
"c_v2_base": {
    "value": 15.0,
    "bounds": [10.0, 25.0]
}
```

**Expected**: E_stability → negative (attractive regime)

### Option B: Increase Grid Resolution

**Hypothesis**: Grid = 32 is too coarse, inflating kinetic energy

**Test**: Run same parameters with grid = 48 or 64

**Implementation**:
```python
# In scf_solver_options
"grid_points": 48  # or 64
```

**Expected**: Kinetic energy drops from ~124 MeV to ~50-60 MeV

### Option C: Add Negative Offset to Loss

**Hypothesis**: Solver gets stuck in local minimum on repulsive branch

**Test**: Modify SCF loss to favor negative energies

**Implementation**:
```python
# In scf_minimize (line 438)
loss = total + 10.0 * vir*vir - 100.0  # Shift basin down
```

**Expected**: Optimizer explores negative energy region

### Option D: Different Initialization

**Hypothesis**: "gauss_cluster" init puts solver on wrong branch

**Test**: Try different initialization schemes

**Implementation**:
```python
# Try random, or custom init with pre-configured well
model.initialize_fields(seed=seed, init_mode="random")
```

**Expected**: May land on attractive branch instead of repulsive

---

## 9. Parameter Regime Analysis (INVALIDATED)

### Previous Understanding (WRONG)

~~C-12 optimized to c_v2_base ≈ 3.64 with "excellent" fit~~  
~~This validated β ≈ 3.6 close to theoretical 3.06~~

### New Understanding (CORRECT)

**C-12 optimization FAILED**:
- Found geometry with correct magnitude but wrong sign
- Parameters give repulsive solution (+82 MeV) not attractive (-81 MeV)
- Low loss was measuring virial convergence, not energy accuracy
- c_v2_base ≈ 3.6 is ~5x too small

**Implication**: All previous parameter fits are invalid!

### True Parameter Regime (Estimated)

Based on energy balance requirements:

```
c_v2_base: ~15-20 (not 3.6)
c_v4_base: ~5-10 (possibly correct)
c_sym: 0.0 (correct - pure soliton)
```

**Still unknown**: Correct grid resolution and field normalization

---

## 10. Outstanding Questions for External Review

### Physics Questions

1. **What should c_v2_base actually be?**
   - Current: 3.6 gives V4 = -46 MeV (too weak)
   - Need: V4 ≈ -250 MeV to get -81 MeV total
   - Implies: c_v2_base ≈ 15-20?

2. **Is kinetic energy realistic?**
   - Current: T ≈ 124 MeV for C-12
   - Seems high - grid artifact?
   - Should grid be 48 or 64 instead of 32?

3. **Why does solver find repulsive branch?**
   - Initialization issue?
   - Loss function structure?
   - Need basin-hopping or simulated annealing?

4. **Is field normalization correct?**
   - All energies in MeV units?
   - dV = dx³ correctly scaled?
   - ψ dimensionless or has units?

### Code Questions

5. **Should we modify SCF loss function?**
   - Current: loss = E + 10*vir²
   - Add negative offset to favor attractive branch?
   - Or use different optimizer (basin-hopping)?

6. **What grid resolution is appropriate?**
   - Current: 32 (fast but coarse)
   - Test: 48 or 64 (slower but more accurate)
   - Trade-off: Speed vs accuracy

7. **How to escape local minimum?**
   - Different initialization (random vs gauss)?
   - Simulated annealing?
   - Multiple starting points?

---

## 11. Files Inventory

### Core Solver
- `src/qfd_solver.py` (658 lines) - Main Phase8Model solver
  - ✓ V4 correctly negative (line 197)
  - ✓ Virial uses Derrick scaling (line 355)
  - ⚠ SCF minimizes toward +0 (not through to negative)

### Optimization
- `src/parallel_objective.py` (315 lines) - GPU-parallel objective
  - ✓ CORRECTED stability energy comparison (lines 266-283)
  - ✓ No M_constituents double-counting
  - ✓ Uses M_proton baseline for QFD physics

### Experiments
- `experiments/c12_golden_probe.runspec.json` - C-12 with c_sym=22.3
  - ✗ INVALID: Parameters give wrong sign
  - ✗ Loss=0.000479 was false positive (virial convergence, not energy)
  
- `experiments/octet_verification.runspec.json` - 8 isotopes with c_sym=0
  - ⚠ NOT RUN with corrected comparison yet
  - ⚠ Parameters likely still wrong

### Diagnostics
- `test_stability_energy_fix.py` - Confirms sign flip
- `diagnose_energy_components.py` - Shows V4 too weak
- `Solver1.md` - This document

---

## 12. Recommended Next Steps

### Immediate Actions

1. **Test Option A** (larger c_v2_base):
   ```bash
   # Try c_v2_base = 15.0 with wider bounds
   # See if E_model goes negative
   ```

2. **Test Option B** (finer grid):
   ```bash
   # Run C-12 with grid=48 or 64
   # Check if kinetic energy drops to ~50 MeV
   ```

3. **Energy unit audit**:
   - Verify dV = dx³ is correct
   - Check ψ normalization
   - Confirm all terms in MeV units

### Validation Plan

4. **If sign flips to negative**:
   - Verify magnitude still ~81 MeV
   - Check virial still < 0.1
   - Re-optimize C-12 from scratch
   - Test on octet

5. **If still positive**:
   - Try Option C (loss offset)
   - Try Option D (different init)
   - Consider basin-hopping optimizer

### Long-term

6. **After finding stable parameters**:
   - Re-run C-12 optimization
   - Test β universality on octet
   - Profile parameter regimes by mass number
   - Check if single β works across A = 1-208

---

## 13. Code Verification Checklist (UPDATED)

- [x] Bug #1 (c_sym): Returns 0 when c_sym=0 ✓
- [x] Bug #2 (M_constituents): Uses stability energy comparison ✓
- [x] Bug #3 (V4 sign): V4 is negative (attractive) ✓
- [x] Virial formula: T + 3V Derrick scaling ✓
- [x] Comparison logic: Stability energy (not total mass) ✓
- [ ] **V4 magnitude**: TOO WEAK (-46 vs need -250 MeV) ✗
- [ ] **Kinetic energy**: TOO HIGH (124 vs expect ~50 MeV) ✗
- [ ] **Energy sign**: WRONG (+82 vs -81 MeV) ✗
- [ ] **Parameter values**: c_v2_base too small by ~5x ✗

**Critical unfixed issues**:
1. V4 attractive potential is 5x too weak
2. Kinetic energy is 2-3x too high (grid artifact?)
3. Solver finds repulsive branch instead of attractive
4. All previous parameter optimizations are invalid

---

## Appendix A: C-12 Energy Component Breakdown (Complete)

From solver diagnostics (actual output, not estimates):

| Component | Value (MeV) | Physical Meaning | Target? |
|-----------|-------------|------------------|---------|
| T_N | +22.65 | Nucleon field gradients | ~10 MeV? |
| T_e | +101.01 | Electron field gradients | ~20 MeV? |
| T_rotor | +0.01 | Angular momentum | Negligible |
| **T_total** | **+123.67** | **Total kinetic** | **~30-50 MeV?** |
| | | | |
| V4_N | -30.53 | Nucleon cohesion (α) | ~-150 MeV? |
| V4_e | -15.54 | Electron cohesion | ~-50 MeV? |
| **V4_total** | **-46.06** | **Total attraction** | **~-200 MeV?** |
| | | | |
| V6_N | +24.50 | Nucleon stiffness (β) | ~20 MeV ✓ |
| V6_e | +0.42 | Electron stiffness | ~1 MeV ✓ |
| V_coul | -20.88 | Nucleus-electron | ~-20 MeV ✓ |
| V_surf | +0.80 | Surface tension | ~1 MeV ✓ |
| V_sym | 0.00 | Disabled (c_sym=0) | 0 ✓ |
| | | | |
| **TOTAL** | **+82.46** | **Stability energy** | **-81.33** |

**Key problems**:
- T_total is 2-3x higher than expected (grid too coarse?)
- V4_total is 4-5x weaker than needed (c_v2_base too small?)
- Net result: Repulsive (+82) instead of attractive (-81)

**Model parameters used**:
- alpha_eff = 3.8429 (from c_v2_base=3.643)
- beta_eff = 9.0347 (from c_v4_base=9.33)
- Grid = 32 points
- dx = 1.0 fm

---

## Appendix B: Sign Flip Diagnosis Script

```python
"""
Quick test to diagnose sign flip in C-12 stability energy
"""
import sys
sys.path.insert(0, 'src')
from parallel_objective import run_solver_direct
from qfd_metaopt_ame2020 import M_PROTON
import pandas as pd

params = {
    'c_v2_base': 3.643,
    'c_v2_iso': 0.0135,
    'c_v2_mass': 0.0005,
    'c_v4_base': 9.33,
    'c_v4_size': -0.129,
    'alpha_e_scale': 1.181,
    'beta_e_scale': 0.523,
    'c_sym': 0.0,
    'kappa_rho': 0.044
}

result = run_solver_direct(A=12, Z=6, params=params, 
                           grid_points=32, iters_outer=150, device='cuda')

ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == 6) & (ame_data['A'] == 12)]
exp_mass = float(row.iloc[0]['E_exp_MeV'])

target_stability = exp_mass - (12 * M_PROTON)
solved_stability = result['E_model']

print(f"Target: {target_stability:.2f} MeV (should be negative)")
print(f"Actual: {solved_stability:.2f} MeV")
print(f"Ratio:  {abs(solved_stability)/abs(target_stability):.3f}")
print(f"Sign:   {'MATCH' if solved_stability*target_stability > 0 else 'FLIPPED'}")
```

**Expected output**:
```
Target: -81.33 MeV (should be negative)
Actual: +82.46 MeV
Ratio:  1.014
Sign:   FLIPPED
```

---

**END OF DOCUMENTATION**

External reviewers: 
1. Confirm V4 is too weak (need c_v2_base ≈ 15-20 instead of 3.6)
2. Confirm kinetic energy too high (need finer grid?)
3. Suggest method to flip solver from repulsive to attractive branch

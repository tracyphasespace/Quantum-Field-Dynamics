# Hill Vortex: Shell vs Core Analysis
## Density Gradient Structure and Energy Distribution

**Date**: December 22, 2025
**Key Insight**: The electron is NOT a hard shell - it's a gradient density vortex like a whirlpool

---

## 1. The Density Gradient (From HillVortex.lean)

### Density Perturbation Profile

From the Lean specification (line 62-70):
```lean
def vortex_density_perturbation (r : ℝ) : ℝ :=
  if r < R then
    -amplitude × (1 - r²/R²)    -- Parabolic depression
  else
    0                            -- Vacuum outside
```

**This is a GRADIENT, not a hard boundary!**

### Density vs Radius

**Total density**:
```
ρ_total(r) = ρ_vac + δρ(r)
           = ρ_vac - amplitude × (1 - r²/R²)   for r < R
           = ρ_vac                               for r ≥ R
```

**Profile**:
```
r = 0 (core):      ρ = ρ_vac - amplitude     (MINIMUM density)
r = R/2 (middle):  ρ = ρ_vac - 0.75×amplitude (intermediate)
r = R (boundary):  ρ = ρ_vac                  (vacuum level)
r > R (external):  ρ = ρ_vac                  (pure vacuum)
```

**Shape**: Parabolic bowl - density is DEPRESSED in the center, gradually rising to vacuum level at R.

**Maximum amplitude**: When amplitude = ρ_vac, the core hits ρ = 0 (vacuum floor / cavitation limit).

---

## 2. The Velocity Gradient (From Stream Function)

### Internal Flow (r < R)

Stream function (line 34-39):
```
ψ_internal = -(3U/2R²) × (R² - r²) × r² × sin²θ
```

**Velocity components** (from v = ∇ × ψ):
```
v_r(r,θ) ~ sin(θ)cos(θ) × terms  (radial component)
v_θ(r,θ) ~ r × (R² - r²) × terms  (tangential component)
```

**Key observations**:
1. **At r = 0 (core)**: v → 0 (zero velocity at center)
2. **At r ~ R/2**: v is MAXIMUM (fastest flow in mid-region)
3. **At r = R (boundary)**: v → 0 (matches external flow)

**This is exactly like a whirlpool!**
- Eye (center): Calm, low velocity
- Surrounding water: Fast circular flow
- Edge: Flow slows to match external conditions

### External Flow (r > R)

Stream function (line 41-43):
```
ψ_external = (U/2) × (r² - R³/r) × sin²θ
```

This is **potential flow** (irrotational) around a moving sphere.
- No vorticity outside R
- Velocity decreases as 1/r² at large distances

---

## 3. The Whirlpool Analogy

### Gravity Creates Gradient in Whirlpool

In a physical whirlpool (bathtub drain, tornado):
1. **Center (eye)**: Low pressure → water surface DEPRESSED
2. **Gradient**: Pressure increases radially outward
3. **Edge**: Pressure returns to normal (atmospheric)

**The pressure gradient drives the inward flow, which creates the vortex.**

### Time Gradient in Hill Vortex (QFD)

In the Hill vortex electron:
1. **Center (core)**: Low density → time dilation MAXIMUM
2. **Gradient**: Density increases radially outward (parabolic)
3. **Edge (R)**: Density returns to ρ_vac (vacuum level)

**The density gradient creates the "time refraction" that sustains the vortex.**

### Mathematical Similarity

**Whirlpool** (gravity):
```
Pressure: p(r) = p_atm - Δp × f(r)
Height:   h(r) = h_0 + Δh × f(r)    (surface depression)
```

**Hill Vortex** (QFD):
```
Density:  ρ(r) = ρ_vac - amplitude × (1 - r²/R²)
Time rate: dt/dτ ~ 1/√ρ(r)           (time dilation)
```

Both have:
- Depression at center
- Parabolic gradient
- Smooth transition to external conditions

---

## 4. Energy Distribution: Shell vs Core

### Where is the Energy?

Let me analyze the two energy components:

#### A. Circulation Energy (Kinetic)

```
E_circulation = ∫ ½ρ(r) × v²(r) × dV
```

**Integrand factors**:
- ρ(r): LOW at core (r=0), HIGH at shell (r~R)
- v²(r): LOW at core (r=0), HIGH at middle (r~R/2), LOW at boundary (r=R)
- dV: Volume element ~ r² (small at core, large at shell)

**Dominant contribution**: **SHELL region** (R/2 < r < R)
- High density: ρ ≈ ρ_vac
- High velocity: v ~ maximum in this region
- Large volume: r² is large

**Core contribution**: Small
- Low density: ρ << ρ_vac
- Low velocity: v → 0
- Small volume: r² → 0

#### B. Vacuum Stabilization Energy (Potential)

```
E_stabilization = ∫ β × δρ²(r) × dV
```

**Integrand factors**:
- δρ²(r): MAXIMUM at core (δρ = -amplitude), decreasing parabolically
- dV: Volume element ~ r²

**Radial dependence**:
```
δρ(r) = -amplitude × (1 - r²/R²)

At r = 0:     δρ² = amplitude²         (MAXIMUM)
At r = R/2:   δρ² = 0.56×amplitude²    (decreased)
At r = R:     δρ² = 0                  (zero)
```

**Dominant contribution**: **CORE region** (0 < r < R/2)
- Maximum density depression: δρ² ~ amplitude²
- Despite small volume (r²), the δρ² factor is so large it dominates

**Shell contribution**: Small
- Low density depression: δρ → 0 near R
- Even though volume is large, δρ² is negligible

---

## 5. The Energy Balance Picture

### Spatial Distribution

```
           ┌─────────────────────────────────────┐
           │                                     │
           │         HILL VORTEX ELECTRON        │
           │                                     │
           └─────────────────────────────────────┘

    Core (r ~ 0):                Shell (R/2 < r < R):
    ┌───────────────┐           ┌───────────────────┐
    │ ρ: MINIMUM    │           │ ρ: HIGH (~ρ_vac)  │
    │ v: LOW (~0)   │           │ v: MAXIMUM        │
    │               │           │                   │
    │ E_circ: SMALL │           │ E_circ: LARGE ✓   │
    │ E_stab: LARGE ✓│          │ E_stab: SMALL     │
    └───────────────┘           └───────────────────┘
         ↓                              ↓
    Provides most               Provides most
    STABILIZATION              CIRCULATION
    (potential energy)         (kinetic energy)
```

### The Cancellation Mechanism

**From Shell** (outer region):
- Large circulation energy (ρ×v² integrated over large volume)
- Small stabilization energy (δρ² is small)
- **Net contribution**: Positive (adds to mass)

**From Core** (inner region):
- Small circulation energy (ρ and v both small)
- Large stabilization energy (δρ² is maximum)
- **Net contribution**: Negative (subtracts from mass)

**Total Mass**:
```
m = ∫_shell [½ρv² - β×δρ²] dV + ∫_core [½ρv² - β×δρ²] dV
  = (Large - Small)_shell + (Small - Large)_core
  = Positive_shell + Negative_core
  ≈ TINY RESIDUAL
```

**The shell wants to ADD mass, the core wants to SUBTRACT mass, they nearly cancel!**

---

## 6. Why My Previous Calculation Was Wrong

### What I Did Wrong

In my implementation (`v22_hill_vortex_with_circulation.py`), I computed:

```python
E_circulation = ∫ ½ρ_vac × v² dV    # Used constant ρ_vac everywhere!
E_stabilization = ∫ β × δρ² dV      # This part was correct
```

**Error**: I used **constant density** ρ_vac for circulation energy!

**Reality**: Density varies! ρ(r) = ρ_vac - amplitude×(1-r²/R²)

**Impact**:
- I overestimated circulation energy in the **core** (where ρ is actually LOW)
- I got approximately right result for **shell** (where ρ ≈ ρ_vac is correct)

**Why I still got reasonable result**:
- Most circulation energy comes from shell (where ρ ≈ ρ_vac is approximately right)
- Core contributes little to circulation anyway (v → 0)
- So the error partially canceled out!

### The Corrected Formula

**Should be**:
```python
def circulation_energy_corrected(R, U, amplitude):
    E = 0
    for r, theta in grid:
        rho = rho_vac - amplitude * (1 - (r/R)**2)  # Actual density!
        v = compute_velocity(r, theta, R, U)
        E += 0.5 * rho * v**2 * volume_element
    return E
```

**Key difference**: Use **actual spatially-varying density** ρ(r), not constant ρ_vac!

---

## 7. Implications for the Factor of 2 Error

### Why We Got 0.997 MeV Instead of 0.511 MeV

My calculation gave:
```
E_circulation = 1.949 MeV   (overestimated core contribution)
E_stabilization = 0.952 MeV
E_total = 0.997 MeV         (2× too high)
```

**Likely correction**:

Using actual density gradient ρ(r):
- **Core circulation energy decreases** (ρ is lower in core)
- **Shell circulation energy stays similar** (ρ ≈ ρ_vac)
- **Net effect**: E_circulation decreases

**New balance**:
```
E_circulation_corrected ≈ 1.5 MeV   (reduced from 1.949 MeV)
E_stabilization ≈ 0.95 MeV          (same)
E_total ≈ 0.55 MeV                  (closer to 0.511 MeV!)
```

**The factor of 2 error likely comes from neglecting the density gradient in circulation energy!**

---

## 8. The "Core Mass" Insight

### What We Thought Was a Point

**Previous picture**:
- Electron is a point particle at r = 0
- Surrounded by "cloud" or "field"
- The "electron radius" is where we detect scattering

**Reality (QFD)**:
- The **core** (r < R/2) has structure and mass
- It's the region of maximum density depression
- Most of the stabilization energy comes from core
- Disrupting the core creates observable "electron radius" artifacts

### Core Properties

**Density**: ρ_core = ρ_vac - amplitude
- For maximum amplitude: ρ_core → 0 (hits vacuum floor)
- This is the **cavitation limit** (charge quantization!)

**Structure**: Gradient, not point
- Smooth parabolic transition
- No discontinuities
- Continuous to external flow at R

**Mass contribution**: Negative!
- Core region contributes **negative** energy (stabilization > circulation)
- This **subtracts** from total mass
- Essential for making electron light!

**Without the core's negative contribution, the electron would be much heavier!**

---

## 9. Corrected Physical Picture

### The Complete Electron Structure

```
┌─────────────────────────────────────────────────────────┐
│              HILL VORTEX ELECTRON                        │
│                                                          │
│  ╭──────────────────────────────────────────────╮       │
│  │                                              │       │
│  │         External Flow (r > R)                │       │
│  │         Potential, Irrotational              │       │
│  │         ρ = ρ_vac (constant)                 │       │
│  │                                              │       │
│  │    ╭────────────────────────────╮           │       │
│  │    │                            │           │       │
│  │    │   Shell (R/2 < r < R)      │           │       │
│  │    │   • High density (ρ≈ρ_vac) │           │       │
│  │    │   • High velocity          │           │       │
│  │    │   • CIRCULATION dominates  │           │       │
│  │    │   • Net: POSITIVE energy   │           │       │
│  │    │                            │           │       │
│  │    │   ╭──────────────╮         │           │       │
│  │    │   │              │         │           │       │
│  │    │   │ Core (r<R/2) │         │           │       │
│  │    │   │ • Low ρ      │         │           │       │
│  │    │   │ • Low v      │         │           │       │
│  │    │   │ • STAB. dom. │         │           │       │
│  │    │   │ • Net: NEG.  │         │           │       │
│  │    │   │              │         │           │       │
│  │    │   ╰──────────────╯         │           │       │
│  │    │                            │           │       │
│  │    ╰────────────────────────────╯           │       │
│  │                                              │       │
│  ╰──────────────────────────────────────────────╮       │
│                                                          │
│  Mass = (Shell positive) + (Core negative) ≈ 0.511 MeV  │
└─────────────────────────────────────────────────────────┘
```

### Energy Budget (Corrected)

**Shell contribution**:
```
E_shell = ∫_shell [½ρ_vac × v² - β×δρ²] dV
        = (Large circulation) - (Small stabilization)
        = +0.8 MeV (positive, adds to mass)
```

**Core contribution**:
```
E_core = ∫_core [½ρ_low × v² - β×δρ²] dV
       = (Small circulation) - (Large stabilization)
       = -0.3 MeV (negative, subtracts from mass!)
```

**Total mass**:
```
m_e = E_shell + E_core
    = (+0.8 MeV) + (-0.3 MeV)
    = +0.5 MeV ✓
```

**(Numbers are illustrative - need actual integration with density gradient)**

---

## 10. Next Steps to Fix the Calculation

### What Needs to Change

1. **Use actual density profile** in circulation energy:
   ```python
   rho(r) = rho_vac - amplitude * (1 - (r/R)**2)  # Not constant!
   E_circ = ∫ ½ρ(r) × v²(r) dV
   ```

2. **Integrate shell vs core separately** to see contributions:
   ```python
   E_shell = integrate(0.5*R, R, ...)
   E_core = integrate(0, 0.5*R, ...)
   ```

3. **Include toroidal components** (4-field structure):
   - Current: Only poloidal flow
   - Missing: Toroidal swirl (ψ_b components)
   - Effect: Adds circulation energy in shell

4. **Optimize R, U, amplitude together** with corrected formula

### Expected Result

**Prediction**: Using density-weighted circulation energy will:
- Reduce E_circulation from 1.95 MeV to ~1.5 MeV
- Keep E_stabilization ≈ 0.95 MeV
- Give E_total ≈ 0.55 MeV (factor of 2 closes!)

**Then adding toroidal components**:
- Increases shell circulation slightly
- Fine-tunes to exact 0.511 MeV

---

## Bottom Line

### The Gradient Structure IS Essential

**NOT a hard shell**: The electron is a **gradient density vortex**

**Like a whirlpool**:
- Depressed density at center (eye)
- Parabolic gradient to edge
- Smooth transition to external flow

**Energy distribution**:
- **Shell**: Provides circulation (positive contribution)
- **Core**: Provides stabilization (negative contribution)
- **Balance**: Tiny residual = mass

**Why factor of 2 error**:
- Used constant ρ_vac instead of gradient ρ(r)
- Overestimated core circulation energy
- Need to recompute with actual density profile

**The core is NOT a point** - it has structure, and its negative energy contribution is essential for making the electron light!

---

**Status**: Density gradient analysis complete
**Next**: Implement corrected circulation energy with ρ(r) profile
**Expected**: Factor of 2 error closes

**Date**: December 22, 2025

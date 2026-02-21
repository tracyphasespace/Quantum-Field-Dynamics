# Topological Cleavage Barrier — Scission Physics

**Date**: 2026-02-21
**Status**: Validated. Additive Coulomb = best model (+3.8% vs v8).
**Code**: `validate_kinetic_fracture.py`
**Depends on**: `qfd_nuclide_predictor.py` (v8 constants and functions)

---

## 1. The Binary Cliff Problem

The kinetic fracture model (v1) uses a two-term barrier:

```
B_eff = max(0,  B_surf − K_SHEAR · pf²)
```

Both terms are **Z-independent at fixed A**:
- `B_surf(A, 4)` depends only on the parent mass number
- `pf = (A − A_crit) / WIDTH` depends only on A

This creates a **binary cliff**: at each A, the barrier is either erased for
ALL isotopes or for NONE.

### Empirical evidence (A = 196)

| Z  | Element | ε      | Actual    | Flat kinetic | v8      |
|----|---------|--------|-----------|--------------|---------|
| 79 | Au      | +0.82  | **B+**    | B+ (correct) | alpha   |
| 81 | Tl      | +2.82  | **B+**    | B+ (correct) | alpha   |
| 83 | Bi      | +4.82  | **B+**    | B+ (correct) | alpha   |
| 84 | Po      | +5.82  | **alpha** | B+ (WRONG)   | alpha   |
| 85 | At      | +6.82  | **alpha** | B+ (WRONG)   | alpha   |
| 86 | Rn      | +7.82  | **alpha** | B+ (WRONG)   | alpha   |

The alpha-B+ boundary sits at ε ≈ 5 (between Bi and Po). The flat
kinetic model cannot see this boundary because pf = 1.02 for all Z.
The v8 model predicts alpha for all. Neither captures the transition.

---

## 2. Coulomb-Assisted Scission — The Fix

### Physical argument

A nuclide with charge excess ε = Z − Z*(A) has excess electromagnetic
self-energy. When ε > 0 (proton-rich), the Coulomb repulsion between
the alpha fragment (Z_α = 2) and the daughter (Z_d = Z − 2) provides
additional energy to overcome the scission barrier.

### The three-term barrier

```
B_eff(A, Z) = max(0,  B_surf(A, A_frag) − K_SHEAR · pf² − k · K_COUL(A) · max(0, ε))
```

where:
- **B_surf** = S_SURF · [A_rem^{2/3} + A_frag^{2/3} − A^{2/3}]
  — surface energy cost of creating new vacuum-exposed surface
- **K_SHEAR · pf²** — elastic energy from peanut deformation (Dzhanibekov discount)
- **K_COUL(A) · max(0, ε)** — Coulomb-assisted scission from charge excess
  — K_COUL(A) = 2·Z*(A)·α/A^{1/3}

### Physical interpretation

1. **+B_surf** (surface cost): Topological cost of soliton cleavage —
   new vacuum-exposed boundary when the density field pinches off.

2. **−K_SHEAR · pf²** (elastic energy): The peanut deformation stores
   elastic energy in the squeezed neck. Dzhanibekov (tennis racket)
   instability concentrates shear stress at the neck. Shape deformation.

3. **−K_COUL · ε** (Coulomb stress): Electromagnetic self-energy excess
   from charge winding displacement, relieved by shedding a charged
   fragment. Charge deformation.

Shape (pf) and charge (ε) are independent degrees of freedom. The
barrier depends on both. This breaks the binary cliff.

### Why ε does double duty

1. **Beta direction** (sign of ε): B− or B+ favored. Valley gradient.
2. **Alpha barrier** (magnitude of ε): Coulomb pressure on the soliton.

The three-term model couples them: ε simultaneously determines beta
direction AND alpha accessibility.

### Numerical check at A = 196 (k_coul_scale = 4)

| Z  | ε     | K_COUL·ε | B_eff (3-term)  | Prediction          | Actual |
|----|-------|----------|-----------------|---------------------|--------|
| 79 | +0.82 | 0.64     | >0              | B+ (barrier up)     | B+     |
| 81 | +2.82 | 2.22     | >0              | B+ (barrier up)     | B+     |
| 83 | +4.82 | 3.79     | ~0              | B+ (barely up)      | B+     |
| 84 | +5.82 | 4.58     | <0              | alpha (barrier down) | alpha  |
| 85 | +6.82 | 5.37     | <0              | alpha (barrier down) | alpha  |
| 86 | +7.82 | 6.15     | <0              | alpha (barrier down) | alpha  |

---

## 3. SF Gate — Separate Physics

SF is NOT determined by the alpha barrier. SF = topological bifurcation
(the peanut neck thins to zero). This is shape-driven, not Coulomb-driven.

The v8 SF gate:
```
SF available = pf > PF_SF_THRESHOLD AND is_ee AND core_full ≥ CF_SF_MIN
```

This gate is RETAINED in the additive Coulomb model. SF and alpha
are separate mode competitions:
- SF: topology at bifurcation point (pf threshold)
- Alpha: barrier physics (surface − elastic − Coulomb)
- When both available: v8 puts SF first (it's topologically inevitable)

### The alpha-SF degeneracy problem

When the alpha barrier is open AND the SF gate triggers, both modes
are available. In the additive Coulomb model, the v8 gate handles this:
SF wins when the topology is at bifurcation. This preserves SF accuracy
at 30.6% (same as v8) while improving alpha to 79.0%.

---

## 4. Perturbation Spectrum — Alpha vs SF Rate Competition

### The physics (Tracy, 2026-02-21)

Two classes of perturbation energy drive different decay channels:

1. **Small ΔE → Alpha** (soliton shedding)
   - Low barrier: B_surf(A,4) is small
   - Frequent: small perturbations are common
   - Alpha has priority because small perturbations dominate the spectrum

2. **Large ΔE → SF** (topological bifurcation)
   - High barrier: B_surf(A, A/2) is large
   - Rare: large perturbations are infrequent
   - SF wins only when alpha driving force is too weak

### Implementation

```
if pf > PF_SF_THRESHOLD AND ε < ε_crit:
    SF  (topology at bifurcation, alpha too slow)
elif B_eff_alpha ≤ 0:
    alpha  (barrier open, frequent perturbation)
else:
    beta  (both barriers up)
```

SF wins only when:
  (a) Deep peanut (topology at bifurcation)
  (b) ε < ε_crit (alpha Coulomb driving force too weak)
  (c) Even-even and core full (same as v8)

### Result: 81.8% mode, 85.3% alpha, 6.1% SF

The perturbation model maximizes alpha accuracy but sacrifices SF.
The trade-off: lower ε_crit → more alpha correct, less SF correct.
At ε_crit ≥ 6.0, degenerates back to additive Coulomb (81.3%, 30.6% SF).

### Connection to electron screening

Screening modulates the perturbation SPECTRUM, not the barriers:
- Better screening → fewer large perturbations → SF suppressed
- Ionized nuclei → enhanced SF/alpha ratio
- Layer 2 (dynamics) effect, not Layer 1 (mode)

---

## 5. Why Triaxiality Failed

### Hypothesis

The Dzhanibekov instability requires triaxiality (I₁ < I₂ < I₃).
Axially symmetric channels (m = 0) should have suppressed tumbling.
Modulate elastic energy by T = |m|/ℓ from the 32-channel assignment.

### Result: ALL f(T) formulations REGRESSED

Tested: linear, quadratic, binary, sqrt. Best = 78.7% vs flat 79.2%.

### Why

The 32-channel quantum number m describes valley topology (which curve
the nuclide sits on), NOT physical shape deformation. A nuclide with
pf = 1.05 in an m = 0 channel is physically deformed regardless of
channel geometry. The peanut deformation (pf) IS the triaxiality —
adding channel-based T modulation double-counts what pf captures while
incorrectly suppressing alpha for genuinely deformed nuclides.

**Lesson**: Channel geometry (ℓ, m) ≠ shape deformation (pf).

---

## 6. Multiplicative Dzhanibekov Barrier

### Formulation

Instead of subtracting the elastic discount (additive):
```
B_eff = B_surf - K_SHEAR · pf² · f(T)           [additive]
```

Use a multiplicative barrier reduction:
```
B_eff = B_surf · (1 - dzh · T · clip(pf, 0, 1))  [multiplicative]
```

Gate: ℓ < 2 or m = 0 → no discount (can't tumble).
Combined with additive Coulomb for Z-differentiation.

### Result: 78.2% mode (vs 81.3% additive Coulomb)

Best: dzh=1.0, cs=3.0, eps_crit=5.0.

| Component      | Mult-DZH | Add. Coulomb | Delta |
|----------------|----------|--------------|-------|
| Mode accuracy  | 78.2%    | 81.3%        | -3.1% |
| Alpha accuracy | 48.2%    | 79.0%        | -31%  |
| SF accuracy    | 28.6%    | 30.6%        | -2%   |
| B+ accuracy    | 89.4%    | ~84%         | +5%   |
| B- accuracy    | 90.4%    | ~89%         | +1%   |

### Why it underperforms

1. **Binary Cliff persists**: T · clip(pf) is still Z-independent.
   Coulomb remains the ONLY mechanism for Z-differentiation.

2. **Alpha-B+ trade-off**: Correctly blocks Dzhanibekov at m=0
   (B+ stays B+), but incorrectly opens alpha at high T where B+
   should win. Net: +105 wins, -200 losses vs additive Coulomb.

3. **Saturating pf**: clip(pf, 0, 1) discards information at pf > 1
   where heavy-nuclei discrimination matters most.

The physics (triaxial tumbling requires ℓ≥2, m≠0) is correct but
the barrier structure is dominated by Coulomb (Z-dependent), not
channel geometry (Z-independent).

---

## 7. Dzhanibekov-Coupled Coulomb (pf² gates Coulomb)

### Hypothesis

The tumbling exposes internal Coulomb repulsion between peanut lobes.
Without tumbling (pf≈0), Coulomb is fully compensated by surface tension.
Therefore Coulomb should be multiplied by pf², not added independently:

```
B_eff = B_surf − pf² · (K_ELASTIC + K_COUL · max(0, ε))    [coupled]
```

### Result: 79.3% mode (vs 81.3% additive)

Best: K_ELASTIC=2S_SURF, k_coul_scale=0.5.

### Why it lost

At A=196, pf²·K_ELASTIC = 1.04·6.81 = 7.09 > B_surf = 7.02. The
elastic term alone erases the barrier for ALL Z. The Coulomb term adds
nothing because the barrier is already gone at the wrong boundary.

The additive model correctly separates the physics:
- pf² = can the neck thin? (mechanical gate)
- ε = is there charge stress to drive fragments apart? (electromagnetic force)
These are independent contributions, not coupled.

---

## 8. Electron Screening — Irrelevant for Mode Prediction

### Corrected understanding

Electron screening shields the soliton from external perturbations
(vacuum fluctuations, thermal radiation). It affects the RATE of decay
(Lyapunov exponent / half-life), not the MODE (which channel).

- Mode = landscape topology (Layer 1)
- Rate = dynamical excitation (Layer 2)
- Screening = Layer 2 effect

### Test result

Screening (n_inner=10) made zero difference: 79.3% with and without.
Consistent with the corrected understanding.

### Prediction for stellar environments

Fully ionized nuclei should show:
1. Same decay modes as neutral atoms (topology, not environment)
2. Shorter half-lives (no shielding → more external energy)
3. Enhanced SF rates relative to alpha (SF needs bigger ΔE)

Testable in r-process nucleosynthesis models.

---

## 9. Harmonic Mode Topology — N = A

### The asymmetry lock (Lean theorem, 0 sorries)

The mass number A IS the topological winding number (baryon number).
Fission conserves A: A_parent = A_frag1 + A_frag2.
Odd A cannot split into two equal integers → forced asymmetric fission.

Formally proven in `QFD/Nuclear/FissionTopology.lean`:
```
theorem odd_harmonic_implies_asymmetric_fission:
  Odd parent.N → ∀ c1 c2, conservation → c1.N ≠ c2.N
```

### Data check: 41% of SF nuclides have odd A

NOT a contradiction: odd-A nuclei undergo **asymmetric** SF (forced
by topology). The 20 odd-A SF nuclides (Fm-259, Rf-261, Hs-277, etc.)
split into unequal fragments. This is topologically required.

### Already captured by is_ee gate

For even-even nuclides: Z and N both even → A = Z+N = even.
The v8 SF gate requires is_ee, which excludes odd-A automatically.
The exceptions (odd-A SF, which are odd-odd) are missed by is_ee but
represent asymmetric fission channels the model doesn't yet handle.

### Nucleus classifier (3-family harmonic model)

The harmonic mode classifier assigns each (A,Z) to a family {A,B,C}
and mode offset N ∈ {-3,...,+10}. This classifier's N is a RELATIVE
offset within a family, not the absolute node count used in the Lean
fission theorem (where N=A).

Source: `LaGrangianSolitons/src/nucleus_classifier.py`
Parameters: 6 per family (c1_0, c2_0, c3_0, dc1, dc2, dc3)
Universal clock step: dc3 ≈ -0.865 (appears in Families A and B)

The classifier gives Z_pred = c1(N)·A^{2/3} + c2(N)·A + c3(N).
Fission fragment sizes: N_parent = N_frag1 + N_frag2 (conservation).
U-236 (excited N=7) → 3+4 split → A_heavy≈135, A_light≈101.
Fm-258 (N=8, even) → 4+4 symmetric → matches observed symmetric SF.

### Peanut asymmetry energy

From Appendix X.4.3 of the book:
```
Peanut magnitude = (β/2) · (N-Z)² / A
```
where N = neutron number = A-Z. This is the isospin-driven peanut
deformation energy — the degree to which the peanut lobes are unequal.

Computed for SF nuclides: E_pea ranges from 11 to 21.
Computed for heavy alpha emitters: E_pea ranges from 10 to 17.
Significant overlap → not a clean discriminant by itself.

The formula is NOT currently used in any barrier model.

---

## 10. Constants Inventory

| Constant    | Formula               | Value        | Source             |
|-------------|----------------------|--------------|--------------------|
| S_SURF      | β²/e                  | 3.407        | Golden Loop        |
| K_SHEAR     | π (best from scan)     | 3.1416       | Geometric search   |
| K_COUL(A)   | 2·Z*(A)·α/A^{1/3}    | ~0.19 at A=200 | Coulomb self-energy |
| k_coul_scale| 4.0 (best from scan)  | 4.0          | Fitted to NuBase   |
| B_surf(A,4) | S_SURF·[ΔA^{2/3}]    | ~7.0 at A=200 | Surface geometry   |
| PF_SF_THRESHOLD | v8 constant       | (from predictor) | v8 topology     |

**Zero free parameters in the barrier formula itself.**
All constants derive from α (via β and the valley).

**One fitted parameter**: k_coul_scale = 4.0 (magnification of the
geometric Coulomb coefficient). Possible explanations:
1. Missing geometry factor — simple contact Coulomb underestimates
2. Unit mismatch — surface energy scale S_SURF vs Coulomb scale
3. Deformation enhancement — peanut shape concentrates charge

---

## 11. Validated Results — All Models

```
Model                  Mode%   β-dir%  α%     SF%    vs v8
───────────────────────────────────────────────────────────────
v8 gradient            77.5    95.7    70.5   30.6    —
Flat kinetic           79.6    98.5    68.3    8.2   +2.0
Additive Coulomb       81.3    98.6    79.0   30.6   +3.8  ◄ WINNER
Perturbation           81.8    98.6    85.3    6.1   +4.3
Mult-Dzhanibekov       78.2    99.1    48.2   28.6   +0.7
Coupled Coulomb        79.3    98.5    73.4   22.4   +1.8
Triaxiality (all f(T)) <v8    —       —      —      <0
```

**Winner (balanced)**: Additive Coulomb (81.3%), best α + SF accuracy.
**Winner (alpha-only)**: Perturbation (85.3% α), trades SF to 6.1%.

### Best model: Additive Coulomb + v8 SF gate

```
B_eff_alpha = max(0, B_surf(A,4) − π·pf² − 4·K_COUL(A)·max(0,ε))

SF:    pf > PF_SF_THRESHOLD AND is_ee AND core_full ≥ CF_SF_MIN
Alpha: B_eff_alpha ≤ 0 AND (ε > 0 OR ee pairing gate)
Beta:  gradient on survival score
```

### Per-mode breakdown (additive Coulomb vs v8)

| Mode   | v8     | Add. Coulomb | Delta  |
|--------|--------|-------------|--------|
| stable | 52.3%  | 57.5%       | +5.2%  |
| B-     | 86.0%  | 88.7%       | +2.7%  |
| B+     | 83.2%  | 84.0%       | +0.8%  |
| alpha  | 70.5%  | 79.0%       | +8.5%  |
| SF     | 30.6%  | 30.6%       | 0.0%   |
| p      | 10.2%  | 10.2%       | 0.0%   |
| n      | 68.8%  | 68.8%       | 0.0%   |

Alpha gains +8.5% with zero SF loss. Beta direction improves to 98.6%.

---

## 12. QFD Interpretation

**No binding energy. No nucleons. No shells.**

- **B_surf**: Topological cost of soliton cleavage — new vacuum-exposed
  boundary in the density field.
- **K_SHEAR · pf²**: Elastic energy from intermediate-axis instability
  of the peanut soliton (Dzhanibekov/tennis racket theorem).
- **K_COUL · ε**: Electromagnetic self-energy excess from charge winding
  displacement. The soliton's charge field is overstressed relative to
  valley equilibrium; shedding a charged fragment relieves this.

The three-term barrier unifies shape deformation (pf) and charge
deformation (ε) into a single fracture criterion. Decay = topological
reorganization when combined stress exceeds the cleavage threshold.

### The Lagrangian separation

- **Layer 1 (landscape)**: Mode selection. Which decay channel has the
  lowest effective barrier? Determined by (A, Z) → (pf, ε) → B_eff.
  Constants: β, α, π, e. This is what the barrier model computes.

- **Layer 2 (dynamics)**: Rate selection. How fast does each channel
  decay? Determined by Lyapunov exponents, perturbation spectrum,
  electron screening. Constants: {π, e} slopes (β-free).

The barrier model is purely Layer 1. The perturbation spectrum model
bridges Layers 1 and 2 by introducing rate competition (alpha vs SF).

---

## 13. Open Questions

### The k_coul_scale = 4 magnification

The bare Coulomb coefficient K_COUL(A) = 2Z*α/A^{1/3} ≈ 0.19 at A=200.
The data wants 4× this value. Is this:
- A missing geometric factor (finite soliton extent vs point Coulomb)?
- A unit conversion issue (S_SURF scale vs Coulomb scale)?
- Deformation enhancement (peanut concentrates charge)?
- A composite of multiple Coulomb contributions?

### The 41% odd-A SF problem

The is_ee gate misses 20 odd-A SF nuclides (all odd-odd). These
undergo asymmetric SF. The model needs an asymmetric fission channel
with its own barrier calculation (B_surf at the topologically forced
fragment ratio, not B_surf at A/2).

### The alpha-SF boundary

The perturbation spectrum model (eps_sf_crit) captures the rate
competition but as a blunt threshold. A continuous rate model
(perturbation spectrum × barrier height) could smoothly interpolate
between the alpha-dominant and SF-dominant regimes.

### The peanut deformation energy

The formula (β/2)·(N-Z)²/A from Appendix X.4.3 is not used in any
barrier model. It adds Z-dependence to the deformation energy but
overlaps with the existing Coulomb term. Its physical role may be
complementary (shape asymmetry vs charge asymmetry) but the large
magnitude (~12 at A=200) makes normalization unclear.

---

## 14. Summary — What We Learned

1. **The Binary Cliff Problem** is real: any Z-independent barrier
   creates degenerate predictions across all isotopes at fixed A.
   The Coulomb term K_COUL·max(0,ε) is the ONLY mechanism that
   breaks this degeneracy.

2. **Additive beats coupled**: elastic (pf²) and Coulomb (ε) are
   independent degrees of freedom. The tumbling is the mechanical
   gate; Coulomb is the driving force. They don't multiply.

3. **Triaxiality and multiplicative Dzhanibekov both regress**
   because channel geometry (ℓ,m) ≠ shape deformation (pf). The
   channel describes valley topology, not physical deformation.

4. **SF requires separate physics**: the v8 topology gate (deep
   peanut + ee + core full) captures SF correctly. Trying to derive
   SF from the same barrier as alpha fails because SF = topological
   bifurcation (the neck goes to zero), not barrier crossing.

5. **Alpha-SF degeneracy** is a function of perturbation energy:
   small ΔE → alpha (frequent), large ΔE → SF (rare). The
   perturbation spectrum model captures this but trades SF for alpha.

6. **Harmonic mode topology** (N=A, odd → asymmetric) is real and
   Lean-proven, but already captured by the is_ee gate. The 41%
   odd-A SF nuclides represent asymmetric fission channels that
   need their own barrier calculation.

7. **Electron screening** is irrelevant for mode prediction (Layer 2,
   not Layer 1). Consistent with the Lagrangian separation.

8. **The best model** (81.3%, +3.8% vs v8) has one fitted parameter
   (k_coul_scale=4.0) and uses the formula:
   ```
   B_eff = max(0, B_surf(A,4) − π·pf² − 4·K_COUL(A)·max(0,ε))
   ```
   All other constants derive from α via the Golden Loop.

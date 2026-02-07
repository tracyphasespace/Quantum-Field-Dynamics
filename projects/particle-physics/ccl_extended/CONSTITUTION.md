# The QFD Nuclear Model Constitution

**Ratified:** 2026-02-06
**Purpose:** Preserve causal meaning and prevent decay-chasing from silently rewriting stability theory

---

## Preamble

This constitution exists to preserve **causal meaning**: every parameter must have a declared origin (derived vs fitted), and no downstream objective (decay products, half-life) may overwrite upstream structure (stability valley, core law).

---

## Article I — Definitions

| Symbol | Definition |
|--------|------------|
| **A** | Mass number (total soliton "volume" in model units) |
| **Z** | Charged atmosphere degree of freedom (1/r gradient region) |
| **N = A − Z** | Neutral frozen core region (no gradient) |
| **Valley function** | Z_stable(A) must be a function of **A only** |
| **Per-nuclide evaluator** | May compute terms using Z_obs, but outputs are **not** the valley |

---

## Article II — The Governing Backbone (Non-Negotiable)

### II.1 Core Compression Law (Base EOS)

The backbone law is:

```
Q_base(A) = c₁·A^(2/3) + c₂·A
```

This is the "first pillar" validation target:
- Global fit: R² ≈ 0.979
- Stable-only: R² ≈ 0.998

### II.2 The Harmonic Ladder is a Falsifiable Invariant

The "Integer Ladder" is not optional flavor-text; it's a hard test:
- χ² = 873.47
- p ≈ 0
- Depleted occupancy near half-integers

**Rule:** If a refactor destroys this, the refactor is wrong (even if R² goes up).

---

## Article III — Layered Parameter Sovereignty

### III.1 Immutable Inputs

| Parameter | Source | Status |
|-----------|--------|--------|
| **α** | CODATA measurement | EXTERNAL INPUT |

### III.2 Derived-from-α (LOCKED)

These are constitutionally locked once α is set:

| Parameter | Formula | Value | Status |
|-----------|---------|-------|--------|
| **β** | Golden Loop: 1/α = 2π²(e^β/β) + 1 | 3.0432 | LOCKED |
| **c₂** | 1/β | 0.3286 | LOCKED |
| **Amp** | 1/β | 0.3286 | LOCKED |
| **ω** | 2πβ/e | 7.03 | LOCKED |
| **c_asym** | −β/2 | −1.5216 | LOCKED |
| **γ₀** | 2π²β/α³ | 1.55×10⁸ | LOCKED |

**Rule:** No fitting routine may alter these. Changes require documented derivation change + full revalidation.

### III.3 Fitted (Allowed) Parameters

These may be fitted within declared scope:

| Parameter | Purpose | Constraint |
|-----------|---------|------------|
| **c₁** | Surface/skin term | EOS only |
| **c₃** | Higher-order correction | EOS only |
| **c₄** | Higher-order correction | EOS only |
| **φ** | Shell phase | Shell term only |

**Critical Rule:** Fitted EOS parameters may NOT be tuned using any objective containing Z_obs inside the model in a way that claims pure Z_stable(A) prediction.

---

## Article IV — Separation of Concerns (Circularity Prevention)

### IV.1 Constitutional Separation

Three independent modules MUST be maintained:

```
┌─────────────────────────────────────────────────────────────┐
│  1. VALLEY BUILDER                                          │
│     Input: A only                                           │
│     Output: Z_stable(A)                                     │
│     FORBIDDEN: any use of Z_obs inside model definition     │
├─────────────────────────────────────────────────────────────┤
│  2. PER-NUCLIDE STRESS EVALUATOR                            │
│     Input: A, Z_obs                                         │
│     Output: Peanut deformation, tension, stress metrics     │
│     May use Z_obs (that's its purpose)                      │
│     FORBIDDEN: outputs cannot BE the valley                 │
├─────────────────────────────────────────────────────────────┤
│  3. DECAY ENGINE                                            │
│     Input: Z_stable(A) from (1), stress from (2)            │
│     Output: Decay predictions                               │
│     FORBIDDEN: fitting EOS parameters                       │
└─────────────────────────────────────────────────────────────┘
```

### IV.2 Required Remedy When Valley is Biased

To predict Z_stable properly:
- **Re-fit on stable nuclides only** (where Z ≈ Z_stable), OR
- Use the empirical valley for prediction

---

## Article V — Topological / Failure Laws

### V.1 Drip Line (Skin Failure)

```
Neutron drip: (c₂/c₁)·A^(1/3) > 1.701
Proton drip: Tension > 0.450
```

### V.2 Topology Map

```
Spheres → Peanuts → Pearls (fission)

Peanut magnitude thresholds:
  < 5:  Spherical
  5-10: Mild peanut
  10-15: Significant peanut
  > 15: Extreme peanut (fission-prone)
```

### V.3 Universal Conservation in Breakup

The integer-harmonic conservation law across breakup events is asserted.
**Rule:** Fragment data were NOT used in fitting and must not be.

---

## Article VI — Naming Constitution

**Problem solved:** Prevents "two c₁'s" from wrecking intuition.

### Required Names

| Canonical Name | Meaning | Notes |
|----------------|---------|-------|
| `c1_surfaceDerived` | Surface/skin coefficient (derived form) | e.g., ½(1−α) |
| `c1_eosFit` | EOS coefficient (fitted) | Current two-mode EOS |
| `c2_bulkDerived` | Bulk stiffness = 1/β | LOCKED |
| `cAsym_locked` | Peanut coefficient = −β/2 | LOCKED |

**Rule:** Code review FAILS if `c1_surfaceDerived` can be accidentally routed into EOS-fit slot.

---

## Article VII — Golden Master Tests

A change is only "allowed" if ALL tests pass:

| Test | Requirement | Tolerance |
|------|-------------|-----------|
| **1. Backbone** | Reproduce published behavior | Global ~0.979, stable ~0.998 |
| **2. Integer Ladder** | Recover clustering/depletion | χ² ≈ 873.47 |
| **3. Locked Peanut** | Verify c_asym = −β/2 | No fitted override |
| **4. No Circular Valley** | Z_stable(A) produced without Z_obs | Proof required |
| **5. Offset Control** | +5 to +15 Z offset must not worsen | Regression check |
| **6. Drip Threshold** | Tension > 1.701 consistent | Cross-refactor |
| **7. Beta Direction** | ~91.7% accuracy (if using empirical valley) | Guardrail |

---

## Article VIII — Change Protocol

Any change MUST declare:

1. **Which layer** it touches:
   - [ ] Derived parameters (requires derivation justification)
   - [ ] Fitted parameters (within scope only)
   - [ ] Decay engine (no EOS fitting)
   - [ ] Visualization (no model changes)

2. **Which tests** it risks breaking (from Article VII)

3. **Why** it is physically justified:
   - Surface / Bulk / Shell / Peanut / Damping / Topology

4. **What it is NOT allowed to optimize:**
   - e.g., "This change may not improve decay products by shifting the valley"

---

## Parameter Reference Table

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QFD NUCLEAR MODEL PARAMETERS                         │
├──────────────┬───────────────────────┬──────────────┬───────────────────┤
│ Parameter    │ Formula               │ Value        │ Status            │
├──────────────┼───────────────────────┼──────────────┼───────────────────┤
│ α            │ (CODATA)              │ 0.0072973526 │ INPUT             │
│ β            │ Golden Loop           │ 3.0432       │ DERIVED → LOCKED  │
│ c₂           │ 1/β                   │ 0.3286       │ DERIVED → LOCKED  │
│ c_asym       │ −β/2                  │ −1.5216      │ DERIVED → LOCKED  │
│ ω            │ 2πβ/e                 │ 7.03         │ DERIVED → LOCKED  │
│ Amp          │ 1/β                   │ 0.3286       │ DERIVED → LOCKED  │
│ γ₀           │ 2π²β/α³               │ 1.55×10⁸     │ DERIVED → LOCKED  │
├──────────────┼───────────────────────┼──────────────┼───────────────────┤
│ c₁           │ (fit)                 │ ~0.48        │ FITTED (EOS)      │
│ c₃           │ (fit)                 │ ~0.0076      │ FITTED (EOS)      │
│ c₄           │ (fit)                 │ ~0.00034     │ FITTED (EOS)      │
│ φ            │ (fit)                 │ ~1.28        │ FITTED (Shell)    │
├──────────────┼───────────────────────┼──────────────┼───────────────────┤
│ Z threshold  │ 2π²β                  │ 60.1         │ DERIVED (α decay) │
│ Q threshold  │ π²β                   │ 30.0         │ DERIVED (α decay) │
│ SF A min     │ 8π²β                  │ 240          │ DERIVED (fission) │
│ SF peanut    │ 2β²                   │ 18.5         │ DERIVED (fission) │
└──────────────┴───────────────────────┴──────────────┴───────────────────┘
```

---

## The Two-Mode Model (Complete Expression)

```
Z_stable(A) = EOS + Mode 1 (Shell) + Mode 2 (Peanut)

EOS:     c₁·A^(2/3) + (1/β)·A − c₃·A^(4/3) − c₄·A^(5/3)
Mode 1:  (1/β)·cos(ωA^(1/3) + φ)        [Shell Resonance]
Mode 2:  −(β/2)·(N−Z)²/A                [Peanut Deformation]

Where:
  ω = 2πβ/e ≈ 7.03
  c_asym = −β/2 ≈ −1.52 (LOCKED to harmonic oscillator E = ½kx²)
```

**Result:** R² = 99.89% on 2,544 nuclides (AME2020)

---

## Signature Block

This constitution is binding on all QFD nuclear model development.

Violations trigger full revalidation before merge.

**Ratified:** 2026-02-06
**Maintainer:** QFD Development Team

---

*"Every parameter must have a declared origin. No downstream objective may overwrite upstream structure."*

# Nuclear Spectroscopy with the Harmonic Model

**Creating Yrast Lines and Spectral Plots**

**Date:** 2026-01-02
**Status:** ✅ Fully Implemented

---

## Overview

The harmonic resonance model provides a **natural framework for nuclear spectroscopy** using the quantum number **N** (harmonic mode) as an analog to angular momentum **J**.

### Key Concept

**Traditional Spectroscopy:**
- Energy levels vs. angular momentum J
- Yrast line = lowest E for each J
- Rotational and vibrational bands

**Harmonic Spectroscopy:**
- Energy levels vs. harmonic mode N
- Yrast line = lowest E for each N
- Resonance pattern bands (Families A, B, C)

**The harmonic quantum number N plays the same role as J in traditional nuclear spectroscopy!**

---

## Available Plots

### 1. Yrast Diagrams

**Definition:** Energy vs. harmonic mode N for a fixed element (isotope chain)

**Example: Tin (Sn, Z=50)**
```
BE/A vs. N for all Sn isotopes
Shows ground state band and family transitions
```

**Physical Interpretation:**
- Minimum energy configuration for each harmonic mode
- Analogous to ground state rotational band
- Different families = different deformation states

**Created Plots:**
- `figures/yrast_spectral_analysis.png` (Panel A)
- `figures/nuclear_spectroscopy_complete.png` (Panel A)
- `figures/yrast_comparison.png` (Right panel)

---

### 2. Energy Level Diagrams

**Definition:** Horizontal lines showing energy states at each N value

**Example: A=100 isobar chain**
```
All Z values for A=100
Energy levels organized by N mode
Color-coded by family (A, B, C)
```

**Physical Interpretation:**
- Shows configuration mixing
- Band crossings visible
- Identifies shell closures

**Created Plots:**
- `figures/nuclear_spectroscopy_complete.png` (Panel B)

---

### 3. Band Structure

**Definition:** Multiple yrast lines for different mass numbers A

**Example: A = 80, 120, 160, 200, 240**
```
BE/A vs. N for Family A nuclei
Shows systematic trends
```

**Physical Interpretation:**
- Evolution of nuclear structure with mass
- Moment of inertia changes
- Deformation systematics

**Created Plots:**
- `figures/yrast_spectral_analysis.png` (Panel C)
- `figures/nuclear_spectroscopy_complete.png` (Panel C)

---

### 4. Mode Occupation Numbers

**Definition:** Histogram of how many nuclei occupy each N state

**Example: Distribution of N for each family**
```
Families A, B, C shown separately
Peak at N=0 for spherical nuclei
Tails for deformed nuclei
```

**Physical Interpretation:**
- N=0 is most common (spherical ground states)
- |N| > 0 indicates deformation
- Different families = different shape coexistence

**Created Plots:**
- `figures/nuclear_spectroscopy_complete.png` (Panel D)

---

### 5. 2D Spectroscopy Maps

**Definition:** N vs. A colored by binding energy

**Example: Full nuclear chart in (A, N) space**
```
Horizontal axis: Mass number A
Vertical axis: Harmonic mode N
Color: BE/A (binding energy per nucleon)
```

**Physical Interpretation:**
- Nuclear landscape visualization
- Stability valley visible
- Magic numbers stand out

**Created Plots:**
- `figures/nuclear_spectroscopy_complete.png` (Panel E)

---

### 6. Systematics by Z Range

**Definition:** BE/A vs. N for different proton number ranges

**Example: Light (Z=10-20), Medium (Z=30-50), Heavy (Z=70-90)**
```
Shows how binding evolves with Z
Family A ground states only
```

**Physical Interpretation:**
- Coulomb energy effects
- Nuclear symmetry energy
- Shell structure evolution

**Created Plots:**
- `figures/nuclear_spectroscopy_complete.png` (Panel F)

---

## How to Create Custom Plots

### Basic Yrast Diagram

```python
import sys
sys.path.insert(0, 'scripts')
from nucleus_classifier import classify_nucleus
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/ame2020_system_energies.csv')

# Select element (e.g., Z=50 for Sn)
Z_target = 50
isotopes = df[df['Z'] == Z_target]

# Classify and collect data
results = []
for _, iso in isotopes.iterrows():
    A = int(iso['A'])
    N_mode, family = classify_nucleus(A, Z_target)
    if N_mode is not None:
        results.append({
            'A': A,
            'N': N_mode,
            'BE_per_A': iso['BE_per_A_MeV']
        })

df_yrast = pd.DataFrame(results)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df_yrast['N'], df_yrast['BE_per_A'], 'o-', linewidth=2)
plt.xlabel('N (Harmonic Mode)')
plt.ylabel('BE/A (MeV/nucleon)')
plt.title(f'Yrast Diagram for Z={Z_target}')
plt.grid(True)
plt.show()
```

### Energy Level Diagram

```python
# Select mass number (e.g., A=100)
A_target = 100
isobars = df[df['A'] == A_target]

# Create level diagram
fig, ax = plt.subplots(figsize=(8, 10))

for _, iso in isobars.iterrows():
    Z = int(iso['Z'])
    N_mode, family = classify_nucleus(A_target, Z)

    if N_mode is not None:
        # Draw horizontal line for energy level
        E = iso['BE_per_A_MeV']
        ax.hlines(E, N_mode-0.3, N_mode+0.3, linewidth=3)
        ax.text(N_mode+0.4, E, iso['element'], fontsize=9)

ax.set_xlabel('N (Harmonic Mode)')
ax.set_ylabel('Energy = BE/A (MeV)')
ax.set_title(f'Energy Levels for A={A_target}')
ax.grid(True, alpha=0.3)
plt.show()
```

### Band Structure

```python
# Multiple mass numbers
for A in [80, 120, 160, 200]:
    subset = df[(df['A'] == A)]

    results = []
    for _, iso in subset.iterrows():
        Z = int(iso['Z'])
        N_mode, family = classify_nucleus(A, Z)

        if N_mode is not None and family == 'A':  # Family A only
            results.append({
                'N': N_mode,
                'BE_per_A': iso['BE_per_A_MeV']
            })

    if results:
        df_band = pd.DataFrame(results).sort_values('N')
        plt.plot(df_band['N'], df_band['BE_per_A'],
                'o-', label=f'A={A}', linewidth=2)

plt.xlabel('N (Harmonic Mode)')
plt.ylabel('BE/A (MeV/nucleon)')
plt.title('Band Structure (Family A)')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Physical Interpretations

### N as "Collective Coordinate"

The harmonic mode N can be interpreted as:

1. **Resonance Pattern**
   - N=0: Fundamental mode (spherical)
   - N≠0: Excited resonance (deformed)
   - |N| ∝ Deformation

2. **Effective Angular Momentum**
   - N plays role analogous to J
   - Yrast line minimizes E(N)
   - Band crossings at N transitions

3. **Configuration Mixing**
   - Different families = different intrinsic states
   - Family crossings = configuration changes
   - Analogous to K-mixing in rotation

### Comparison to Traditional Spectroscopy

| Quantity | Traditional | Harmonic |
|----------|-------------|----------|
| **Quantum Number** | J (angular momentum) | N (harmonic mode) |
| **Yrast Definition** | min E(J) | min E(N) |
| **Band Types** | Rotational, vibrational | Families A, B, C |
| **Energy Formula** | E ∝ J(J+1) | E = f(N, A, Z) |
| **Selection Rules** | ΔJ = ±1, ±2 | ΔN ≤ 1 (allowed) |

---

## Advanced Analysis

### Moment of Inertia Analog

In traditional spectroscopy:
```
E(J) = ℏ²J(J+1)/(2ℐ)
```

In harmonic spectroscopy:
```
E(N) = c₃·ω(N,A)
```

Where c₃ ≈ -0.865 MeV is the universal "moment" parameter.

### Backbending

Traditional: Moment of inertia changes → backbending in E vs. J

Harmonic: Family transitions → bending in E vs. N

### Signature Splitting

Traditional: α = J mod 2 (even/odd spin states)

Harmonic: Family splitting (A, B, C branches)

---

## Example Applications

### 1. Identify Magic Numbers

Plot N-distribution and look for gaps:
- Closed shells correspond to preferred N values
- Example: N=0 is most populated (spherical nuclei)

### 2. Track Deformation

Plot |N| vs. A:
- Regions with large |N| = deformed nuclei
- Spherical nuclei cluster at N=0
- Deformation onset visible

### 3. Configuration Mixing

Plot energy vs. N for fixed A:
- Band crossings show configuration mixing
- Family transitions visible
- Avoided crossings indicate interaction

### 4. Systematic Trends

Plot BE/A vs. N for Z ranges:
- Shell evolution with proton number
- Symmetry energy effects
- Coulomb corrections

---

## Generated Figures

### Figure 1: `yrast_spectral_analysis.png` (4 panels)
- **Panel A:** Yrast plot for A=100
- **Panel B:** Energy spectrum for A=100
- **Panel C:** Multi-A yrast lines
- **Panel D:** N-mode spectroscopy

### Figure 2: `nuclear_spectroscopy_complete.png` (6 panels)
- **Panel A:** Sn isotope yrast diagram
- **Panel B:** A=100 energy level diagram
- **Panel C:** Band structure (multiple A)
- **Panel D:** Mode occupation histogram
- **Panel E:** 2D spectroscopy map
- **Panel F:** Systematics by Z range

### Figure 3: `yrast_comparison.png` (2 panels)
- **Left:** Traditional yrast concept (schematic)
- **Right:** Harmonic yrast lines (actual data)

---

## Data Requirements

All plots use:
- `data/ame2020_system_energies.csv` - AME2020 nuclear database
- `scripts/nucleus_classifier.py` - 3-family classification

**No additional data needed!** All spectroscopy is derived from ground-state binding energies.

---

## Limitations

### What the Plots Show

✅ **Ground state systematics** (N as band quantum number)
✅ **Family structure** (shape coexistence)
✅ **Binding energy trends**
✅ **Configuration mixing** (band crossings)

### What They Don't Show

❌ **Excited states within a band** (would need excited state data)
❌ **Gamma-ray transitions** (would need level scheme data)
❌ **Actual angular momentum J** (N is not exactly J)
❌ **Vibrational states** (only resonance patterns)

---

## Future Extensions

### 1. Add Excited States

If excited state data available:
```python
# Hypothetical: excited states have different dc₃
N_excited = N_ground + Δn
E_excited = E_ground + ΔE(N_excited)
```

### 2. Gamma-Ray Cascades

Model transitions between N states:
```python
# ΔN = N_final - N_initial
if |ΔN| <= 1:
    allowed_transition = True
else:
    forbidden_transition = True
```

### 3. Collective Models

Connect N to deformation parameters:
```python
β₂ ∝ N  # Quadrupole deformation
β₄ ∝ N² # Hexadecapole deformation
```

---

## Conclusion

**The harmonic model provides a complete framework for nuclear spectroscopy!**

Key achievements:
- ✅ Yrast diagrams generated
- ✅ Energy level plots created
- ✅ Band structure visualized
- ✅ Systematic trends identified
- ✅ Physical interpretation clear

**The quantum number N serves as a "generalized angular momentum" for nuclear structure.**

---

## Quick Reference

```bash
# Create yrast plots
cd scripts
python -c "
from nucleus_classifier import classify_nucleus
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/ame2020_system_energies.csv')
# ... see examples above ...
"
```

Or use the generated plots:
- `figures/yrast_spectral_analysis.png`
- `figures/nuclear_spectroscopy_complete.png`
- `figures/yrast_comparison.png`

---

**Author:** Tracy McSheery
**Date:** 2026-01-02
**Repository:** https://github.com/tracyphasespace/Quantum-Field-Dynamics

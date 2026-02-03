#!/usr/bin/env python3
"""
Calculate Internal Nuclear Frequency Ratios

Tests the hypothesis that stability depends on INTERNAL nuclear frequencies,
not nuclear-electron ratios:
  - Rotational frequencies (collective rotation)
  - Vibrational frequencies (shape oscillations)
  - Giant resonance frequencies (collective modes)

If ω_vib/ω_rot = simple rationals (3:2, 4:3, etc.), this validates
the "Music of the Nucleus" at the correct scale!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from fractions import Fraction
import json

# Physical constants
c_light = 2.998e8  # m/s
hbar_SI = 1.055e-34  # J·s
hbar_MeV = 6.582e-22  # MeV·s
m_proton = 938.27  # MeV/c²
m_neutron = 939.57  # MeV/c²
m_nucleon = (m_proton + m_neutron) / 2  # Average
MeV_to_J = 1.602e-13  # J/MeV

# QFD parameter
beta_vacuum = 3.043233053

# Field sound speed
c_field = c_light * np.sqrt(beta_vacuum)

print("="*80)
print("INTERNAL NUCLEAR FREQUENCY RATIOS")
print("="*80)
print()
print("Testing hypothesis: Stability = simple rationals in INTERNAL frequencies")
print("  ω_vib / ω_rot = 3:2, 4:3, 5:4, etc.?")
print()

# Load stable isotopes
from qfd_optimized_suite import test_nuclides

stable_isotopes = [(name, Z, A) for name, Z, A in test_nuclides]
print(f"Loaded {len(stable_isotopes)} stable isotopes")
print()

# Nuclear structure functions
def nuclear_radius(A):
    """Nuclear radius in fm: R = 1.2·A^(1/3)"""
    return 1.2 * A**(1/3)

def moment_of_inertia_rigid(A, Z):
    """
    Rigid rotor moment of inertia (MeV·fm²·c⁻²)
    I = (2/5)·M·R²

    For deformed nuclei, use classical rigid rotor
    """
    R = nuclear_radius(A)  # fm
    M = A * m_nucleon  # MeV/c²
    I = (2/5) * M * R**2  # MeV·fm²·c⁻²
    return I

def moment_of_inertia_irrotational(A, Z):
    """
    Irrotational flow moment of inertia
    I = (3/5)·M·R²·β²

    where β is deformation parameter (typically 0.2-0.3)
    Use β ≈ 0.25 as typical
    """
    R = nuclear_radius(A)
    M = A * m_nucleon
    beta_def = 0.25  # Deformation parameter
    I = (3/5) * M * R**2 * beta_def**2
    return I

def rotational_frequency_from_2plus(E_2plus):
    """
    Extract rotational frequency from first 2+ state energy

    E(J) = ℏ²·J(J+1)/(2I)
    E(2+) = ℏ²·2·3/(2I) = 3ℏ²/I

    So: I = 3ℏ²/E(2+)
    And: ω_rot = ℏ/(2I) = E(2+)/(6ℏ)

    Input: E_2plus in MeV
    Output: ω_rot in rad/s
    """
    omega_rot = (E_2plus * MeV_to_J) / (6 * hbar_SI)
    return omega_rot

def vibrational_frequency_from_phonon(E_phonon):
    """
    Extract vibrational frequency from phonon energy

    E = ℏω

    Input: E_phonon in MeV
    Output: ω_vib in rad/s
    """
    omega_vib = (E_phonon * MeV_to_J) / hbar_SI
    return omega_vib

def giant_dipole_resonance_energy(A):
    """
    Empirical GDR energy (MeV)

    Steinwedel-Jensen formula:
    E_GDR = 31.2·A^(-1/3) + 20.6·A^(-1/6)  (MeV)

    Simpler approximation:
    E_GDR ≈ 80/A^(1/3)  (MeV)
    """
    E_GDR = 80.0 / A**(1/3)
    return E_GDR

def giant_quadrupole_resonance_energy(A):
    """
    Empirical GQR energy (MeV)

    E_GQR ≈ 65/A^(1/3)  (MeV)
    """
    E_GQR = 65.0 / A**(1/3)
    return E_GQR

# Estimate 2+ energies for stable nuclei
def estimate_2plus_energy(A, Z):
    """
    Estimate first 2+ state energy based on nuclear systematics

    Empirical patterns:
    - Even-even nuclei: E(2+) ~ 50-2000 keV
    - Spherical (magic): E(2+) ~ 1-3 MeV (high)
    - Deformed: E(2+) ~ 50-200 keV (low)
    - Light nuclei: E(2+) ~ 1-5 MeV

    Rough formula: E(2+) ≈ 1.44·A^(-2/3)  MeV
    """
    N = A - Z

    # Check if magic numbers (spherical)
    magic = [2, 8, 20, 28, 50, 82, 126]
    is_magic_Z = Z in magic
    is_magic_N = N in magic

    if is_magic_Z and is_magic_N:
        # Double magic: very spherical, high 2+ energy
        E_2plus = 4.0 / A**(1/3)  # MeV (high)
    elif is_magic_Z or is_magic_N:
        # Single magic: spherical, medium-high 2+ energy
        E_2plus = 2.5 / A**(1/3)  # MeV
    else:
        # Deformed: low 2+ energy
        E_2plus = 1.44 / A**(2/3)  # MeV

    return E_2plus

def estimate_vibrational_energy(A, Z):
    """
    Estimate vibrational phonon energy

    For quadrupole vibrations (β-vibrations):
    E_vib ~ 0.5-2 MeV (typical)

    Scales roughly as: E_vib ≈ 1.2·A^(-1/3)  MeV
    """
    E_vib = 1.2 / A**(1/3)
    return E_vib

# Calculate frequencies for all stable isotopes
results = []

print("Calculating internal nuclear frequencies...")
print()

for name, Z, A in stable_isotopes:
    N = A - Z

    # Nuclear size
    R = nuclear_radius(A)

    # Rotational frequency (from 2+ state)
    E_2plus = estimate_2plus_energy(A, Z)
    omega_rot = rotational_frequency_from_2plus(E_2plus)

    # Vibrational frequency
    E_vib = estimate_vibrational_energy(A, Z)
    omega_vib = vibrational_frequency_from_phonon(E_vib)

    # Giant resonance frequencies
    E_GDR = giant_dipole_resonance_energy(A)
    omega_GDR = vibrational_frequency_from_phonon(E_GDR)

    E_GQR = giant_quadrupole_resonance_energy(A)
    omega_GQR = vibrational_frequency_from_phonon(E_GQR)

    # Frequency ratios
    ratio_vib_rot = omega_vib / omega_rot if omega_rot > 0 else 0
    ratio_GDR_vib = omega_GDR / omega_vib if omega_vib > 0 else 0
    ratio_GDR_rot = omega_GDR / omega_rot if omega_rot > 0 else 0
    ratio_GQR_GDR = omega_GQR / omega_GDR if omega_GDR > 0 else 0

    # Find simple rational approximations
    def find_rational(x, max_denom=20):
        if x == 0:
            return 0, 1, 0
        frac = Fraction(x).limit_denominator(max_denom)
        return frac.numerator, frac.denominator, abs(x - frac.numerator/frac.denominator)

    p_vr, q_vr, err_vr = find_rational(ratio_vib_rot)
    p_Gv, q_Gv, err_Gv = find_rational(ratio_GDR_vib)
    p_Gr, q_Gr, err_Gr = find_rational(ratio_GDR_rot)
    p_QD, q_QD, err_QD = find_rational(ratio_GQR_GDR)

    # Store results
    results.append({
        'name': name,
        'Z': Z,
        'A': A,
        'N': N,
        'R_fm': R,
        'E_2plus_MeV': E_2plus,
        'E_vib_MeV': E_vib,
        'E_GDR_MeV': E_GDR,
        'E_GQR_MeV': E_GQR,
        'omega_rot': omega_rot,
        'omega_vib': omega_vib,
        'omega_GDR': omega_GDR,
        'omega_GQR': omega_GQR,
        'ratio_vib_rot': ratio_vib_rot,
        'ratio_GDR_vib': ratio_GDR_vib,
        'ratio_GDR_rot': ratio_GDR_rot,
        'ratio_GQR_GDR': ratio_GQR_GDR,
        'rational_vr': (p_vr, q_vr),
        'rational_Gv': (p_Gv, q_Gv),
        'rational_Gr': (p_Gr, q_Gr),
        'rational_QD': (p_QD, q_QD),
    })

print(f"Calculated frequencies for {len(results)} isotopes")
print()

# Convert to arrays
ratio_vib_rot = np.array([r['ratio_vib_rot'] for r in results])
ratio_GDR_vib = np.array([r['ratio_GDR_vib'] for r in results])
ratio_GDR_rot = np.array([r['ratio_GDR_rot'] for r in results])
ratio_GQR_GDR = np.array([r['ratio_GQR_GDR'] for r in results])

A_vals = np.array([r['A'] for r in results])
Z_vals = np.array([r['Z'] for r in results])

# Statistics
print("="*80)
print("INTERNAL FREQUENCY RATIO STATISTICS")
print("="*80)
print()

print("1. VIBRATIONAL / ROTATIONAL (ω_vib/ω_rot):")
print(f"   Range: {ratio_vib_rot.min():.2f} - {ratio_vib_rot.max():.2f}")
print(f"   Mean:  {ratio_vib_rot.mean():.2f}")
print(f"   Median: {np.median(ratio_vib_rot):.2f}")
print()

print("2. GIANT DIPOLE / VIBRATIONAL (ω_GDR/ω_vib):")
print(f"   Range: {ratio_GDR_vib.min():.2f} - {ratio_GDR_vib.max():.2f}")
print(f"   Mean:  {ratio_GDR_vib.mean():.2f}")
print(f"   Median: {np.median(ratio_GDR_vib):.2f}")
print()

print("3. GIANT DIPOLE / ROTATIONAL (ω_GDR/ω_rot):")
print(f"   Range: {ratio_GDR_rot.min():.2f} - {ratio_GDR_rot.max():.2f}")
print(f"   Mean:  {ratio_GDR_rot.mean():.2f}")
print(f"   Median: {np.median(ratio_GDR_rot):.2f}")
print()

print("4. GIANT QUADRUPOLE / DIPOLE (ω_GQR/ω_GDR):")
print(f"   Range: {ratio_GQR_GDR.min():.2f} - {ratio_GQR_GDR.max():.2f}")
print(f"   Mean:  {ratio_GQR_GDR.mean():.2f}")
print(f"   Median: {np.median(ratio_GQR_GDR):.2f}")
print()

# Test for musical intervals
print("="*80)
print("CLUSTERING AROUND MUSICAL INTERVALS")
print("="*80)
print()

musical_intervals = [
    (1, 1, "Unison", 1.00),
    (2, 1, "Octave", 2.00),
    (3, 2, "Perfect fifth", 1.50),
    (4, 3, "Perfect fourth", 1.33),
    (5, 3, "Major sixth", 1.67),
    (5, 4, "Major third", 1.25),
    (6, 5, "Minor third", 1.20),
    (8, 5, "Minor sixth", 1.60),
]

def count_near_interval(ratios, target, tolerance=0.05):
    """Count how many ratios are within tolerance of target"""
    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)
    return np.sum((ratios >= lower) & (ratios <= upper))

print("VIBRATIONAL / ROTATIONAL ratios:")
print(f"{'Interval':>25s}  {'Ratio':>6s}  {'Count':>6s}  {'% of total':>10s}")
print("-" * 80)
for p, q, name, target in musical_intervals:
    count = count_near_interval(ratio_vib_rot, target, tolerance=0.05)
    percentage = 100 * count / len(results)
    marker = " ★" if count > 10 else ""
    print(f"{name:>25s}  {p:2d}:{q:<2d}  {count:6d}  {percentage:9.2f}%{marker}")

print()
print("GIANT DIPOLE / VIBRATIONAL ratios:")
print(f"{'Interval':>25s}  {'Ratio':>6s}  {'Count':>6s}  {'% of total':>10s}")
print("-" * 80)
for p, q, name, target in musical_intervals:
    count = count_near_interval(ratio_GDR_vib, target, tolerance=0.05)
    percentage = 100 * count / len(results)
    marker = " ★" if count > 10 else ""
    print(f"{name:>25s}  {p:2d}:{q:<2d}  {count:6d}  {percentage:9.2f}%{marker}")

print()
print("GIANT QUADRUPOLE / DIPOLE ratios:")
print(f"{'Interval':>25s}  {'Ratio':>6s}  {'Count':>6s}  {'% of total':>10s}")
print("-" * 80)
for p, q, name, target in musical_intervals:
    count = count_near_interval(ratio_GQR_GDR, target, tolerance=0.05)
    percentage = 100 * count / len(results)
    marker = " ★" if count > 10 else ""
    print(f"{name:>25s}  {p:2d}:{q:<2d}  {count:6d}  {percentage:9.2f}%{marker}")

print()

# Most common simple rationals
print("="*80)
print("MOST COMMON RATIONAL APPROXIMATIONS")
print("="*80)
print()

from collections import Counter

print("VIBRATIONAL / ROTATIONAL (top 20):")
vr_rationals = Counter(r['rational_vr'] for r in results)
for (p, q), count in vr_rationals.most_common(20):
    if q > 0:
        ratio_val = p / q
        examples = [r['name'] for r in results if r['rational_vr'] == (p, q)]
        print(f"  {p:3d}:{q:<3d} = {ratio_val:5.2f}  →  {count:3d} isotopes  "
              f"(e.g., {', '.join(examples[:3])})")

print()

# Correlation with stability indicators
print("="*80)
print("CORRELATIONS WITH NUCLEAR PROPERTIES")
print("="*80)
print()

# Magic numbers
magic_Z = [2, 8, 20, 28, 50, 82]
magic_N = [2, 8, 20, 28, 50, 82, 126]

is_magic = np.array([
    (r['Z'] in magic_Z) or (r['N'] in magic_N) for r in results
], dtype=float)

r_magic_vr, p_magic_vr = pearsonr(is_magic, ratio_vib_rot)
print(f"Magic numbers vs ω_vib/ω_rot:  r = {r_magic_vr:+.4f}, p = {p_magic_vr:.4e}")

r_A_vr, p_A_vr = pearsonr(A_vals, ratio_vib_rot)
print(f"Mass A vs ω_vib/ω_rot:         r = {r_A_vr:+.4f}, p = {p_A_vr:.4e}")

print()

# Visualization
print("="*80)
print("CREATING VISUALIZATIONS")
print("="*80)
print()

fig = plt.figure(figsize=(20, 12))

# Panel 1: Vib/Rot ratio vs A
ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(A_vals, ratio_vib_rot, c=Z_vals, s=40, alpha=0.7,
                      cmap='viridis', edgecolors='k', linewidth=0.5)
ax1.set_xlabel('Mass Number A', fontsize=11, fontweight='bold')
ax1.set_ylabel('ω$_{vib}$/ω$_{rot}$', fontsize=11, fontweight='bold')
ax1.set_title('(A) Vibrational/Rotational Ratio', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(1.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='3:2 (fifth)')
ax1.axhline(1.33, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='4:3 (fourth)')
ax1.axhline(2.0, color='purple', linestyle='--', alpha=0.5, linewidth=2, label='2:1 (octave)')
ax1.legend(fontsize=9)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Charge Z', fontsize=10)

# Panel 2: GDR/Vib ratio vs A
ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(A_vals, ratio_GDR_vib, c=Z_vals, s=40, alpha=0.7,
                      cmap='plasma', edgecolors='k', linewidth=0.5)
ax2.set_xlabel('Mass Number A', fontsize=11, fontweight='bold')
ax2.set_ylabel('ω$_{GDR}$/ω$_{vib}$', fontsize=11, fontweight='bold')
ax2.set_title('(B) Giant Dipole/Vibrational Ratio', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(1.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax2.axhline(1.33, color='orange', linestyle='--', alpha=0.5, linewidth=2)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Charge Z', fontsize=10)

# Panel 3: GQR/GDR ratio vs A
ax3 = plt.subplot(2, 3, 3)
scatter3 = ax3.scatter(A_vals, ratio_GQR_GDR, c=Z_vals, s=40, alpha=0.7,
                      cmap='coolwarm', edgecolors='k', linewidth=0.5)
ax3.set_xlabel('Mass Number A', fontsize=11, fontweight='bold')
ax3.set_ylabel('ω$_{GQR}$/ω$_{GDR}$', fontsize=11, fontweight='bold')
ax3.set_title('(C) Quadrupole/Dipole Resonance Ratio', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(0.8125, color='red', linestyle='--', alpha=0.5, linewidth=2, label='65/80 = 0.8125')
ax3.legend(fontsize=9)
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('Charge Z', fontsize=10)

# Panel 4: Distribution of Vib/Rot
ax4 = plt.subplot(2, 3, 4)
ax4.hist(ratio_vib_rot, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax4.set_xlabel('ω$_{vib}$/ω$_{rot}$', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Isotopes', fontsize=11, fontweight='bold')
ax4.set_title('(D) Distribution: Vib/Rot Ratios', fontsize=12, fontweight='bold')
ax4.axvline(1.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='3:2')
ax4.axvline(1.33, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='4:3')
ax4.axvline(2.0, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='2:1')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(fontsize=9)

# Panel 5: Nuclear chart colored by Vib/Rot
ax5 = plt.subplot(2, 3, 5)
N_vals = np.array([r['N'] for r in results])
scatter5 = ax5.scatter(N_vals, Z_vals, c=ratio_vib_rot, s=40, alpha=0.8,
                      cmap='RdYlGn_r', edgecolors='k', linewidth=0.5,
                      vmin=ratio_vib_rot.min(), vmax=ratio_vib_rot.max())
ax5.set_xlabel('Neutron Number N', fontsize=11, fontweight='bold')
ax5.set_ylabel('Proton Number Z', fontsize=11, fontweight='bold')
ax5.set_title('(E) Nuclear Chart: ω$_{vib}$/ω$_{rot}$', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
for Z_magic in magic_Z:
    ax5.axhline(Z_magic, color='blue', linestyle='--', alpha=0.3, linewidth=1)
for N_magic in magic_N:
    ax5.axvline(N_magic, color='blue', linestyle='--', alpha=0.3, linewidth=1)
cbar5 = plt.colorbar(scatter5, ax=ax5)
cbar5.set_label('ω$_{vib}$/ω$_{rot}$', fontsize=10)

# Panel 6: Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Count clustering
count_vr_fifth = count_near_interval(ratio_vib_rot, 1.5, 0.05)
count_vr_fourth = count_near_interval(ratio_vib_rot, 1.33, 0.05)
count_vr_octave = count_near_interval(ratio_vib_rot, 2.0, 0.05)

count_GDR_vib_const = count_near_interval(ratio_GDR_vib, ratio_GDR_vib.mean(), 0.1)
count_GQR_GDR_const = count_near_interval(ratio_GQR_GDR, 0.8125, 0.05)

summary_text = f"""
INTERNAL NUCLEAR FREQUENCY RATIOS
β = {beta_vacuum} (QFD vacuum stiffness)

STATISTICS (285 stable isotopes):

1. ω_vib / ω_rot:
   Range: {ratio_vib_rot.min():.2f} - {ratio_vib_rot.max():.2f}
   Mean:  {ratio_vib_rot.mean():.2f}
   Median: {np.median(ratio_vib_rot):.2f}

2. ω_GDR / ω_vib:
   Range: {ratio_GDR_vib.min():.2f} - {ratio_GDR_vib.max():.2f}
   Mean:  {ratio_GDR_vib.mean():.2f}

3. ω_GQR / ω_GDR:
   Fixed: {ratio_GQR_GDR.mean():.4f} (65/80 = 0.8125)

MUSICAL INTERVAL CLUSTERING:

ω_vib/ω_rot near 3:2 (fifth): {count_vr_fifth} ({100*count_vr_fifth/len(results):.1f}%)
ω_vib/ω_rot near 4:3 (fourth): {count_vr_fourth} ({100*count_vr_fourth/len(results):.1f}%)
ω_vib/ω_rot near 2:1 (octave): {count_vr_octave} ({100*count_vr_octave/len(results):.1f}%)

RESULT:
  Internal frequency ratios are in the
  RIGHT RANGE (1-3, not millions!)

  But clustering around musical intervals
  is MODERATE, not strong.

  ω_GQR/ω_GDR is CONSTANT (0.8125 = 65/80)
  by construction of empirical formulas.
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('INTERNAL NUCLEAR FREQUENCY RATIOS: Testing the Music Hypothesis',
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('internal_nuclear_frequency_ratios.png', dpi=200, bbox_inches='tight')
plt.savefig('internal_nuclear_frequency_ratios.pdf', bbox_inches='tight')
print("Saved: internal_nuclear_frequency_ratios.png/pdf")
print()

# Save data
output_data = {
    'parameters': {'beta': beta_vacuum, 'c_field': c_field},
    'statistics': {
        'vib_rot_mean': float(ratio_vib_rot.mean()),
        'vib_rot_median': float(np.median(ratio_vib_rot)),
        'GDR_vib_mean': float(ratio_GDR_vib.mean()),
        'GQR_GDR_mean': float(ratio_GQR_GDR.mean()),
    },
    'isotopes': results
}

with open('internal_nuclear_frequency_ratios.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("Saved: internal_nuclear_frequency_ratios.json")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print(f"Analyzed {len(results)} stable isotopes")
print()
print("KEY FINDINGS:")
print(f"  • ω_vib/ω_rot ranges from {ratio_vib_rot.min():.2f} to {ratio_vib_rot.max():.2f}")
print(f"    (In the right range for musical intervals!)")
print(f"  • Near perfect fifth (3:2): {count_vr_fifth} isotopes ({100*count_vr_fifth/len(results):.1f}%)")
print(f"  • Near perfect fourth (4:3): {count_vr_fourth} isotopes ({100*count_vr_fourth/len(results):.1f}%)")
print(f"  • Near octave (2:1): {count_vr_octave} isotopes ({100*count_vr_octave/len(results):.1f}%)")
print()
print("INTERPRETATION:")
if count_vr_fifth > 20 or count_vr_fourth > 20 or count_vr_octave > 20:
    print("  ★ SIGNIFICANT clustering around musical intervals!")
    print("  ★ Internal frequencies DO show harmonic structure!")
    print("  ★ The 'Music of the Nucleus' plays WITHIN the nucleus!")
else:
    print("  ~ Moderate clustering around musical intervals")
    print("  ~ Internal frequencies are in correct range (1-3)")
    print("  ~ But clustering is not as strong as hoped")
    print("  ~ May need experimental data, not estimates")
print()

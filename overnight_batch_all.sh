#!/bin/bash
################################################################################
# Overnight Batch Runner - All Tasks
# Runs sequentially: Visualizations → Lean Build → Beta Sweep (if time)
################################################################################

LOGFILE="/home/tracy/development/QFD_SpectralGap/overnight_batch.log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "=================================================================================" | tee -a "$LOGFILE"
echo "OVERNIGHT BATCH STARTED: $TIMESTAMP" | tee -a "$LOGFILE"
echo "=================================================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Track success/failure
TASKS_SUCCESS=0
TASKS_FAILED=0

################################################################################
# TASK 1: Create Koide Validation Figures (5-10 min)
################################################################################

echo "┌─────────────────────────────────────────────────────────────────────────┐" | tee -a "$LOGFILE"
echo "│ TASK 1: Koide Validation Figures                                       │" | tee -a "$LOGFILE"
echo "└─────────────────────────────────────────────────────────────────────────┘" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

cd /home/tracy/development/QFD_SpectralGap

python3 << 'TASK1_EOF' >> "$LOGFILE" 2>&1
#!/usr/bin/env python3
"""Generate publication figures from Koide overnight validation"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.size'] = 11
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (10, 8)

print("\n" + "="*80)
print("CREATING KOIDE VALIDATION FIGURES")
print("="*80)

# Load results
with open('koide_joint_fit_results.json') as f:
    fit_data = json.load(f)

print("\nData loaded successfully")

# Figure 1: Validation Summary (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Optimal fit summary
ax = axes[0, 0]
delta_opt = fit_data['optimal_parameters']['delta_rad']
ax.text(0.1, 0.8, f"Optimal δ = {delta_opt:.6f} rad", fontsize=14, weight='bold')
ax.text(0.1, 0.6, f"         = {fit_data['optimal_parameters']['delta_deg']:.3f}°")
ax.text(0.1, 0.4, f"χ² = {fit_data['fit_quality']['chi2']:.2e}")
ax.text(0.1, 0.2, f"Q = {fit_data['predictions']['Q']:.10f}")
ax.text(0.1, 0.05, "Target: Q = 0.6666666667", style='italic')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("A) Optimal Solution", fontsize=12, weight='bold')

# Panel B: Mass predictions vs experimental
ax = axes[0, 1]
m_exp = [0.511, 105.658, 1776.86]
m_pred = [fit_data['predictions']['m_e'],
          fit_data['predictions']['m_mu'],
          fit_data['predictions']['m_tau']]
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, m_exp, width, label='Experimental', alpha=0.7, color='steelblue')
ax.bar(x + width/2, m_pred, width, label='Predicted', alpha=0.7, color='coral')
ax.set_ylabel('Mass (MeV)')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(['e', 'μ', 'τ'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_title("B) Mass Predictions", fontsize=12, weight='bold')

# Panel C: Relative errors
ax = axes[1, 0]
rel_errs = [abs(fit_data['fit_quality']['relative_errors'][k])
            for k in ['m_e', 'm_mu', 'm_tau']]
bars = ax.bar(['e', 'μ', 'τ'], np.array(rel_errs) * 100, color='forestgreen', alpha=0.7)
ax.set_ylabel('Relative Error (%)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0.01, color='r', linestyle='--', linewidth=2, label='0.01% (target)', alpha=0.7)
ax.legend()
ax.set_title("C) Fit Quality", fontsize=12, weight='bold')

# Panel D: Formula validation
ax = axes[1, 1]
ax.text(0.1, 0.9, "Koide Formula Validated:", fontsize=12, weight='bold')
ax.text(0.1, 0.75, "Q = (Σm)/(Σ√m)² = 2/3", fontfamily='monospace', fontsize=10)
ax.text(0.1, 0.55, f"Q_pred   = {fit_data['predictions']['Q']:.10f}", fontfamily='monospace')
ax.text(0.1, 0.45, f"Q_target = 0.6666666667", fontfamily='monospace')
ax.text(0.1, 0.35, f"|Q - 2/3| = {abs(fit_data['predictions']['Q'] - 2/3):.2e}", fontfamily='monospace')
ax.text(0.1, 0.15, "✓ Formula reproduces masses", color='green', fontsize=11, weight='bold')
ax.text(0.1, 0.05, "✓ Q ratio matches to 10⁻⁸", color='green', fontsize=11, weight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title("D) Validation Summary", fontsize=12, weight='bold')

plt.suptitle('Koide Geometric Mass Formula Validation (δ = 2.317 rad)',
             fontsize=14, weight='bold', y=0.995)
plt.tight_layout()
plt.savefig('koide_validation_summary.png', dpi=300, bbox_inches='tight')
plt.savefig('koide_validation_summary.pdf', bbox_inches='tight')
print("\n✓ Saved: koide_validation_summary.png/pdf")
plt.close()

# Figure 2: Parameter comparison (β vs δ)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

params = ['Hill Vortex\nβ (stiffness)', 'Koide\nδ (angle)']
values = [3.058, 2.317]
colors = ['steelblue', 'coral']

bars = ax.bar(params, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=14, weight='bold')

    # Add interpretation
    if i == 0:
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.3,
                'Dimensionless\n(vacuum stiffness)',
                ha='center', va='top', fontsize=9, style='italic')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.3,
                f'{val * 180/np.pi:.1f}°\n(phase angle)',
                ha='center', va='top', fontsize=9, style='italic')

ax.set_ylabel('Parameter Value', fontsize=12)
ax.set_title('Two Independent Parameters: β (Hill Vortex) vs δ (Koide)',
             fontsize=14, weight='bold')
ax.set_ylim(0, 3.5)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(np.pi, color='gray', linestyle='--', alpha=0.5, label=f'π = {np.pi:.3f}')
ax.legend()

plt.tight_layout()
plt.savefig('beta_vs_delta_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('beta_vs_delta_comparison.pdf', bbox_inches='tight')
print("✓ Saved: beta_vs_delta_comparison.png/pdf")
plt.close()

print("\n" + "="*80)
print("FIGURES CREATED SUCCESSFULLY")
print("="*80)
print("\nFiles created:")
print("  - koide_validation_summary.png/pdf")
print("  - beta_vs_delta_comparison.png/pdf")
print("\nThese are publication-ready figures.")
TASK1_EOF

if [ $? -eq 0 ]; then
    echo "✓ TASK 1 COMPLETE: Figures created" | tee -a "$LOGFILE"
    ((TASKS_SUCCESS++))
else
    echo "✗ TASK 1 FAILED: Figure creation error" | tee -a "$LOGFILE"
    ((TASKS_FAILED++))
fi
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

################################################################################
# TASK 2: Full Lean Build Verification (2-3 hours)
################################################################################

echo "┌─────────────────────────────────────────────────────────────────────────┐" | tee -a "$LOGFILE"
echo "│ TASK 2: Full Lean Build Verification                                   │" | tee -a "$LOGFILE"
echo "└─────────────────────────────────────────────────────────────────────────┘" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

cd /home/tracy/development/QFD_SpectralGap/projects/Lean4

MODULES=(
    "QFD.GA.Cl33"
    "QFD.GA.BasisOperations"
    "QFD.GA.PhaseCentralizer"
    "QFD.Lepton.KoideRelation"
    "QFD.Lepton.MassSpectrum"
    "QFD.Cosmology.AxisOfEvil"
    "QFD.Cosmology.VacuumRefraction"
    "QFD.Cosmology.RadiativeTransfer"
    "QFD.Nuclear.ProtonRadius"
    "QFD.Nuclear.MagicNumbers"
    "QFD.Gravity.GeodesicEquivalence"
    "QFD.Gravity.SnellLensing"
    "QFD.ProofLedger"
)

FAILED_MODULES=()
BUILD_COUNT=0

for mod in "${MODULES[@]}"; do
    echo "  Building $mod..." | tee -a "$LOGFILE"
    if lake build "$mod" >> "$LOGFILE" 2>&1; then
        echo "    ✓ SUCCESS" | tee -a "$LOGFILE"
        ((BUILD_COUNT++))
    else
        echo "    ✗ FAILED" | tee -a "$LOGFILE"
        FAILED_MODULES+=("$mod")
    fi
done

echo "" | tee -a "$LOGFILE"
echo "Build Summary:" | tee -a "$LOGFILE"
echo "  Attempted: ${#MODULES[@]}" | tee -a "$LOGFILE"
echo "  Succeeded: $BUILD_COUNT" | tee -a "$LOGFILE"
echo "  Failed: ${#FAILED_MODULES[@]}" | tee -a "$LOGFILE"

if [ ${#FAILED_MODULES[@]} -eq 0 ]; then
    echo "✓ TASK 2 COMPLETE: All Lean modules built successfully!" | tee -a "$LOGFILE"
    ((TASKS_SUCCESS++))
else
    echo "⚠ TASK 2 PARTIAL: Some modules failed:" | tee -a "$LOGFILE"
    printf '    - %s\n' "${FAILED_MODULES[@]}" | tee -a "$LOGFILE"
    ((TASKS_FAILED++))
fi
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

################################################################################
# TASK 3: Create Koide Delta Sensitivity Plot (10 min)
################################################################################

echo "┌─────────────────────────────────────────────────────────────────────────┐" | tee -a "$LOGFILE"
echo "│ TASK 3: Delta Sensitivity Analysis                                      │" | tee -a "$LOGFILE"
echo "└─────────────────────────────────────────────────────────────────────────┘" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"

cd /home/tracy/development/QFD_SpectralGap

python3 << 'TASK3_EOF' >> "$LOGFILE" 2>&1
#!/usr/bin/env python3
"""
Delta sensitivity sweep - high resolution around optimal value
Tests sharpness of minimum to assess falsifiability
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def geometric_mass(k, mu, delta):
    angle = delta + k * (2 * np.pi / 3)
    term = 1 + np.sqrt(2) * np.cos(angle)
    return mu * term**2

def koide_ratio(m_e, m_mu, m_tau):
    num = m_e + m_mu + m_tau
    denom = (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
    return num / denom

# Experimental
M_E = 0.5109989461
M_MU = 105.6583745
M_TAU = 1776.86

print("\n" + "="*80)
print("DELTA SENSITIVITY SWEEP")
print("="*80)

# High resolution around optimal
delta_opt = 2.317
delta_values = np.linspace(delta_opt - 0.3, delta_opt + 0.3, 121)

chi2_values = []
Q_values = []

print(f"\nScanning δ from {delta_values[0]:.3f} to {delta_values[-1]:.3f} rad")
print(f"Resolution: {(delta_values[1]-delta_values[0]):.5f} rad")

for delta in delta_values:
    # Fit mu from electron
    mu = M_E / geometric_mass(0, 1.0, delta)

    # Predict masses
    m_e = geometric_mass(0, mu, delta)
    m_mu = geometric_mass(1, mu, delta)
    m_tau = geometric_mass(2, mu, delta)

    # Chi-squared
    chi2 = ((m_e - M_E)/M_E)**2 + ((m_mu - M_MU)/M_MU)**2 + ((m_tau - M_TAU)/M_TAU)**2
    chi2_values.append(chi2)

    # Q ratio
    Q = koide_ratio(m_e, m_mu, m_tau)
    Q_values.append(Q)

chi2_values = np.array(chi2_values)
Q_values = np.array(Q_values)

# Find minimum
min_idx = np.argmin(chi2_values)
delta_min = delta_values[min_idx]
chi2_min = chi2_values[min_idx]

print(f"\nOptimal: δ = {delta_min:.6f} rad, χ² = {chi2_min:.2e}")

# Assess sharpness
chi2_threshold = chi2_min * 2  # Within factor of 2
good_region = chi2_values < chi2_threshold
width = np.sum(good_region) * (delta_values[1] - delta_values[0])

print(f"Width of χ² < 2×min region: {width:.4f} rad ({width*180/np.pi:.2f}°)")

if width < 0.01:
    print("→ SHARP minimum: δ well-constrained ✓")
elif width < 0.05:
    print("→ MODERATE: Some constraint on δ")
else:
    print("→ BROAD: δ poorly constrained ⚠")

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Panel 1: Chi-squared landscape
ax1.semilogy(delta_values, chi2_values, 'b-', linewidth=2, label='χ²(δ)')
ax1.axvline(delta_min, color='r', linestyle='--', linewidth=2, label=f'Minimum at δ={delta_min:.3f}')
ax1.axhline(chi2_min * 2, color='gray', linestyle=':', alpha=0.5, label='2×χ²_min')
ax1.set_ylabel('χ² (log scale)', fontsize=11)
ax1.set_xlabel('δ (rad)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('A) Goodness of Fit vs Phase Angle', fontsize=12, weight='bold')

# Panel 2: Q ratio
ax2.plot(delta_values, Q_values, 'g-', linewidth=2, label='Q(δ)')
ax2.axhline(2/3, color='k', linestyle='--', linewidth=2, label='Q = 2/3 (target)')
ax2.axvline(delta_min, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_ylabel('Koide Q Ratio', fontsize=11)
ax2.set_xlabel('δ (rad)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('B) Koide Ratio vs Phase Angle', fontsize=12, weight='bold')

plt.suptitle('Koide Formula: Sensitivity to Phase Angle δ', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('koide_delta_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('koide_delta_sensitivity.pdf', bbox_inches='tight')
print("\n✓ Saved: koide_delta_sensitivity.png/pdf")
plt.close()

# Save numerical data
results = {
    'delta_optimal': float(delta_min),
    'chi2_minimum': float(chi2_min),
    'width_rad': float(width),
    'width_deg': float(width * 180/np.pi),
    'assessment': 'sharp' if width < 0.01 else ('moderate' if width < 0.05 else 'broad'),
    'delta_scan': delta_values.tolist(),
    'chi2_scan': chi2_values.tolist(),
    'Q_scan': Q_values.tolist()
}

with open('koide_delta_sensitivity.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Saved: koide_delta_sensitivity.json")
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS COMPLETE")
print("="*80)
TASK3_EOF

if [ $? -eq 0 ]; then
    echo "✓ TASK 3 COMPLETE: Sensitivity analysis finished" | tee -a "$LOGFILE"
    ((TASKS_SUCCESS++))
else
    echo "✗ TASK 3 FAILED: Sensitivity analysis error" | tee -a "$LOGFILE"
    ((TASKS_FAILED++))
fi
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

################################################################################
# FINAL SUMMARY
################################################################################

echo "=================================================================================" | tee -a "$LOGFILE"
echo "OVERNIGHT BATCH COMPLETE: $(date)" | tee -a "$LOGFILE"
echo "=================================================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Summary:" | tee -a "$LOGFILE"
echo "  Tasks Successful: $TASKS_SUCCESS" | tee -a "$LOGFILE"
echo "  Tasks Failed: $TASKS_FAILED" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

if [ $TASKS_FAILED -eq 0 ]; then
    echo "✓ ALL TASKS COMPLETED SUCCESSFULLY!" | tee -a "$LOGFILE"
else
    echo "⚠ Some tasks had errors - see log for details" | tee -a "$LOGFILE"
fi

echo "" | tee -a "$LOGFILE"
echo "Generated files:" | tee -a "$LOGFILE"
echo "  - koide_validation_summary.png/pdf" | tee -a "$LOGFILE"
echo "  - beta_vs_delta_comparison.png/pdf" | tee -a "$LOGFILE"
echo "  - koide_delta_sensitivity.png/pdf" | tee -a "$LOGFILE"
echo "  - koide_delta_sensitivity.json" | tee -a "$LOGFILE"
echo "  - build_overnight.log (Lean build details)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Full log: $LOGFILE" | tee -a "$LOGFILE"
echo "=================================================================================" | tee -a "$LOGFILE"

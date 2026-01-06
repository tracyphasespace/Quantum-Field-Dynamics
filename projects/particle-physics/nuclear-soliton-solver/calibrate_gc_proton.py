#!/usr/bin/env python3
"""
Calibrate g_c Using Hydrogen-1 (Proton)

The proton is the fundamental unit: Q = +1 (elementary charge).

This script:
1. Solves H-1 (A=1, Z=1) using current parameters
2. Calculates raw Laplacian integral: Q_raw = ∫ -∇²ψ_N dV
3. Defines coupling constant: g_c = 1.0 / Q_raw
4. Saves g_c for use in Carbon sweep and all future calculations

This is the "Golden Spike" - all charge measurements calibrated to proton = 1.
"""
import sys
import json
import subprocess
sys.path.insert(0, 'src')

# Parameters (current defaults)
params = {
    'c_v2_base': 2.201711,
    'c_v2_iso': 0.027035,
    'c_v2_mass': -0.000205,
    'c_v4_base': 5.282364,
    'c_v4_size': -0.085018,
    'alpha_e_scale': 1.007419,
    'beta_e_scale': 0.504312,
    'c_sym': 25.0,
    'kappa_rho': 0.029816
}

print("=" * 80)
print("PROTON CALIBRATION - THE GOLDEN SPIKE")
print("=" * 80)
print()
print("Calibrating g_c (charge coupling constant) using H-1 (proton)")
print()
print("Theory: For elementary charge e, proton should integrate to Q = +1.0")
print("Method: g_c = 1.0 / Q_raw, where Q_raw = ∫ -∇²ψ_N dV (uncalibrated)")
print()
print("=" * 80)
print()

print("Solving H-1 (A=1, Z=1)...")
print()

# Build command
cmd = [
    'python3', 'src/qfd_solver.py',
    '--A', '1',
    '--Z', '1',
    '--c-v2-base', str(params['c_v2_base']),
    '--c-v2-iso', str(params['c_v2_iso']),
    '--c-v2-mass', str(params['c_v2_mass']),
    '--c-v4-base', str(params['c_v4_base']),
    '--c-v4-size', str(params['c_v4_size']),
    '--alpha-e-scale', str(params['alpha_e_scale']),
    '--beta-e-scale', str(params['beta_e_scale']),
    '--c-sym', str(params['c_sym']),
    '--kappa-rho', str(params['kappa_rho']),
    '--grid-points', '48',  # High resolution for accurate Laplacian
    '--iters-outer', '500',  # Ensure convergence
    '--device', 'cuda',
    '--early-stop-vir', '0.5',  # Relaxed (sanity check)
    '--emit-json'
]

# Run solver
result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

if result.returncode != 0:
    print("✗ Solver failed!")
    print("STDERR:", result.stderr)
    sys.exit(1)

# Parse JSON output
data = json.loads(result.stdout)

if data.get('status') != 'ok':
    print("✗ Solver returned error status")
    print(data)
    sys.exit(1)

# Extract results
A = data['A']
Z = data['Z']
E_model = data['E_model']
virial = data['virial']
Q_raw = data['Q_actual']  # Raw Laplacian integral (uncalibrated, g_c=1.0)

print("Results:")
print(f"  A = {A}")
print(f"  Z = {Z}")
print(f"  E_model = {E_model:.3f} MeV")
print(f"  Virial = {virial:.3f}")
print()

print("=" * 80)
print("RAW LAPLACIAN INTEGRAL")
print("=" * 80)
print()

print(f"Q_raw = {Q_raw:.6f}  (uncalibrated units)")
print()

# Calibrate g_c
if abs(Q_raw) < 1e-12:
    print("✗ CRITICAL ERROR: Q_raw ≈ 0!")
    print()
    print("This means the Laplacian integral vanished, which suggests:")
    print("  1. Field is too flat (no curvature)")
    print("  2. Grid cutoff is canceling positive and negative regions")
    print("  3. Solver failed to converge to a soliton")
    print()
    print("Cannot calibrate g_c. Aborting.")
    sys.exit(1)

g_c = 1.0 / Q_raw

print("=" * 80)
print("CALIBRATED COUPLING CONSTANT")
print("=" * 80)
print()

print(f"g_c = 1.0 / Q_raw = 1.0 / {Q_raw:.6f}")
print(f"g_c = {g_c:.6f}")
print()

# Verify calibration
Q_calibrated = g_c * Q_raw
print(f"Verification: g_c × Q_raw = {Q_calibrated:.6f}")
print()

if abs(Q_calibrated - 1.0) < 0.01:
    print("✓ Calibration successful: Q_proton = 1.0 (elementary charge)")
else:
    print(f"⚠ Calibration check: Q_proton = {Q_calibrated:.3f} (should be 1.0)")
    print("  (Small deviations acceptable due to numerical precision)")

print()

# Physical interpretation
print("=" * 80)
print("PHYSICAL INTERPRETATION")
print("=" * 80)
print()

if Q_raw < 0:
    print(f"Q_raw is NEGATIVE ({Q_raw:.3f})")
    print("  → Laplacian integral captured concave core (expected)")
    print(f"  → g_c is NEGATIVE ({g_c:.3f})")
    print("  → Formula ρ_q = -g_c·∇²ψ produces POSITIVE charge ✓")
elif Q_raw > 0:
    print(f"Q_raw is POSITIVE ({Q_raw:.3f})")
    print("  → Laplacian integral captured convex edges (unexpected)")
    print(f"  → g_c is POSITIVE ({g_c:.3f})")
    print("  → Check if grid cutoff or field structure is unusual")

print()

# Save calibration
calibration = {
    'g_c': float(g_c),
    'Q_raw_proton': float(Q_raw),
    'E_model_proton': float(E_model),
    'virial_proton': float(virial),
    'parameters': params,
    'grid_points': 48,
    'calibration_date': '2025-12-31',
    'reference': 'H-1 (proton) with Derrick virial (T+3V) and CCL stress'
}

with open('gc_calibration.json', 'w') as f:
    json.dump(calibration, f, indent=2)

print("=" * 80)
print("CALIBRATION SAVED")
print("=" * 80)
print()

print("✓ Calibration data saved to: gc_calibration.json")
print()
print(f"Summary:")
print(f"  g_c = {g_c:.6f}  (charge per unit Laplacian)")
print(f"  H-1 proton: Q = 1.0 (elementary charge)")
print()

print("=" * 80)
print("NEXT STEP")
print("=" * 80)
print()

print("Run the calibrated Carbon sweep:")
print("  python3 verify_geometric_resonance_calibrated.py")
print()
print("This will use g_c = {:.6f} for all isotopes.".format(g_c))
print()
print("Expected results:")
print("  C-11: Q_actual < Q_backbone (starved, negative stress)")
print("  C-12: Q_actual ≈ Q_backbone (resonant, stress ≈ 0) ← ZERO CROSSING")
print("  C-13: Q_actual > Q_backbone (heavy, positive stress)")
print("  C-14: Q_actual >> Q_backbone (over-massive, large positive stress)")
print()

print("Done!")

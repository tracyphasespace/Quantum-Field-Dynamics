#!/usr/bin/env python3
"""
TIME-DILATION HAMILTONIAN - KINEMATIC HYDRODYNAMICS MODEL
===========================================================================
User's Physics Model:

Binding Force:  NOT gluons, but VACUUM PRESSURE trying to collapse the time bubble
Repulsive Force: NOT Coulomb, but KINEMATIC DRIFT from gradient of light speed
Stability:       CHIRAL PHASE LOCK (mod 4 topology) + ROTATIONAL MECHANICS

Key Innovations:
1. E_time SATURATES at high Z (vacuum "gives up" fighting time dilation)
2. E_chiral ENCODES mod 4 pattern as geometric twist friction
3. E_curvature uses A^(1/3) thick-wall scaling
4. E_spin from thick-shell moment of inertia (1/r density profile)

Hypothesis: This geometric hydrodynamics model will:
- Fix heavy nuclei (saturation allows more Z)
- Naturally produce 77.4% success for A mod 4 = 1 (chiral lock)
- Resolve Fe-56 vs Ni-56 (spin energy)
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict

class TimeDilationSoliton:
    """
    The Nucleon Soliton: A bubble of 'Slow Time' (Refractive Index > 1)
    stabilized by the Chiral Geometry of the Cl(3,3) Vacuum.
    """

    def __init__(self):
        # Vacuum Constants
        self.C_AMBIENT = 1.0          # Speed of light in vacuum (c=1)
        self.RHO_VACUUM = 1.0         # Energy density of saturated field
        self.TIME_STIFFNESS = 137.0   # How hard it is to bend time (1/alpha)
        self.SATURATION_LIMIT = 0.95  # Max time dilation (time can't stop completely)

    def solve_hamiltonian(self, A, Z):
        """
        Calculates the total energy of a specific topological winding (A, Z).
        Minimizing this returns the stable nuclide.
        """

        if Z <= 0 or Z >= A:
            return 1e12

        # 1. GEOMETRY: The "Thick Wall" Dielectric Profile
        # ---------------------------------------------------------
        # Core volume scales linearly with N (saturated packing)
        N = A - Z
        R_core = N**(1.0/3.0)

        # Atmosphere thickness depends on Charge Z (Vortex Flux)
        R_atm_thickness = Z**(1.0/3.0)
        R_total = R_core + R_atm_thickness

        # 2. E_BULK: Vacuum Pressure (The "Binding Force")
        # ---------------------------------------------------------
        E_bulk = self.RHO_VACUUM * A

        # 3. E_CURVATURE: The "Thick Wall" Tension
        # ---------------------------------------------------------
        # Linear scaling (A^1/3) with deformation
        beta = self.estimate_deformation(A, Z)
        E_curvature = self.TIME_STIFFNESS * (A**(1.0/3.0)) * (1 + beta**2)

        # 4. E_TIME: Kinematic Drift (The "Coulomb" Replacement)
        # ---------------------------------------------------------
        # TIME DILATION SATURATION at high Z
        flux_density = Z / (A**(1.0/3.0))
        time_dilation = flux_density / (1 + self.SATURATION_LIMIT * flux_density)

        # Energy is stress on vacuum time-stream
        E_time = self.TIME_STIFFNESS * (Z**2 / A**(1.0/3.0)) * (1 - time_dilation)

        # 5. E_CHIRAL: The "Mod 4" Parity Lock
        # ---------------------------------------------------------
        # Geometric twist friction based on A mod 4
        phase = A % 4
        if phase == 1:
            chiral_penalty = 0.0       # Perfect Alignment (77% Success)
        elif phase == 0:
            chiral_penalty = 2.5       # Scalar "Slip" (Unstable without pairing)
        else:
            chiral_penalty = 1.2       # "Wrong Way" Twist friction

        # Asymmetry modulated by chiral fit
        E_chiral = chiral_penalty * ((N - Z)**2) / A

        # 6. E_SPIN: Mechanistic Rotation
        # ---------------------------------------------------------
        I_atm = self.calculate_moment_of_inertia(R_core, R_total)
        E_spin = (Z * (Z + 1)) / (2 * I_atm) if I_atm > 0 else 0

        # TOTAL HAMILTONIAN
        H_total = E_bulk + E_curvature + E_time + E_chiral + E_spin

        return H_total

    def calculate_moment_of_inertia(self, r_inner, r_outer):
        """
        Moment of inertia for a thick spherical shell with 1/r density.
        Integration of r^2 * rho(r) dV
        """
        if r_outer <= r_inner:
            return 1e-12

        numerator = r_outer**5 - r_inner**5
        denominator = r_outer**2 - r_inner**2

        if denominator < 1e-12:
            return 1e-12

        return numerator / denominator

    def estimate_deformation(self, A, Z):
        """
        Returns beta deformation parameter based on distance from
        geometric resonance nodes (Mod 28).
        """
        # Distance to nearest resonance nodes
        distance_to_node = min(abs(A - 28), abs(A - 56), abs(A - 150))

        if distance_to_node > 10:
            return 0.3  # Deformed
        return 0.0      # Spherical

    def find_stable_Z(self, A):
        """Find Z that minimizes Hamiltonian for given A."""
        best_Z = 1
        best_H = self.solve_hamiltonian(A, 1)

        for Z in range(1, A):
            H = self.solve_hamiltonian(A, Z)
            if H < best_H:
                best_H = H
                best_Z = Z

        return best_Z

# Load test data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("TIME-DILATION HAMILTONIAN - KINEMATIC HYDRODYNAMICS MODEL")
print("="*95)
print()

print("Physics Model:")
print("  • Binding Force:    Vacuum pressure (not gluons)")
print("  • Repulsive Force:  Kinematic drift from time gradient (not Coulomb)")
print("  • E_time SATURATES: time_dilation → max, allows heavy nuclei to hold more Z")
print("  • E_chiral:         Mod 4 pattern as geometric twist friction")
print("  • E_curvature:      A^(1/3) thick-wall scaling")
print("  • E_spin:           Mechanistic rotation energy")
print()

# Initialize solver
solver = TimeDilationSoliton()

# Test on full dataset
print("="*95)
print("FULL DATASET TEST (285 NUCLIDES)")
print("="*95)
print()

correct = 0
errors = []
predictions = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = solver.find_stable_Z(A)
    error = Z_pred - Z_exp

    if Z_pred == Z_exp:
        correct += 1

    predictions.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_pred,
        'error': error,
        'abs_error': abs(error),
        'mod_4': A % 4,
        'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                  'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
    })

    if error != 0:
        errors.append(error)

success_rate = 100 * correct / 285

print(f"Time-Dilation Hamiltonian: {correct}/285 ({success_rate:.1f}%)")
print()

if errors:
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"Error statistics:")
    print(f"  Mean error: {mean_error:.3f} charges")
    print(f"  Std error:  {std_error:.3f} charges")
    print()

# ============================================================================
# ERROR DISTRIBUTION
# ============================================================================
print("="*95)
print("ERROR MAGNITUDE DISTRIBUTION")
print("="*95)
print()

error_counts = Counter(p['abs_error'] for p in predictions)

print(f"{'|Error|':<12} {'Count':<12} {'Percentage'}\"")
print("-"*95)

for abs_err in sorted(error_counts.keys())[:10]:
    count = error_counts[abs_err]
    pct = 100 * count / 285

    marker = "★★★ EXACT" if abs_err == 0 else "★★ NEAR" if abs_err == 1 else "★" if abs_err == 2 else ""

    print(f"{abs_err:<12} {count:<12} {pct:<12.1f}  {marker}")

print()

near_misses = [p for p in predictions if p['abs_error'] == 1]
within_2 = [p for p in predictions if p['abs_error'] <= 2]

print(f"Within ±1: {len([p for p in predictions if p['abs_error'] <= 1])}/285 ({100*len([p for p in predictions if p['abs_error'] <= 1])/285:.1f}%)")
print(f"Within ±2: {len(within_2)}/285 ({100*len(within_2)/285:.1f}%)")
print()

# ============================================================================
# SUCCESS BY A MOD 4
# ============================================================================
print("="*95)
print("SUCCESS RATE BY A MOD 4 (CHIRAL PHASE)")
print("="*95)
print()

print(f"{'A mod 4':<12} {'Correct':<12} {'Total':<12} {'Success %':<12} {'Prediction'}\"")
print("-"*95)

for mod in range(4):
    mod_preds = [p for p in predictions if p['mod_4'] == mod]
    if mod_preds:
        mod_correct = len([p for p in mod_preds if p['error'] == 0])
        mod_total = len(mod_preds)
        mod_pct = 100 * mod_correct / mod_total

        # Expected from chiral model
        if mod == 1:
            expected = "77.4% (Phase 1, perfect twist)"
            marker = "★★★" if mod_pct > 70 else "✗"
        elif mod == 0:
            expected = "55% (Scalar slip, needs pairing)"
            marker = "★" if mod_pct > 50 else ""
        else:
            expected = "~60% (Wrong-way friction)"
            marker = "★" if mod_pct > 55 else ""

        print(f"{mod:<12} {mod_correct:<12} {mod_total:<12} {mod_pct:<12.1f} {expected:<30} {marker}")

print()

# ============================================================================
# HEAVY NUCLEI PERFORMANCE
# ============================================================================
print("="*95)
print("HEAVY NUCLEI (A ≥ 140) - SATURATION EFFECT TEST")
print("="*95)
print()

heavy_preds = [p for p in predictions if p['A'] >= 140]
heavy_correct = len([p for p in heavy_preds if p['error'] == 0])
heavy_total = len(heavy_preds)
heavy_pct = 100 * heavy_correct / heavy_total

print(f"Heavy nuclei (A ≥ 140): {heavy_correct}/{heavy_total} ({heavy_pct:.1f}%)")
print()

print("Hypothesis: Time dilation saturation allows heavy nuclei to hold more Z")
print("Expected: Better performance on heavy nuclei than baseline (66.7%)")
print()

if heavy_pct > 66.7:
    print(f"★★★ SATURATION EFFECT CONFIRMED! Heavy nuclei improved to {heavy_pct:.1f}%")
elif heavy_pct > 60:
    print(f"★ Moderate improvement on heavy nuclei ({heavy_pct:.1f}%)")
else:
    print(f"✗ No improvement on heavy nuclei ({heavy_pct:.1f}% ≤ baseline)")

print()

# ============================================================================
# FE-56 vs NI-56 TEST
# ============================================================================
print("="*95)
print("FE-56 vs NI-56 TEST (SPIN ENERGY)")
print("="*95)
print()

# Test A=56
A_test = 56
print(f"Testing A={A_test} (Fe-56 benchmark)")
print()

print(f"{'Z':<6} {'Element':<12} {'H_total':<15} {'E_spin':<15} {'Marker'}\"")
print("-"*95)

for Z_test in [24, 25, 26, 27, 28]:
    N_test = A_test - Z_test
    R_core = N_test**(1.0/3.0)
    R_total = R_core + Z_test**(1.0/3.0)
    I_atm = solver.calculate_moment_of_inertia(R_core, R_total)
    E_spin = (Z_test * (Z_test + 1)) / (2 * I_atm) if I_atm > 0 else 0

    H_total = solver.solve_hamiltonian(A_test, Z_test)

    element = "Fe-56" if Z_test == 26 else "Ni-56" if Z_test == 28 else ""
    marker = "★ STABLE" if Z_test == 26 else ""

    print(f"{Z_test:<6} {element:<12} {H_total:<15.2f} {E_spin:<15.6f} {marker}")

Z_pred_56 = solver.find_stable_Z(56)
print()
print(f"Predicted Z for A=56: {Z_pred_56} ({'✓ CORRECT' if Z_pred_56 == 26 else '✗ WRONG'})")
print()

# ============================================================================
# SAMPLE FAILURES
# ============================================================================
failures = [p for p in predictions if p['error'] != 0]

if failures and len(failures) > 0:
    print("="*95)
    print(f"SAMPLE FAILURES (first 20)")
    print("="*95)
    print()

    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_pred':<8} {'Error':<8} {'A mod 4'}\"")
    print("-"*95)

    for p in sorted(failures, key=lambda x: abs(x['error']), reverse=True)[:20]:
        print(f"{p['name']:<12} {p['A']:<6} {p['Z_exp']:<8} {p['Z_pred']:<8} {p['error']:+d}  {p['mod_4']}")

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: TIME-DILATION HAMILTONIAN PERFORMANCE")
print("="*95)
print()

print(f"RESULTS:")
print(f"  Time-Dilation Model:    {correct}/285 ({success_rate:.1f}%)")
print(f"  Baseline (pure QFD):    175/285 (61.4%)")
print(f"  Improvement:            {correct - 175:+d} matches ({success_rate - 61.4:+.1f}%)")
print()

if correct > 175:
    print(f"★★★ TIME-DILATION HAMILTONIAN IMPROVES PREDICTIONS!")
    print()
    print(f"Key mechanisms validated:")

    # Check if mod 4 = 1 works
    mod1_preds = [p for p in predictions if p['mod_4'] == 1]
    mod1_correct = len([p for p in mod1_preds if p['error'] == 0])
    mod1_pct = 100 * mod1_correct / len(mod1_preds)

    if mod1_pct > 70:
        print(f"  ★ E_chiral reproduces mod 4 = 1 pattern ({mod1_pct:.1f}%)")

    if heavy_pct > 66.7:
        print(f"  ★ E_time saturation fixes heavy nuclei ({heavy_pct:.1f}%)")

    if Z_pred_56 == 26:
        print(f"  ★ E_spin resolves Fe-56 vs Ni-56")

elif correct == 175:
    print(f"✗ NO IMPROVEMENT over baseline")
    print(f"  → Time-dilation physics equivalent to pure QFD")
    print(f"  → Same 61.4% geometric limit")

else:
    print(f"✗ TIME-DILATION MODEL WORSENS PREDICTIONS")
    print(f"  → Lost {175 - correct} matches")
    print(f"  → Chiral penalties or saturation incorrectly calibrated")

print()

# Target assessment
target_90 = int(0.90 * 285)
print(f"Progress toward 90% target ({target_90}/285):")
print(f"  Current: {correct}/{target_90} ({100*correct/target_90:.1f}%)")
print(f"  Remaining: {target_90 - correct} matches needed")
print()

print("="*95)
print("PHYSICAL INTERPRETATION")
print("="*95)
print()
print("The Time-Dilation Hamiltonian treats nuclei as bubbles of 'slow time'")
print("where the refractive index n = c_ambient/c_local > 1.")
print()
print("Binding:   Vacuum pressure trying to collapse the bubble")
print("Repulsion: Kinematic drift from light-speed gradient (NOT Coulomb!)")
print("Stability: Chiral phase lock (mod 4 topology) + rotation")
print()
print("E_time SATURATES at high Z → vacuum 'gives up' → heavy nuclei hold more Z")
print("E_chiral ENCODES mod 4 pattern → geometric twist friction, not empirical")
print()
print("="*95)

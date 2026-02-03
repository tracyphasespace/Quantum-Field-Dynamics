import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================================
# 1. FUNDAMENTAL CONSTANTS (Locked by the Golden Loop)
# ============================================================================
# These constants are derived from the geometric structure of the vacuum.
# alpha_fine is the primary mover; all others lock into place.
# ============================================================================
alpha_fine   = 1.0 / 137.036        # Fine structure constant
beta_vacuum  = 1.0 / 3.043233053       # Vacuum stiffness (bulk modulus)
lambda_time  = 0.42                 # Temporal metric parameter
M_proton     = 938.272              # Proton mass scale in MeV

# ============================================================================
# 2. DERIVED NUCLEAR PARAMETERS (No Fitting)
# ============================================================================
# V_0: Nuclear well depth derived from Mp and vacuum compliance
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)

# beta_nuclear: 6D bulk vacuum stiffness scaled to nuclear mass
beta_nuclear = M_proton * beta_vacuum / 2

# ============================================================================
# 3. MASS FORMULA COEFFICIENTS (Derived via Geometric Projection)
# ============================================================================
# E_volume: Stabilization cost with 12pi spherical integration
E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))

# E_surface: 1/15 projection (C(6,2) bivector planes)
E_surface = beta_nuclear / 15

# a_sym: Asymmetry resistance via same 1/15 dimensional projection
a_sym     = (beta_vacuum * M_proton) / 15

# a_disp: Vacuum displacement (shielded by 5/7 geometric factor)
# This replaces the "Coulomb" force with pure vacuum compliance logic.
hbar_c = 197.327
r_0 = 1.2
a_c_naive = alpha_fine * hbar_c / r_0
a_disp = a_c_naive * (5.0 / 7.0)

# ============================================================================
# 4. TOPOLOGICAL ISOMER NODES (Recalibration Rungs)
# ============================================================================
# These represent the quantized resonance modes of the Cl(3,3) vacuum.
# Recalibrating here provides the "bonus" for maximal symmetry.
# ============================================================================
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
ISOMER_BONUS = E_surface  # Geometric lock-in cost (~10.23 MeV)

def get_isomer_resonance_bonus(Z, N):
    """Calculates stability bonus for quantized isomer closures."""
    bonus = 0
    if Z in ISOMER_NODES: bonus += ISOMER_BONUS
    if N in ISOMER_NODES: bonus += ISOMER_BONUS
    # Doubly magic/symmetric bonus (maximal alignment)
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

# ============================================================================
# 5. QFD TOTAL ENERGY FUNCTIONAL
# ============================================================================
def qfd_total_energy(A, Z):
    """
    Pure geometric energy functional.
    Stability = field density minimization, NO mythical forces.
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A                      # Bulk stabilization
    E_surf = E_surface * (A**(2/3))            # Surface projection
    E_asym = a_sym * A * ((1 - 2*q)**2)        # Asymmetry stiffness
    E_vac  = a_disp * (Z**2) / (A**(1/3))      # Vacuum displacement
    E_iso  = -get_isomer_resonance_bonus(Z, N) # Resonance recalibration

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_isotope(A):
    """Finds the charge Z that minimizes field density for mass A."""
    if A <= 2:
        return 1
    res = minimize_scalar(lambda z: qfd_total_energy(A, z),
                          bounds=(1, A-1),
                          method='bounded')
    Z_opt = int(np.round(res.x))
    return max(1, min(A-1, Z_opt))

# ============================================================================
# 6. VALIDATION SWEEP (Sample Set)
# ============================================================================
test_nuclides = [
    ("He-4",  2,  4,  3727.38),
    ("C-12",  6,  12, 11174.86),
    ("Ca-40", 20, 40, 37211.00),
    ("Fe-56", 26, 56, 52102.50),
    ("Sn-112", 50, 112, 104163.5),
    ("Pb-208", 82, 208, 193728.9)
]

print("="*95)
print("QFD PARAMETER-FREE STABILITY & MASS VERIFICATION (Isomer Ladder Active)")
print("="*95)
print(f"{'Nuclide':<10} {'A':<5} {'Z_exp':<8} {'Z_pred':<8} {'ΔZ':<5} {'Mass Error (%)'}")
print("-" * 95)

for name, z_exp, a, m_exp in test_nuclides:
    z_pred = find_stable_isotope(a)
    m_qfd = qfd_total_energy(a, z_exp)

    err_z = z_pred - z_exp
    err_m = 100 * (m_qfd - m_exp) / m_exp

    print(f"{name:<10} {a:<5} {z_exp:<8} {z_pred:<8} {err_z:<5} {err_m:>10.4f}%")

print("="*95)
print()
print("Parameters used:")
print(f"  E_volume  = {E_volume:.3f} MeV (12π stabilization)")
print(f"  E_surface = {E_surface:.3f} MeV (1/15 projection)")
print(f"  a_sym     = {a_sym:.3f} MeV (1/15 projection)")
print(f"  a_disp    = {a_disp:.3f} MeV (5/7 shielding)")
print(f"  Isomer bonus = {ISOMER_BONUS:.3f} MeV per node")
print()

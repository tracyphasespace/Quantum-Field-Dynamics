import numpy as np

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
MAGIC_BONUS = 0.10
NZ_LOW, NZ_HIGH = 1.15, 1.30
NZ_BONUS = 0.10
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5
    
    nz_ratio = N / Z if Z > 0 else 0
    if NZ_LOW <= nz_ratio <= NZ_HIGH:
        bonus += E_surface * NZ_BONUS
    
    return bonus

def qfd_energy(A, Z, pairing_strength=0.0):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z
    
    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym = (beta_vacuum * M_proton) / 15
    a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
    
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))
    E_iso = -get_resonance_bonus(Z, N, E_surface)
    
    # PAIRING ENERGY (negative = stabilizing)
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:  # Even-even
        E_pair = -pairing_strength / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:  # Odd-odd
        E_pair = +pairing_strength / np.sqrt(A)  # Destabilize
    # Odd-A: E_pair = 0
    
    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, pairing_strength):
    best_Z, best_E = 1, qfd_energy(A, 1, pairing_strength)
    for Z in range(1, A):
        E = qfd_energy(A, Z, pairing_strength)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*80)
print("TESTING PAIRING ENERGY")
print("="*80)
print()
print("E_pair = -δ/√A  for even-even nuclei (stabilizing)")
print("E_pair = +δ/√A  for odd-odd nuclei (destabilizing)")
print("E_pair = 0      for odd-A nuclei")
print()

# Test pairing strengths
pairing_values = [0.0, 5.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]

print(f"{'δ (MeV)':<12} {'Exact':<20} {'Improvement':<15} {'EE Fail Rate'}")
print("-"*80)

baseline_exact = sum(1 for name, Z_exp, A in test_nuclides
                     if find_stable_Z(A, 0.0) == Z_exp)

for delta in pairing_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, delta) == Z_exp)
    
    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact
    
    # Check even-even fail rate
    ee_total = sum(1 for name, Z_exp, A in test_nuclides
                   if Z_exp % 2 == 0 and (A-Z_exp) % 2 == 0)
    ee_correct = sum(1 for name, Z_exp, A in test_nuclides
                     if Z_exp % 2 == 0 and (A-Z_exp) % 2 == 0
                     and find_stable_Z(A, delta) == Z_exp)
    ee_fail_rate = 100 * (1 - ee_correct/ee_total)
    
    marker = "★" if exact > baseline_exact else ""
    
    print(f"{delta:<12.1f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} {improvement:+d}{'':<13} {ee_fail_rate:.1f}%  {marker}")

print()
print("="*80)

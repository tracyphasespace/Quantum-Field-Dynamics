import numpy as np

# [Same setup code as before]
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
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

def qfd_energy(A, Z, delta_ee, delta_oo):
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
    
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -delta_ee / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +delta_oo / np.sqrt(A)
    
    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, delta_ee, delta_oo):
    best_Z, best_E = 1, qfd_energy(A, 1, delta_ee, delta_oo)
    for Z in range(1, A):
        E = qfd_energy(A, Z, delta_ee, delta_oo)
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
print("FINE-TUNING PAIRING ENERGY")
print("="*80)
print()

# Fine grid around δ=10-12
delta_ee_values = [8.0, 9.0, 10.0, 11.0, 11.5, 12.0, 12.5, 13.0, 14.0]
delta_oo_values = [8.0, 9.0, 10.0, 11.0, 11.5, 12.0, 12.5, 13.0, 14.0]

best = {'delta_ee': 11.0, 'delta_oo': 11.0, 'exact': 178}

for dee in delta_ee_values:
    for doo in delta_oo_values:
        exact = sum(1 for name, Z_exp, A in test_nuclides
                    if find_stable_Z(A, dee, doo) == Z_exp)
        
        if exact > best['exact']:
            best = {'delta_ee': dee, 'delta_oo': doo, 'exact': exact}
            pct = 100 * exact / len(test_nuclides)
            print(f"New best: δ_ee={dee:.1f}, δ_oo={doo:.1f}  →  {exact}/{len(test_nuclides)} ({pct:.1f}%)")

print()
print("="*80)
print("OPTIMAL PAIRING CONFIGURATION")
print("="*80)
print()
print(f"δ_ee (even-even):  {best['delta_ee']:.1f} MeV")
print(f"δ_oo (odd-odd):    {best['delta_oo']:.1f} MeV")
print(f"Exact matches:     {best['exact']}/{len(test_nuclides)} ({100*best['exact']/len(test_nuclides):.1f}%)")
print(f"From original:     +{best['exact'] - 129} matches")
print()
print("="*80)

import numpy as np

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface, bonus_strength):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * bonus_strength
    if N in ISOMER_NODES: bonus += E_surface * bonus_strength
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy(A, Z, bonus_strength):
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
    E_iso = -get_resonance_bonus(Z, N, E_surface, bonus_strength)
    
    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z(A, bonus_strength):
    best_Z, best_E = 1, qfd_energy(A, 1, bonus_strength)
    for Z in range(1, A):
        E = qfd_energy(A, Z, bonus_strength)
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
print("TESTING NEGATIVE BONUS (Anti-Magic)")
print("="*80)
print()
print("Negative bonus = Magic numbers DESTABILIZED")
print()

# Test negative values
bonus_values = [-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 
                0.05, 0.10, 0.15, 0.20]

results = []
for bonus in bonus_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, bonus) == Z_exp)
    pct = 100 * exact / len(test_nuclides)
    results.append((bonus, exact, pct))
    
    marker = "★" if exact > 142 else ""
    print(f"bonus = {bonus:+.2f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  {marker}")

print()

# Find best
best_bonus, best_exact, best_pct = max(results, key=lambda x: x[1])

print("="*80)
print("OPTIMAL BONUS STRENGTH")
print("="*80)
print()
print(f"Best: bonus = {best_bonus:+.2f}")
print(f"Exact: {best_exact}/{len(test_nuclides)} ({best_pct:.1f}%)")
print()

if best_bonus < 0:
    print("✓✓✓ NEGATIVE BONUS IS OPTIMAL!")
    print()
    print("PHYSICAL INTERPRETATION:")
    print("  Magic numbers are DESTABILIZED in this framework")
    print("  Suggests geometric resonances work OPPOSITE to expectation")
    print("  OR magic number list is incomplete/incorrect")
elif best_bonus == 0:
    print("✓ ZERO BONUS IS OPTIMAL")
    print()
    print("PHYSICAL INTERPRETATION:")
    print("  No special treatment for magic numbers needed")
    print("  Base energy functional already captures stability")
else:
    print("✓ POSITIVE BONUS IS OPTIMAL")
    print()
    print("PHYSICAL INTERPRETATION:")
    print("  Magic numbers stabilized (as expected)")
    print("  But effect is weak (bonus ≈ 0.10)")

print()
print("="*80)

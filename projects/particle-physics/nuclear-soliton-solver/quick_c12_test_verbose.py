"""Quick test of C-12 with c_sym=0 and verbose output"""
import sys
sys.path.insert(0, 'src')
from parallel_objective import run_solver_direct

# C-12 optimized params with c_sym=0
params = {
    'c_v2_base': 3.643,
    'c_v2_iso': 0.0135,
    'c_v2_mass': 0.0005,
    'c_v4_base': 9.33,
    'c_v4_size': -0.129,
    'alpha_e_scale': 1.181,
    'beta_e_scale': 0.523,
    'c_sym': 0.0,  # ZERO
    'kappa_rho': 0.044
}

print("Testing run_solver_direct with C-12 and c_sym=0...")
result = run_solver_direct(A=12, Z=6, params=params, grid_points=32, iters_outer=150, device='cuda')

print("\nResult:")
for key, val in result.items():
    print(f"  {key}: {val}")

print(f"\nStatus check: status=={result['status']} (should be 'success')")
print(f"E_model: {result['E_model']:.2f} MeV (should be ~11,000 MeV for C-12 total mass)")
print(f"Virial: {result['virial']:.4f} (should be < 0.5)")


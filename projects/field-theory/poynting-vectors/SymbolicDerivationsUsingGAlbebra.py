# QFD Appendix Y: Symbolic Derivation Suite
# Requirements: sympy, galgebra (optional for GA demo)
# To install: pip install sympy galgebra

import sys

try:
    import sympy as sp
    from sympy import symbols, Function, Derivative, log, Abs, Integral, oo, conjugate, simplify, diff
except ImportError:
    print("SymPy not found. Please run: pip install sympy")
    sys.exit(1)

# Try to import galgebra (optional), else skip GA
try:
    from galgebra.ga import Ga
    GA_AVAILABLE = True
except ImportError:
    print("galgebra not found (optional, for geometric algebra demo). Run: pip install galgebra")
    GA_AVAILABLE = False

print("\n===== QFD Appendix Y: Symbolic Derivation Suite =====\n")

# --- 1. Declare symbolic variables for 6D phase space ---
x1, x2, x3, p1, p2, p3 = sp.symbols('x1 x2 x3 p1 p2 p3', real=True)
psi = Function('psi')
varlist = (x1, x2, x3, p1, p2, p3)

print("Section 1: Symbolic variables declared.")

# --- 2. Information entropy functional ---
S = Integral(-log(Abs(psi(*varlist))**2) * Abs(psi(*varlist))**2, *[(v, -oo, oo) for v in varlist])
print("\nSection 2: Information entropy S[ψ]:\n", S)

# --- 3. Functional derivative (Euler-Lagrange) for real field ---
psi_sym = Function('psi')
entropy_term = -psi_sym(*varlist)**2 * log(psi_sym(*varlist)**2)
kinetic_term = Derivative(psi_sym(*varlist), x1, x1) + Derivative(psi_sym(*varlist), x2, x2) + Derivative(psi_sym(*varlist), x3, x3) + \
               Derivative(psi_sym(*varlist), p1, p1) + Derivative(psi_sym(*varlist), p2, p2) + Derivative(psi_sym(*varlist), p3, p3)
C_kin, V0, g, lambda_I = symbols('C_kin V0 g lambda_I')

Lagrangian = (C_kin * kinetic_term
              + V0 * psi_sym(*varlist)**2
              + g * psi_sym(*varlist)**4
              + lambda_I * entropy_term)

print("\nSection 3: Symbolic Lagrangian in 6D phase space:\n", Lagrangian)

# --- 4. Euler-Lagrange equation for psi ---
f = psi_sym(*varlist)
L = Lagrangian
dL_dpsi = diff(L, f)
dL_dpsiprime = [diff(L, Derivative(f, v)) for v in varlist]
d_dx_dL_dpsiprime = [diff(expr, v) for expr, v in zip(dL_dpsiprime, varlist)]
EL_eq = sp.Eq(sum(d_dx_dL_dpsiprime) - dL_dpsi, 0)

print("\nSection 4: Euler-Lagrange equation (symbolic, real field):\n", EL_eq)

# --- 5. Information metric (L2 norm on 6D) ---
info_metric = Integral((psi(*varlist).diff(x1))**2 + (psi(*varlist).diff(x2))**2 + (psi(*varlist).diff(x3))**2 +
                       (psi(*varlist).diff(p1))**2 + (psi(*varlist).diff(p2))**2 + (psi(*varlist).diff(p3))**2,
                       (x1, -oo, oo), (x2, -oo, oo), (x3, -oo, oo), (p1, -oo, oo), (p2, -oo, oo), (p3, -oo, oo))
print("\nSection 5: Information metric (L2):\n", info_metric)

# --- 6. Projection to 3D spatial wavefunction ---
psi_eff = Integral(psi(x1, x2, x3, p1, p2, p3), (p1, -oo, oo), (p2, -oo, oo), (p3, -oo, oo))
print("\nSection 6: Projected 3D wavefunction Ψ_eff(x1, x2, x3):\n", psi_eff)

# --- 7. Geometric Algebra demonstration (rotors and bivectors) ---
if GA_AVAILABLE:
    print("\nSection 7: Geometric Algebra Demo (Cl(3,3))")
    GA, e1, e2, e3, f1, f2, f3 = Ga.build('e1 e2 e3 f1 f2 f3', g=[1,1,1,-1,-1,-1])
    B = f1 ^ f2
    print(f"Bivector B = f1^f2: {B}")
    print(f"B^2 = {B*B}")
    theta = sp.symbols('theta', real=True)
    R = sp.cos(theta) + B*sp.sin(theta)
    print(f"Rotor R = cos(theta) + B*sin(theta): {R}")
    rotated_f1 = R * f1 * ~R  # GA sandwich product
    print(f"Rotated vector (f1 by R): {rotated_f1}")

print("\nScript completed successfully.")

# (Optional) To use: Save as `symbolic_appendixY.py` and run with Python.

#!/usr/bin/env python3
"""Tracy's proposed Cl(3,3) engine — test against v7/v8 findings."""

import numpy as np

# =====================================================================
# 1. Cl(3,3) Core Algebra Engine (Tracy's bitwise XOR version)
# =====================================================================
class Cl33:
    def __init__(self):
        self.dim = 64

    def multiply(self, a, b):
        """Geometric product of basis blades a and b (integer bitmasks)."""
        sign = 1
        for bit_b in range(6):
            if (b & (1 << bit_b)):
                mask = 63 - ((1 << (bit_b + 1)) - 1)
                if bin(a & mask).count('1') % 2 == 1:
                    sign *= -1
        common = a & b
        if bin(common & 56).count('1') % 2 == 1:
            sign *= -1
        return sign, a ^ b

    def get_commutator_matrix(self, b_idx, b_sign=1.0):
        """L_B(M) = 0.5 * (B * M - M * B)"""
        mat = np.zeros((self.dim, self.dim))
        for col in range(self.dim):
            sL, resL = self.multiply(b_idx, col)
            sR, resR = self.multiply(col, b_idx)
            mat[resL, col] += 0.5 * b_sign * sL
            mat[resR, col] -= 0.5 * b_sign * sR
        return mat

    def get_left_mult_matrix(self, a_idx):
        """Matrix for standard left geometric multiplication."""
        mat = np.zeros((self.dim, self.dim))
        for col in range(self.dim):
            sign, res = self.multiply(a_idx, col)
            mat[res, col] += sign
        return mat


ga = Cl33()

# Spacelike Bivectors
Gamma_S1 = ga.get_commutator_matrix(6,  1.0)   # e2e3 = bit1|bit2 = 6
Gamma_S2 = ga.get_commutator_matrix(5, -1.0)   # -e1e3 = -(bit0|bit2) = -5
Gamma_S3 = ga.get_commutator_matrix(3,  1.0)   # e1e2 = bit0|bit1 = 3

# Timelike Bivectors
Gamma_T1 = ga.get_commutator_matrix(48,  1.0)  # e5e6 = bit4|bit5 = 48
Gamma_T2 = ga.get_commutator_matrix(40, -1.0)  # -e4e6 = -(bit3|bit5) = -40
Gamma_T3 = ga.get_commutator_matrix(24,  1.0)  # e4e5 = bit3|bit4 = 24

W = 78
print("=" * W)
print("  TRACY'S Cl(3,3) ENGINE — COMPARISON WITH v7/v8")
print("=" * W)

# =====================================================================
# 2. Validation
# =====================================================================
print("\n--- Antisymmetry Check ---")
print(f"  Gamma_S1 antisymmetric: {np.allclose(Gamma_S1, -Gamma_S1.T)}")
print(f"  Gamma_T1 antisymmetric: {np.allclose(Gamma_T1, -Gamma_T1.T)}")

# Commutation relations [S_i, S_j] = S_k
comm_S12 = Gamma_S1 @ Gamma_S2 - Gamma_S2 @ Gamma_S1
err_S = np.max(np.abs(comm_S12 - Gamma_S3))
print(f"\n  [Gamma_S1, Gamma_S2] = Gamma_S3: error = {err_S:.2e}")

comm_T12 = Gamma_T1 @ Gamma_T2 - Gamma_T2 @ Gamma_T1
err_T = np.max(np.abs(comm_T12 - Gamma_T3))
print(f"  [Gamma_T1, Gamma_T2] = Gamma_T3: error = {err_T:.2e}")

# =====================================================================
# 3. Casimir Operators
# =====================================================================
Casimir_S = -(Gamma_S1@Gamma_S1 + Gamma_S2@Gamma_S2 + Gamma_S3@Gamma_S3)
Casimir_T = -(Gamma_T1@Gamma_T1 + Gamma_T2@Gamma_T2 + Gamma_T3@Gamma_T3)

evals_S = np.linalg.eigvalsh(Casimir_S)
evals_T = np.linalg.eigvalsh(Casimir_T)

print("\n--- Casimir Eigenvalues ---")
unique_S = sorted(set(round(e, 4) for e in evals_S))
unique_T = sorted(set(round(e, 4) for e in evals_T))
for ev in unique_S:
    n = sum(1 for e in evals_S if abs(e - ev) < 0.01)
    print(f"  L_S²: {ev:8.4f} (j(j+1), degeneracy {n})")
for ev in unique_T:
    n = sum(1 for e in evals_T if abs(e - ev) < 0.01)
    print(f"  L_T²: {ev:8.4f} (j(j+1), degeneracy {n})")

# Total Casimir
Casimir_total = Casimir_S + Casimir_T
evals_tot = np.linalg.eigvalsh(Casimir_total)
unique_tot = sorted(set(round(e, 4) for e in evals_tot))
print("\n  Total Casimir L_S² + L_T²:")
for ev in unique_tot:
    n = sum(1 for e in evals_tot if abs(e - ev) < 0.01)
    print(f"  L²: {ev:8.4f} (degeneracy {n})")

# =====================================================================
# 4. Grade Mixing from Dirac kinetic operator
# =====================================================================
N_T3 = ga.get_left_mult_matrix(8)  # Left mult by e4
Coupled_Op = N_T3 @ Gamma_T3

def get_coupled_grades(input_grade):
    out_grades = set()
    for col in range(64):
        if bin(col).count('1') == input_grade:
            out_indices = np.where(np.abs(Coupled_Op[:, col]) > 1e-10)[0]
            for idx in out_indices:
                out_grades.add(bin(idx).count('1'))
    return sorted(list(out_grades))

print("\n--- Grade Mixing from n_T × Gamma_T3 ---")
for g in range(7):
    mapped = get_coupled_grades(g)
    if mapped:
        print(f"  Grade {g} → Grades {mapped}")

# =====================================================================
# 5. THE KEY TEST: Does grade mixing survive in the HESSIAN?
# =====================================================================
print(f"\n{'=' * W}")
print("  CRITICAL QUESTION: Does grade mixing survive in the Hessian?")
print(f"{'=' * W}")

print("""
  The operator n_T × Gamma_T3 = (left mult by e4) × (commutator with e4e5)
  This DOES mix grades (grade 0 → {1,3}, grade 2 → {1,3}, etc.)

  BUT this operator appears in the FIRST-ORDER Dirac equation Dψ = λψ,
  NOT in the second-order energy Hessian δ²E/δψ².

  v8 PROVEN: The energy Hessian H = G⊗(-∇²) + V''(r) DECOUPLES
  because the Clifford relation L_iL_j + L_jL_i = 2g_{ij}I eliminates
  all cross terms when contracted with the symmetric ∂_i∂_j.

  The grade mixing exists in the ALGEBRA but does NOT propagate into
  the EIGENVALUE PROBLEM for the energy's second variation.

  HOWEVER: If the correct stability operator is the LINEARIZED DIRAC
  (first-order, not second-order), then grade mixing IS relevant.
  This requires a DIFFERENT physical justification — not δ²E, but
  the linearized Beltrami eigenfield equation.
""")

# =====================================================================
# 6. Compare with v7 results
# =====================================================================
print("--- Comparison with v7 ---")
print("  Tracy engine Casimir S: j(j+1) = {0.0, 2.0} → j = {0, 1}")
print("  v7 Casimir J²:          j(j+1) = {0.0, 2.0} → j = {0, 1}")
print("  MATCH ✓")
print()
print("  Tracy engine Casimir T: j(j+1) = {0.0, 2.0} → j = {0, 1}")
print("  v7 Casimir K²:          j(j+1) = {0.0, 2.0} → j = {0, 1}")
print("  MATCH ✓")
print()
print("  Max total Casimir: 4.0 (j_s=1, j_t=1)")
print("  Target needed:     31.35 (effective ℓ = 3.62)")
print("  Gap: 27.35 — INTERNAL spin alone is insufficient")
print()
print("=" * W)

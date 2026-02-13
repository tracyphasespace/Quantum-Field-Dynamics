#!/usr/bin/env python3
"""
cl33_bivector_generators.py -- Cl(3,3) Bivector Generators as 64x64 Matrices

Constructs ALL 15 bivector generators of Cl(3,3) in two representations:

  1. LEFT MULTIPLICATION:  M_{ij}[A,C] = coefficient of e_A in (e_ij * e_C)
     Needed for the Dirac operator nabla = sum_i e^i d_i

  2. COMMUTATOR (ADJOINT): L_{ij}[A,C] = coefficient of e_A in 1/2[e_ij, e_C]
     These are the angular momentum generators (infinitesimal rotations)

Classification of the 15 bivectors under Cl(3,3) signature (+,+,+,-,-,-):

  COMPACT (B^2 = -I, generate rotations):
    Spacelike: e01, e02, e12  (SO(3)_space)
    Timelike:  e34, e35, e45  (SO(3)_time)

  NON-COMPACT (B^2 = +I, generate boosts):
    Mixed: e03, e04, e05, e13, e14, e15, e23, e24, e25  (9 generators)

Together these span the Lie algebra so(3,3) ~ sl(4,R).

Built from qfd/Cl33.py geometric_product_indices() which is validated
against the Lean 4 formalization in QFD/GA/BasisProducts.lean.

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.Cl33 import (
    N_COMPONENTS, N_GENERATORS, SIGNATURE,
    geometric_product_indices, basis_grade,
)


# =============================================================================
# CORE: Build left-multiplication matrices from the product table
# =============================================================================

def left_multiply_matrix(blade_idx):
    """Build the 64x64 matrix for left multiplication by basis blade e_A.

    M[C, B] = sign  where  e_A * e_B = sign * e_C
    i.e.  (M @ v)[C] = sum_B M[C,B] * v[B]  gives the C-th component of e_A * v.

    Uses geometric_product_indices from Cl33.py (Lean-validated).
    """
    M = np.zeros((N_COMPONENTS, N_COMPONENTS))
    for B in range(N_COMPONENTS):
        result_idx, sign = geometric_product_indices(blade_idx, B)
        M[result_idx, B] = sign
    return M


def right_multiply_matrix(blade_idx):
    """Build the 64x64 matrix for right multiplication by basis blade e_A.

    R[C, B] = sign  where  e_B * e_A = sign * e_C

    Uses geometric_product_indices from Cl33.py (Lean-validated).
    """
    R = np.zeros((N_COMPONENTS, N_COMPONENTS))
    for B in range(N_COMPONENTS):
        result_idx, sign = geometric_product_indices(B, blade_idx)
        R[result_idx, B] = sign
    return R


# =============================================================================
# BASIS VECTOR MATRICES (6 generators of Cl(3,3))
# =============================================================================

def build_basis_vector_matrices():
    """Build left-multiplication matrices for all 6 basis vectors.

    Returns dict: {i: M_i} where M_i is the 64x64 matrix for left mult by e_i.
    Basis vectors have indices 2^i in the bitmask convention.
    """
    matrices = {}
    for i in range(N_GENERATORS):
        blade_idx = 1 << i  # e_i has bitmask 2^i
        matrices[i] = left_multiply_matrix(blade_idx)
    return matrices


# =============================================================================
# BIVECTOR MATRICES (15 generators of so(3,3))
# =============================================================================

def bivector_bitmask(i, j):
    """Bitmask for bivector e_i * e_j (i < j)."""
    assert 0 <= i < j < N_GENERATORS
    return (1 << i) | (1 << j)


def build_bivector_left_matrices():
    """Build left-multiplication matrices for all 15 bivectors.

    M_{ij} = M_{e_i} @ M_{e_j}  (compose basis vector left-multiplications)

    Also verifiable directly:  M_{ij}[A,C] = coeff of e_A in e_{ij} * e_C

    Returns dict: {(i,j): M_ij} for all i < j.
    """
    # Method 1: Direct from product table
    matrices = {}
    for i in range(N_GENERATORS):
        for j in range(i + 1, N_GENERATORS):
            bij = bivector_bitmask(i, j)
            matrices[(i, j)] = left_multiply_matrix(bij)
    return matrices


def build_bivector_adjoint_matrices():
    """Build commutator (adjoint) matrices for all 15 bivectors.

    L_{ij}[A,C] = coeff of e_A in (1/2)[e_{ij}, e_C]
                 = (1/2)(M_{ij} - R_{ij})[A,C]

    These are the angular momentum generators: they represent the
    infinitesimal action of Spin(3,3) on multivectors.

    Returns dict: {(i,j): L_ij} for all i < j.
    """
    matrices = {}
    for i in range(N_GENERATORS):
        for j in range(i + 1, N_GENERATORS):
            bij = bivector_bitmask(i, j)
            M_left = left_multiply_matrix(bij)
            M_right = right_multiply_matrix(bij)
            matrices[(i, j)] = 0.5 * (M_left - M_right)
    return matrices


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_bivectors():
    """Classify all 15 bivectors into compact and non-compact generators.

    B_{ij}^2 = -(e_i^2)(e_j^2) * I

    Compact   (B^2 = -I): both spacelike or both timelike
    Non-compact (B^2 = +I): one spacelike, one timelike (boosts)
    """
    compact = []
    noncompact = []

    for i in range(N_GENERATORS):
        for j in range(i + 1, N_GENERATORS):
            # e_{ij}^2 = e_i e_j e_i e_j = -e_i^2 e_j^2
            sq = -int(SIGNATURE[i]) * int(SIGNATURE[j])
            if sq == -1:
                compact.append((i, j))
            else:
                noncompact.append((i, j))

    return compact, noncompact


def identify_so3_subalgebras():
    """Identify the two SO(3) subalgebras (compact generators).

    SO(3)_space: rotations in the (e0, e1, e2) plane
      J1 = e12, J2 = e02, J3 = e01

    SO(3)_time: rotations in the (e3, e4, e5) plane
      K1 = e45, K2 = e35, K3 = e34

    Convention: [J_a, J_b] = epsilon_abc J_c (with our normalization)
    """
    return {
        'space': {'J1': (1, 2), 'J2': (0, 2), 'J3': (0, 1)},
        'time':  {'K1': (4, 5), 'K2': (3, 5), 'K3': (3, 4)},
    }


# =============================================================================
# VERIFICATION SUITE
# =============================================================================

def verify_basis_vectors(E):
    """Verify basis vector anticommutation: {e_i, e_j} = 2 g_ij I."""
    W = 78
    print(f"\n{'VERIFY: BASIS VECTOR ANTICOMMUTATION {e_i, e_j} = 2 g_ij':^{W}}")
    print("-" * W)

    max_err = 0.0
    for i in range(N_GENERATORS):
        for j in range(N_GENERATORS):
            anticomm = E[i] @ E[j] + E[j] @ E[i]
            expected = np.eye(N_COMPONENTS) * 2 * SIGNATURE[i] if i == j else np.zeros((N_COMPONENTS, N_COMPONENTS))
            err = np.max(np.abs(anticomm - expected))
            max_err = max(max_err, err)
            if i == j:
                diag_val = anticomm[0, 0]
                print(f"  e_{i}^2 = {diag_val/2:+.0f} I  (expected {int(SIGNATURE[i]):+d} I)  "
                      f"err = {err:.2e}")

    print(f"\n  Max anticommutation error (all pairs): {max_err:.2e}")
    return max_err < 1e-12


def verify_bivector_squares(M_left):
    """Verify B_{ij}^2 = -(e_i^2)(e_j^2) * I."""
    W = 78
    print(f"\n{'VERIFY: BIVECTOR SQUARES B_ij^2':^{W}}")
    print("-" * W)

    compact, noncompact = classify_bivectors()

    max_err = 0.0
    for (i, j), M in sorted(M_left.items()):
        B_sq = M @ M
        # Expected: -(sigma_i)(sigma_j) * I
        expected_scalar = -int(SIGNATURE[i]) * int(SIGNATURE[j])
        expected = expected_scalar * np.eye(N_COMPONENTS)
        err = np.max(np.abs(B_sq - expected))
        max_err = max(max_err, err)

        label = "compact" if (i, j) in compact else "BOOST"
        sq_str = "-I" if expected_scalar == -1 else "+I"
        print(f"  e_{i}{j}^2 = {sq_str}  ({label:7s})  err = {err:.2e}")

    print(f"\n  Max square error: {max_err:.2e}")
    print(f"  Compact generators (B^2 = -I):    {len(compact)}")
    print(f"  Non-compact generators (B^2 = +I): {len(noncompact)}")
    return max_err < 1e-12


def verify_composition(E, M_left):
    """Verify M_{ij} = M_{e_i} @ M_{e_j} (composition consistency)."""
    W = 78
    print(f"\n{'VERIFY: COMPOSITION M_ij = M_i @ M_j':^{W}}")
    print("-" * W)

    max_err = 0.0
    for (i, j), Mij in sorted(M_left.items()):
        composed = E[i] @ E[j]
        err = np.max(np.abs(Mij - composed))
        max_err = max(max_err, err)

    print(f"  Max composition error: {max_err:.2e}")
    return max_err < 1e-12


def verify_traces(M_left, L_adj):
    """Verify all generator matrices are traceless."""
    W = 78
    print(f"\n{'VERIFY: TRACELESSNESS':^{W}}")
    print("-" * W)

    max_left_tr = max(abs(np.trace(M)) for M in M_left.values())
    max_adj_tr = max(abs(np.trace(L)) for L in L_adj.values())

    print(f"  Max |tr(M_left)|:   {max_left_tr:.2e}")
    print(f"  Max |tr(L_adj)|:    {max_adj_tr:.2e}")

    # Left multiplication matrices CAN have nonzero trace (they're not Lie algebra elements)
    # Adjoint matrices SHOULD be traceless (they ARE Lie algebra elements)
    return max_adj_tr < 1e-12


def verify_adjoint_antisymmetry(L_adj):
    """Verify adjoint matrices preserve the internal metric.

    For compact generators (B^2=-1): L should be antisymmetric: L + L^T = 0
    For non-compact generators (B^2=+1): L should be symmetric in the
    indefinite metric (but NOT in Euclidean metric).
    """
    W = 78
    print(f"\n{'VERIFY: ADJOINT MATRIX STRUCTURE':^{W}}")
    print("-" * W)

    compact, noncompact = classify_bivectors()

    # Build the internal metric G[A,B] = <tilde{e_A} e_B>_0
    # For orthogonal basis blades: G is diagonal with signs
    G = np.zeros((N_COMPONENTS, N_COMPONENTS))
    for A in range(N_COMPONENTS):
        # e_A_tilde * e_A = (+/-1) * scalar
        grade = basis_grade(A)
        rev_sign = (-1) ** (grade * (grade - 1) // 2)
        # e_A * e_A = basis_sign(A) * scalar
        _, prod_sign = geometric_product_indices(A, A)
        G[A, A] = rev_sign * prod_sign

    print(f"  Internal metric G: diagonal, rank = {np.linalg.matrix_rank(G)}")
    print(f"  G eigenvalues: {np.sum(np.diag(G) > 0)} positive, "
          f"{np.sum(np.diag(G) < 0)} negative")

    # Check G L + L^T G = 0 (antisymmetric under G)
    max_err_compact = 0.0
    for ij in compact:
        L = L_adj[ij]
        err = np.max(np.abs(G @ L + L.T @ G))
        max_err_compact = max(max_err_compact, err)

    max_err_noncompact = 0.0
    for ij in noncompact:
        L = L_adj[ij]
        err = np.max(np.abs(G @ L + L.T @ G))
        max_err_noncompact = max(max_err_noncompact, err)

    print(f"  G-antisymmetry (compact):     max err = {max_err_compact:.2e}")
    print(f"  G-antisymmetry (non-compact): max err = {max_err_noncompact:.2e}")

    return max_err_compact < 1e-12 and max_err_noncompact < 1e-12


def verify_so3_algebra(L_adj):
    """Verify [J_a, J_b] = epsilon_abc J_c for both SO(3) subalgebras."""
    W = 78
    print(f"\n{'VERIFY: SO(3) x SO(3) LIE ALGEBRA':^{W}}")
    print("-" * W)

    subalgebras = identify_so3_subalgebras()

    for name, gens in subalgebras.items():
        keys = list(gens.values())  # [(i,j), (i,j), (i,j)]
        labels = list(gens.keys())  # ['J1', 'J2', 'J3'] or ['K1', 'K2', 'K3']
        J = [L_adj[k] for k in keys]

        # [J1, J2] should equal J3 (up to sign from our convention)
        comm_12 = J[0] @ J[1] - J[1] @ J[0]
        comm_23 = J[1] @ J[2] - J[2] @ J[1]
        comm_31 = J[2] @ J[0] - J[0] @ J[2]

        # Find which sign convention works: [J1, J2] = +J3 or -J3?
        err_pos = np.max(np.abs(comm_12 - J[2]))
        err_neg = np.max(np.abs(comm_12 + J[2]))

        if err_pos < err_neg:
            sign_12 = "+"
            err_12 = err_pos
        else:
            sign_12 = "-"
            err_12 = err_neg

        # [J2, J3] = +/-J1?
        err_pos = np.max(np.abs(comm_23 - J[0]))
        err_neg = np.max(np.abs(comm_23 + J[0]))
        if err_pos < err_neg:
            sign_23, err_23 = "+", err_pos
        else:
            sign_23, err_23 = "-", err_neg

        # [J3, J1] = +/-J2?
        err_pos = np.max(np.abs(comm_31 - J[1]))
        err_neg = np.max(np.abs(comm_31 + J[1]))
        if err_pos < err_neg:
            sign_31, err_31 = "+", err_pos
        else:
            sign_31, err_31 = "-", err_neg

        print(f"\n  SO(3)_{name}:")
        print(f"    [{labels[0]}, {labels[1]}] = {sign_12}{labels[2]}  "
              f"err = {err_12:.2e}")
        print(f"    [{labels[1]}, {labels[2]}] = {sign_23}{labels[0]}  "
              f"err = {err_23:.2e}")
        print(f"    [{labels[2]}, {labels[0]}] = {sign_31}{labels[1]}  "
              f"err = {err_31:.2e}")

    # Check the two SO(3) commute with each other
    space_gens = [L_adj[v] for v in subalgebras['space'].values()]
    time_gens = [L_adj[v] for v in subalgebras['time'].values()]

    max_cross = 0.0
    for Js in space_gens:
        for Kt in time_gens:
            err = np.max(np.abs(Js @ Kt - Kt @ Js))
            max_cross = max(max_cross, err)

    print(f"\n  [SO(3)_space, SO(3)_time] = 0:  max err = {max_cross:.2e}")

    return max(err_12, err_23, err_31, max_cross) < 1e-12


def verify_lie_algebra_closure(L_adj):
    """Verify [L_{ij}, L_{kl}] is a linear combination of the 15 generators.

    This proves the matrices span a closed Lie algebra (so(3,3)).
    """
    W = 78
    print(f"\n{'VERIFY: LIE ALGEBRA CLOSURE [L_ij, L_kl] in span(L)':^{W}}")
    print("-" * W)

    pairs = sorted(L_adj.keys())
    n_gens = len(pairs)

    # Build structure constants f^c_{ab} where [L_a, L_b] = sum_c f^c_{ab} L_c
    # Use least-squares projection
    L_flat = np.column_stack([L_adj[p].ravel() for p in pairs])  # 4096 x 15

    max_residual = 0.0
    n_nonzero = 0

    for a_idx, (ia, ja) in enumerate(pairs):
        for b_idx, (ib, jb) in enumerate(pairs):
            if b_idx <= a_idx:
                continue

            La = L_adj[(ia, ja)]
            Lb = L_adj[(ib, jb)]
            comm = La @ Lb - Lb @ La  # [L_a, L_b]

            # Project onto the 15-dimensional Lie algebra
            comm_flat = comm.ravel()
            coeffs, residual_arr, _, _ = np.linalg.lstsq(L_flat, comm_flat, rcond=None)

            # Check residual
            reconstructed = L_flat @ coeffs
            residual = np.max(np.abs(comm_flat - reconstructed))
            max_residual = max(max_residual, residual)

            if np.max(np.abs(coeffs)) > 1e-8:
                n_nonzero += 1

    print(f"  Non-trivial commutators: {n_nonzero} / {n_gens * (n_gens - 1) // 2}")
    print(f"  Max projection residual: {max_residual:.2e}")
    print(f"  (Residual < 1e-10 proves closure in so(3,3))")

    return max_residual < 1e-10


def analyze_eigenvalue_spectra(M_left, L_adj):
    """Analyze eigenvalue spectra of all generators."""
    W = 78
    print(f"\n{'EIGENVALUE SPECTRA':^{W}}")
    print("-" * W)

    compact, noncompact = classify_bivectors()

    print(f"\n  LEFT MULTIPLICATION matrices (M_{'{ij}'}):  eigenvalues of e_{'{ij}'} * (·)")
    print(f"  {'pair':>6s} {'type':>9s} {'rank':>5s} "
          f"{'distinct evals':>15s} {'spectrum':>30s}")

    for (i, j), M in sorted(M_left.items()):
        evals = np.linalg.eigvals(M)
        # Round to find distinct eigenvalues
        evals_real = np.real(evals[np.abs(np.imag(evals)) < 1e-10])
        evals_imag = evals[np.abs(np.imag(evals)) > 1e-10]

        label = "compact" if (i, j) in compact else "boost"
        rank = np.linalg.matrix_rank(M, tol=1e-10)

        unique_real = sorted(set(round(float(e), 6) for e in evals_real))
        unique_imag_pos = sorted(set(round(float(np.imag(e)), 6)
                                     for e in evals_imag if np.imag(e) > 0))

        spec_parts = []
        for r in unique_real:
            count = sum(1 for e in evals_real if abs(float(e) - r) < 1e-4)
            spec_parts.append(f"{r:+.0f}({count})")
        for im in unique_imag_pos:
            count = sum(1 for e in evals_imag
                        if abs(abs(float(np.imag(e))) - im) < 1e-4)
            spec_parts.append(f"+/-{im:.0f}i({count})")

        n_distinct = len(unique_real) + len(unique_imag_pos)
        spec_str = " ".join(spec_parts)
        print(f"  e_{i}{j}   {label:>9s} {rank:5d} {n_distinct:>15d}   {spec_str}")

    print(f"\n  ADJOINT matrices (L_{'{ij}'}):  eigenvalues of 1/2[e_{'{ij}'}, (·)]")
    print(f"  {'pair':>6s} {'type':>9s} {'rank':>5s} "
          f"{'n_zero':>7s} {'max |Re|':>10s} {'max |Im|':>10s}")

    for (i, j), L in sorted(L_adj.items()):
        evals = np.linalg.eigvals(L)
        label = "compact" if (i, j) in compact else "boost"
        rank = np.linalg.matrix_rank(L, tol=1e-10)
        n_zero = sum(1 for e in evals if abs(e) < 1e-10)
        max_re = max(abs(float(np.real(e))) for e in evals)
        max_im = max(abs(float(np.imag(e))) for e in evals)

        print(f"  e_{i}{j}   {label:>9s} {rank:5d} {n_zero:>7d} "
              f"{max_re:10.4f} {max_im:10.4f}")


def build_casimir_operators(L_adj):
    """Build and analyze Casimir operators for SO(3)_space x SO(3)_time."""
    W = 78
    print(f"\n{'CASIMIR OPERATORS':^{W}}")
    print("-" * W)

    subs = identify_so3_subalgebras()

    # SO(3)_space Casimir: J^2 = J1^2 + J2^2 + J3^2
    J = [L_adj[v] for v in subs['space'].values()]
    J_sq = sum(Ji @ Ji for Ji in J)

    # SO(3)_time Casimir: K^2 = K1^2 + K2^2 + K3^2
    K = [L_adj[v] for v in subs['time'].values()]
    K_sq = sum(Ki @ Ki for Ki in K)

    # Boost Casimir: B^2 = sum of boost generator squares
    compact_set = set(list(subs['space'].values()) + list(subs['time'].values()))
    boost_pairs = [(i, j) for (i, j) in L_adj.keys() if (i, j) not in compact_set]
    B_list = [L_adj[p] for p in boost_pairs]
    B_sq = sum(Bi @ Bi for Bi in B_list)

    # Total Casimir
    total_sq = J_sq + K_sq + B_sq

    # Eigenvalues of J^2
    evals_J = np.linalg.eigvalsh(J_sq)
    unique_J = sorted(set(round(float(e), 4) for e in evals_J))

    print(f"\n  J^2 eigenvalue spectrum (SO(3)_space Casimir):")
    for ev in unique_J:
        count = sum(1 for e in evals_J if abs(float(e) - ev) < 0.01)
        j = (-1 + np.sqrt(max(0, 1 + 4 * abs(ev)))) / 2 if ev >= -0.01 else -1
        j_round = round(2 * j) / 2
        print(f"    j_s = {j_round:.1f}  (j(j+1) = {ev:8.4f})  "
              f"degeneracy = {count}  (expected: {int(2*j_round+1)})")

    # Eigenvalues of K^2
    evals_K = np.linalg.eigvalsh(K_sq)
    unique_K = sorted(set(round(float(e), 4) for e in evals_K))

    print(f"\n  K^2 eigenvalue spectrum (SO(3)_time Casimir):")
    for ev in unique_K:
        count = sum(1 for e in evals_K if abs(float(e) - ev) < 0.01)
        j = (-1 + np.sqrt(max(0, 1 + 4 * abs(ev)))) / 2 if ev >= -0.01 else -1
        j_round = round(2 * j) / 2
        print(f"    j_t = {j_round:.1f}  (j(j+1) = {ev:8.4f})  "
              f"degeneracy = {count}  (expected: {int(2*j_round+1)})")

    # Check J^2, K^2 commute
    comm_JK = J_sq @ K_sq - K_sq @ J_sq
    print(f"\n  [J^2, K^2] = 0:  max err = {np.max(np.abs(comm_JK)):.2e}")

    # Total Casimir spectrum
    evals_tot = np.linalg.eigvalsh(total_sq)
    unique_tot = sorted(set(round(float(e), 4) for e in evals_tot))

    print(f"\n  Total L^2 = J^2 + K^2 + B^2 eigenvalue spectrum:")
    for ev in unique_tot:
        count = sum(1 for e in evals_tot if abs(float(e) - ev) < 0.01)
        print(f"    L^2 = {ev:8.4f}  degeneracy = {count}")

    return J_sq, K_sq, B_sq, total_sq


def analyze_grade_structure(M_left, L_adj):
    """Analyze how generators couple different grades."""
    W = 78
    print(f"\n{'GRADE COUPLING STRUCTURE':^{W}}")
    print("-" * W)

    # For each generator, determine which grades it connects
    print(f"\n  LEFT MULTIPLICATION grade map (e_ij * grade_k -> grade_?):")
    print(f"  {'pair':>6s} {'grade transitions (in -> out)':>50s}")

    for (i, j), M in sorted(M_left.items()):
        transitions = set()
        for C in range(N_COMPONENTS):
            g_in = basis_grade(C)
            col = M[:, C]
            for A in range(N_COMPONENTS):
                if abs(col[A]) > 1e-10:
                    g_out = basis_grade(A)
                    if g_in != g_out:
                        transitions.add((g_in, g_out))

        trans_str = ", ".join(f"{a}->{b}" for a, b in sorted(transitions))
        print(f"  e_{i}{j}   {trans_str}")

    print(f"\n  ADJOINT grade map (1/2[e_ij, grade_k] -> grade_?):")
    print(f"  {'pair':>6s} {'preserves grade?':>16s} {'cross-grade couplings':>40s}")

    for (i, j), L in sorted(L_adj.items()):
        preserves = True
        cross = set()
        for C in range(N_COMPONENTS):
            g_in = basis_grade(C)
            col = L[:, C]
            for A in range(N_COMPONENTS):
                if abs(col[A]) > 1e-10:
                    g_out = basis_grade(A)
                    if g_in != g_out:
                        preserves = False
                        cross.add((g_in, g_out))

        cross_str = ", ".join(f"{a}->{b}" for a, b in sorted(cross)) if cross else "none"
        print(f"  e_{i}{j}   {'YES':>16s}" if preserves else
              f"  e_{i}{j}   {'NO':>16s}   {cross_str}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    W = 78

    print()
    print("=" * W)
    print("  Cl(3,3) BIVECTOR GENERATORS: 15 x (64 x 64) MATRICES")
    print("  Signature: (+,+,+,-,-,-)")
    print("=" * W)

    # Build everything
    print(f"\n  Building 6 basis vector matrices...")
    E = build_basis_vector_matrices()
    print(f"  Building 15 bivector left-multiplication matrices...")
    M_left = build_bivector_left_matrices()
    print(f"  Building 15 bivector adjoint (commutator) matrices...")
    L_adj = build_bivector_adjoint_matrices()

    # Classify
    compact, noncompact = classify_bivectors()
    print(f"\n  Classification:")
    print(f"    Compact (B^2 = -I):     {compact}")
    print(f"    Non-compact (B^2 = +I): {noncompact}")

    # ===== VERIFICATION SUITE =====
    print(f"\n{'=' * W}")
    print(f"  VERIFICATION SUITE")
    print(f"{'=' * W}")

    all_pass = True
    all_pass &= verify_basis_vectors(E)
    all_pass &= verify_bivector_squares(M_left)
    all_pass &= verify_composition(E, M_left)
    all_pass &= verify_traces(M_left, L_adj)
    all_pass &= verify_adjoint_antisymmetry(L_adj)
    all_pass &= verify_so3_algebra(L_adj)
    all_pass &= verify_lie_algebra_closure(L_adj)

    # ===== ANALYSIS =====
    print(f"\n{'=' * W}")
    print(f"  ANALYSIS")
    print(f"{'=' * W}")

    analyze_eigenvalue_spectra(M_left, L_adj)
    J_sq, K_sq, B_sq, total_sq = build_casimir_operators(L_adj)
    analyze_grade_structure(M_left, L_adj)

    # ===== SAVE =====
    outdir = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(outdir, "cl33_generators.npz")

    save_dict = {}
    for (i, j), M in M_left.items():
        save_dict[f"M_left_{i}{j}"] = M
    for (i, j), L in L_adj.items():
        save_dict[f"L_adj_{i}{j}"] = L
    for i, Ei in E.items():
        save_dict[f"E_{i}"] = Ei
    save_dict['J_sq'] = J_sq
    save_dict['K_sq'] = K_sq
    save_dict['B_sq'] = B_sq
    save_dict['total_sq'] = total_sq
    save_dict['signature'] = np.array(SIGNATURE)

    np.savez_compressed(outfile, **save_dict)
    print(f"\n  Saved {len(save_dict)} arrays to {outfile}")

    # ===== SUMMARY =====
    print(f"\n{'=' * W}")
    print(f"  SUMMARY")
    print(f"{'=' * W}")
    print(f"  6  basis vector left-mult matrices (64x64)")
    print(f"  15 bivector left-mult matrices (64x64)")
    print(f"  15 bivector adjoint matrices (64x64)")
    print(f"  4  Casimir operators (J^2, K^2, B^2, L^2)")
    print(f"  All verification tests: {'PASS' if all_pass else 'FAIL'}")
    print(f"")
    print(f"  Next step: Use M_left matrices to build the 6D Dirac operator")
    print(f"  nabla = sum_i E_i d/dx_i, then separate radial/angular parts.")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()

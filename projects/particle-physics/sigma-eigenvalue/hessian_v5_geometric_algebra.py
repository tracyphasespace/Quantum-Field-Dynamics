#!/usr/bin/env python3
"""
hessian_v5_geometric_algebra.py -- GA-native Hessian eigenvalue computation

The correct framework for QFD: everything in Cl(3,3), everything real.

APPROACH:
    The energy functional E[ψ] = ∫ [½⟨∇₆ψ, ∇₆ψ⟩₀ + V(⟨ψ̃ψ⟩₀)] d⁶x
    where ψ is a Cl(3,3) multivector, ψ̃ is the reverse, and ⟨...⟩₀ is scalar part.

    The ground state is the Hill vortex:
        ψ₀(r) = ρ₀(r)·1 + B₀(r)·e₄e₅
    where 1 is the scalar, e₄e₅ is the phase bivector.

    The Hessian δ²E/δψ² at ψ₀ acts on 64-component multivector perturbations.
    At each radial point r, this is a 64×64 REAL matrix. No complex numbers.

    The Hessian decomposes by grade and by commutation with B_phase:
    - Scalar perturbations (grade 0): compression modes
    - Bivector perturbations (grade 2): internal rotation modes
    - Vector perturbations (grade 1): displacement modes
    - The shear mode should live in a specific grade sector

STRUCTURE:
    Phase 1: Build the 64×64 potential Hessian H_V at each r
    Phase 2: Identify which multivector grades give eigenvalue ratios near σ/β
    Phase 3: Compare with β²/(4π²)

Copyright (c) 2026 Tracy McSheery
Licensed under the MIT License
"""

import sys
import os
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA
from qfd.Cl33 import (Multivector, N_COMPONENTS, SIGNATURE,
                       geometric_product_indices, basis_grade,
                       e0, e1, e2, e3, e4, e5, B_phase,
                       commutes_with_phase)


# =============================================================================
# PARAMETERS
# =============================================================================

beta = BETA
lam = beta
rho_0 = 1.0
a = 1.0
U_circ = np.sqrt(beta) / 2.0  # Circulation speed / c


# =============================================================================
# GROUND STATE
# =============================================================================

def ground_state(r):
    """Hill vortex ground state as Cl(3,3) multivector.

    ψ₀(r) = ρ(r)·1 + B(r)·e₄e₅

    The scalar part ρ(r) is the parabolic density.
    The bivector part B(r) = ρ(r) × U_circ is the circulation.
    """
    rho = rho_0 * max(0.0, 1.0 - (r/a)**2)
    B = rho * U_circ

    psi = Multivector()
    psi.components[0] = rho       # Scalar part
    # e4*e5 has index: bit 4 (=16) | bit 5 (=32) = 48
    psi.components[48] = B        # Bivector part (e4e5)
    return psi


def ground_state_density(r):
    """Scalar density ρ₀ = ⟨ψ̃₀ψ₀⟩₀ at radius r."""
    psi = ground_state(r)
    # For ψ = ρ + B·e₄e₅:
    # ψ̃ = ρ + B·e₅e₄ = ρ - B·e₄e₅  (reverse of bivector flips sign)
    # ψ̃ψ = (ρ - B·e₄e₅)(ρ + B·e₄e₅) = ρ² - B²(e₄e₅)²
    # (e₄e₅)² = e₄e₅e₄e₅ = -e₄e₄e₅e₅ = -(-1)(-1) = -1
    # So ψ̃ψ = ρ² + B²  (positive definite!)
    rho = psi.components[0]
    B = psi.components[48]
    return rho**2 + B**2


# =============================================================================
# POTENTIAL AND HESSIAN
# =============================================================================

def reverse_mv(psi):
    """Compute the reverse ψ̃ (reverses order of basis vectors in each blade)."""
    result = Multivector()
    for idx in range(N_COMPONENTS):
        grade = basis_grade(idx)
        # Reverse sign: (-1)^(k(k-1)/2) where k is grade
        sign = (-1) ** (grade * (grade - 1) // 2)
        result.components[idx] = sign * psi.components[idx]
    return result


def compute_density(psi):
    """Compute ρ = ⟨ψ̃ψ⟩₀ (scalar part of reverse times psi)."""
    psi_rev = reverse_mv(psi)
    product = psi_rev * psi
    return product.scalar_part()


def potential_hessian_at_r(r):
    """Compute the 64×64 potential Hessian matrix at radius r.

    The potential is V(ρ) where ρ = ⟨ψ̃ψ⟩₀.
    V(ρ) = -μ²ρ² + λρ⁴ (quartic Mexican hat)

    The second functional derivative w.r.t. ψ in direction η is:
        δ²V/δψ² · η = V''(ρ₀) × 4⟨ψ̃₀η + η̃ψ₀⟩₀ × ψ₀
                     + V'(ρ₀) × 2(η̃ψ₀ + ψ̃₀η)  [symmetric parts]

    Wait — let me be more careful. Let ψ = ψ₀ + εη.
    ρ(ψ) = ⟨(ψ₀+εη)˜(ψ₀+εη)⟩₀
          = ⟨ψ̃₀ψ₀⟩₀ + ε⟨ψ̃₀η + η̃ψ₀⟩₀ + ε²⟨η̃η⟩₀
          = ρ₀ + ε·δρ + ε²·δ²ρ

    where δρ = ⟨ψ̃₀η + η̃ψ₀⟩₀ and δ²ρ = ⟨η̃η⟩₀.

    V(ρ) = V(ρ₀) + V'(ρ₀)(εδρ + ε²δ²ρ) + ½V''(ρ₀)(εδρ)² + ...

    Second variation (coefficient of ε²):
        δ²V = V'(ρ₀)·δ²ρ + ½V''(ρ₀)·(δρ)²
            = V'(ρ₀)⟨η̃η⟩₀ + ½V''(ρ₀)⟨ψ̃₀η + η̃ψ₀⟩₀²

    This is quadratic in η, so it defines a matrix H_V such that:
        δ²V = η^T H_V η  (in the 64-component basis)

    For each pair of basis elements (e_A, e_B):
        H_V[A,B] = V'(ρ₀)⟨ẽ_A e_B⟩₀
                  + ½V''(ρ₀)⟨ψ̃₀e_A + ẽ_Aψ₀⟩₀ · ⟨ψ̃₀e_B + ẽ_Bψ₀⟩₀

    Here ẽ_A is the reverse of e_A.
    """
    psi0 = ground_state(r)
    psi0_rev = reverse_mv(psi0)
    rho = compute_density(psi0)

    # Mexican hat: V(ρ) = -μ²ρ² + λρ⁴
    # V'(ρ) = -2μ²ρ + 4λρ³
    # V''(ρ) = -2μ² + 12λρ²
    mu_sq = 2.0 * lam * rho_0**2
    Vp = -2.0 * mu_sq * rho + 4.0 * lam * rho**3
    Vpp = -2.0 * mu_sq + 12.0 * lam * rho**2

    H = np.zeros((N_COMPONENTS, N_COMPONENTS))

    # Precompute the "coupling vectors" for each basis element
    # c_A = ⟨ψ̃₀ e_A + ẽ_A ψ₀⟩₀ (scalar)
    coupling = np.zeros(N_COMPONENTS)
    for A in range(N_COMPONENTS):
        eA = Multivector.basis_element(A)
        eA_rev = reverse_mv(eA)

        # ψ̃₀ e_A
        prod1 = psi0_rev * eA
        # ẽ_A ψ₀
        prod2 = eA_rev * psi0

        coupling[A] = (prod1 + prod2).scalar_part()

    # H[A,B] = V'(ρ₀) × ⟨ẽ_A e_B⟩₀ + ½V''(ρ₀) × c_A × c_B
    for A in range(N_COMPONENTS):
        eA_rev = reverse_mv(Multivector.basis_element(A))
        for B in range(N_COMPONENTS):
            eB = Multivector.basis_element(B)

            # Term 1: V'(ρ₀) × ⟨ẽ_A e_B⟩₀
            prod_AB = eA_rev * eB
            inner = prod_AB.scalar_part()
            H[A, B] = Vp * inner + 0.5 * Vpp * coupling[A] * coupling[B]

    return H


# =============================================================================
# ANALYSIS
# =============================================================================

def classify_basis_elements():
    """Classify all 64 basis elements by grade and B_phase commutation."""
    classes = {}
    for idx in range(N_COMPONENTS):
        grade = basis_grade(idx)
        mv = Multivector.basis_element(idx)
        commutes = commutes_with_phase(mv)
        key = (grade, commutes)
        if key not in classes:
            classes[key] = []
        classes[key].append(idx)
    return classes


def main():
    W = 78
    ratio_target = beta**2 / (4 * np.pi**2)

    print()
    print("=" * W)
    print("  HESSIAN v5: Geometric Algebra Native (Cl(3,3), ALL REAL)")
    print("=" * W)
    print(f"  β = {beta:.10f}")
    print(f"  Target ratio σ/β = β²/(4π²) = {ratio_target:.6f}")
    print(f"  U_circ/c = √β/2 = {U_circ:.6f}")
    print(f"  N_COMPONENTS = {N_COMPONENTS} (2⁶ = 64)")

    # ===== Verify ground state =====
    print(f"\n{'GROUND STATE VERIFICATION':^{W}}")
    print("-" * W)

    psi0_center = ground_state(0.0)
    psi0_half = ground_state(0.5)
    psi0_surf = ground_state(a)

    print(f"  ψ₀(0):   scalar = {psi0_center.components[0]:.4f}, "
          f"bivector(e₄e₅) = {psi0_center.components[48]:.4f}")
    print(f"  ψ₀(a/2): scalar = {psi0_half.components[0]:.4f}, "
          f"bivector(e₄e₅) = {psi0_half.components[48]:.4f}")
    print(f"  ψ₀(a):   scalar = {psi0_surf.components[0]:.4f}, "
          f"bivector(e₄e₅) = {psi0_surf.components[48]:.4f}")

    rho_center = compute_density(psi0_center)
    rho_half = compute_density(psi0_half)
    print(f"  ρ(0) = ⟨ψ̃₀ψ₀⟩₀ = {rho_center:.4f}")
    print(f"  ρ(a/2) = {rho_half:.4f}")
    print(f"  (ρ includes both scalar² + bivector²)")

    # ===== Classify basis elements =====
    print(f"\n{'BASIS ELEMENT CLASSIFICATION':^{W}}")
    print("-" * W)

    classes = classify_basis_elements()
    for (grade, commutes), indices in sorted(classes.items()):
        label = "centralizer" if commutes else "anti-centralizer"
        print(f"  Grade {grade}, {label:20s}: {len(indices)} elements")

    # ===== Compute potential Hessian at several radii =====
    print(f"\n{'POTENTIAL HESSIAN (64×64) AT REPRESENTATIVE POINTS':^{W}}")
    print("-" * W)

    for r_val in [0.01, 0.3, 0.5, 0.8, 1.0, 1.5]:
        H = potential_hessian_at_r(r_val)

        # Check symmetry (should be symmetric since it's a real second derivative)
        asym = np.max(np.abs(H - H.T))

        # Eigenvalues
        evals = np.linalg.eigvalsh(H)
        n_pos = np.sum(evals > 1e-10)
        n_neg = np.sum(evals < -1e-10)
        n_zero = N_COMPONENTS - n_pos - n_neg

        print(f"  r={r_val:.2f}: ||H-H^T||={asym:.2e}, "
              f"rank(+)={n_pos}, rank(-)={n_neg}, rank(0)={n_zero}, "
              f"λ_min={evals[0]:.4f}, λ_max={evals[-1]:.4f}")

    # ===== Detailed eigenvalue analysis at r = 0 (center) =====
    print(f"\n{'EIGENVALUE SPECTRUM AT r = 0.01 (CENTER)':^{W}}")
    print("-" * W)

    H_center = potential_hessian_at_r(0.01)
    evals_center, evecs_center = np.linalg.eigh(H_center)

    # Identify which eigenvalues correspond to which grades
    print(f"  Eigenvalue spectrum (all 64):")
    for i, ev in enumerate(evals_center):
        if abs(ev) > 1e-8:
            # Identify the dominant grade of the eigenvector
            vec = evecs_center[:, i]
            grade_weights = {}
            for idx in range(N_COMPONENTS):
                g = basis_grade(idx)
                if g not in grade_weights:
                    grade_weights[g] = 0.0
                grade_weights[g] += vec[idx]**2
            dominant_grade = max(grade_weights, key=grade_weights.get)
            dom_weight = grade_weights[dominant_grade]

            # Check centralizer membership
            mv = Multivector(vec)
            in_cent = commutes_with_phase(mv)

            print(f"    λ_{i:2d} = {ev:12.4f}  "
                  f"grade {dominant_grade} ({dom_weight:.1%})  "
                  f"{'cent' if in_cent else 'anti'}")

    # ===== The key question: eigenvalue ratios by grade =====
    print(f"\n{'EIGENVALUE RATIOS BY GRADE SECTOR':^{W}}")
    print("-" * W)

    # At the center, group eigenvalues by the dominant grade of their eigenvectors
    grade_eigenvalues = {}
    for i, ev in enumerate(evals_center):
        vec = evecs_center[:, i]
        grade_weights = {}
        for idx in range(N_COMPONENTS):
            g = basis_grade(idx)
            if g not in grade_weights:
                grade_weights[g] = 0.0
            grade_weights[g] += vec[idx]**2
        dominant_grade = max(grade_weights, key=grade_weights.get)

        if dominant_grade not in grade_eigenvalues:
            grade_eigenvalues[dominant_grade] = []
        grade_eigenvalues[dominant_grade].append(ev)

    for grade in sorted(grade_eigenvalues.keys()):
        evs = grade_eigenvalues[grade]
        print(f"  Grade {grade}: {len(evs)} eigenvalues, "
              f"range [{min(evs):.4f}, {max(evs):.4f}]")

    # Ratio of shear (grade 2) to compression (grade 0) eigenvalues
    if 0 in grade_eigenvalues and 2 in grade_eigenvalues:
        ev0 = [e for e in grade_eigenvalues[0] if abs(e) > 1e-8]
        ev2 = [e for e in grade_eigenvalues[2] if abs(e) > 1e-8]
        if ev0 and ev2:
            # Use the most negative eigenvalue in each sector
            ev0_min = min(ev0)
            ev2_min = min(ev2)
            ratio = ev2_min / ev0_min if ev0_min != 0 else float('nan')
            dev = abs(ratio / ratio_target - 1) * 100 if ratio_target != 0 else float('nan')
            print(f"\n  Grade-0 min eigenvalue: {ev0_min:.6f}")
            print(f"  Grade-2 min eigenvalue: {ev2_min:.6f}")
            print(f"  Ratio grade-2/grade-0:  {ratio:.6f}")
            print(f"  Target β²/(4π²):        {ratio_target:.6f}")
            print(f"  Deviation:              {dev:.2f}%")

    # ===== Radial dependence =====
    print(f"\n{'RADIAL PROFILE OF KEY EIGENVALUES':^{W}}")
    print("-" * W)

    r_points = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]
    for r_val in r_points:
        H = potential_hessian_at_r(r_val)
        evals = np.linalg.eigvalsh(H)
        # Report min, max, and a few key eigenvalues
        print(f"  r={r_val:.2f}: λ_min={evals[0]:10.4f}, "
              f"λ_max={evals[-1]:10.4f}, "
              f"n_distinct={np.sum(np.abs(np.diff(evals)) > 1e-6)}")

    # Summary
    print(f"\n{'=' * W}")
    print(f"  SUMMARY")
    print(f"{'=' * W}")
    print(f"  This solver computes the potential part of the Hessian in the")
    print(f"  NATIVE Cl(3,3) multivector basis — 64×64 real matrix at each r.")
    print(f"  No complex numbers anywhere. The eigenvalue spectrum decomposes")
    print(f"  by grade (0,1,2,3,...,6) and centralizer membership.")
    print(f"")
    print(f"  NOTE: This is the POTENTIAL Hessian only. The kinetic part")
    print(f"  (-∇² acting on each of the 64 components) adds the standard")
    print(f"  radial eigenvalue problem for each internal eigenvector.")
    print(f"  The full computation combines internal (64×64) and radial (N×N).")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()

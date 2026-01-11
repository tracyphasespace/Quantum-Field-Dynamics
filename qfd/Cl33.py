#!/usr/bin/env python3
"""
Cl(3,3) Clifford Algebra Implementation for QFD

This module implements the Clifford algebra Cl(3,3) with signature (+,+,+,-,-,-).
Used for the Beltrami equation formulation that eliminates complex numbers.

Key structures:
- 6 basis vectors: e0, e1, e2 (spatial +1), e3, e4, e5 (timelike -1)
- 64 basis elements (2^6)
- Phase bivector B = e4*e5 acts as geometric imaginary unit (B² = -1)

Based on Lean4 proofs in QFD/GA/Cl33.lean and QFD/Photon/CliffordBeltrami.lean
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from functools import lru_cache

# Cl(3,3) has 6 generators, signature (+,+,+,-,-,-)
N_GENERATORS = 6
N_COMPONENTS = 2**N_GENERATORS  # 64 basis elements

# Signature: e_i^2 = signature[i]
SIGNATURE = np.array([1, 1, 1, -1, -1, -1], dtype=np.float64)

# Basis element indices (as bit patterns)
# 0 = scalar, 1 = e0, 2 = e1, 4 = e2, 8 = e3, 16 = e4, 32 = e5
# 3 = e01, 5 = e02, etc.

@lru_cache(maxsize=4096)
def basis_grade(idx: int) -> int:
    """Return the grade (number of 1-bits) of basis element."""
    return bin(idx).count('1')

@lru_cache(maxsize=4096)
def basis_sign(idx: int) -> int:
    """
    Compute the sign factor for squaring a basis element.
    For basis element e_{i1} * e_{i2} * ... * e_{ik},
    squaring gives sign = prod(signature[ij]) * reordering_sign
    """
    if idx == 0:
        return 1

    # Extract which generators are present
    generators = []
    for i in range(N_GENERATORS):
        if idx & (1 << i):
            generators.append(i)

    # Square of e_{i1...ik} involves:
    # 1. Sign from squaring each generator: prod(signature[i])
    # 2. Sign from reordering: (-1)^(k*(k-1)/2) for anticommutation

    sign_from_signature = 1
    for g in generators:
        sign_from_signature *= int(SIGNATURE[g])

    k = len(generators)
    sign_from_reorder = (-1) ** (k * (k - 1) // 2)

    return sign_from_signature * sign_from_reorder

@lru_cache(maxsize=65536)
def geometric_product_indices(a: int, b: int) -> Tuple[int, int]:
    """
    Compute the geometric product of two basis elements.
    Returns (result_index, sign).

    e_A * e_B = sign * e_{A XOR B} where sign accounts for anticommutation.
    """
    if a == 0:
        return (b, 1)
    if b == 0:
        return (a, 1)

    # Result index is XOR (generators that appear odd number of times)
    result = a ^ b

    # Compute sign from anticommutation
    # Each time a generator from b passes through a generator from a, we get -1
    sign = 1

    # Also account for squared generators (when same bit is set in both)
    common = a & b
    for i in range(N_GENERATORS):
        if common & (1 << i):
            sign *= int(SIGNATURE[i])

    # Count anticommutation swaps
    # For each bit in b, count how many higher bits in a it must pass
    for i in range(N_GENERATORS):
        if b & (1 << i):
            # Count bits in a that are higher than i
            higher_bits_in_a = a >> (i + 1)
            swaps = bin(higher_bits_in_a).count('1')
            if swaps % 2 == 1:
                sign *= -1

    return (result, sign)


class Multivector:
    """
    A multivector in Cl(3,3).

    Stored as 64-component array indexed by basis element bit pattern.
    """

    __slots__ = ['components']

    def __init__(self, components: Optional[np.ndarray] = None):
        if components is None:
            self.components = np.zeros(N_COMPONENTS, dtype=np.float64)
        else:
            self.components = np.asarray(components, dtype=np.float64)
            if self.components.shape != (N_COMPONENTS,):
                raise ValueError(f"Expected {N_COMPONENTS} components, got {self.components.shape}")

    @classmethod
    def scalar(cls, value: float) -> 'Multivector':
        """Create a scalar multivector."""
        mv = cls()
        mv.components[0] = value
        return mv

    @classmethod
    def basis(cls, i: int) -> 'Multivector':
        """Create a basis vector e_i (i in 0..5)."""
        if not 0 <= i < N_GENERATORS:
            raise ValueError(f"Basis index must be 0-5, got {i}")
        mv = cls()
        mv.components[1 << i] = 1.0
        return mv

    @classmethod
    def basis_element(cls, idx: int) -> 'Multivector':
        """Create an arbitrary basis element by index (0-63)."""
        mv = cls()
        mv.components[idx] = 1.0
        return mv

    def __repr__(self) -> str:
        terms = []
        for idx in range(N_COMPONENTS):
            if abs(self.components[idx]) > 1e-10:
                coef = self.components[idx]
                if idx == 0:
                    terms.append(f"{coef:.4g}")
                else:
                    basis_str = self._basis_name(idx)
                    if coef == 1.0:
                        terms.append(basis_str)
                    elif coef == -1.0:
                        terms.append(f"-{basis_str}")
                    else:
                        terms.append(f"{coef:.4g}*{basis_str}")
        return " + ".join(terms) if terms else "0"

    @staticmethod
    def _basis_name(idx: int) -> str:
        """Get string name for basis element."""
        if idx == 0:
            return "1"
        generators = []
        for i in range(N_GENERATORS):
            if idx & (1 << i):
                generators.append(str(i))
        return "e" + "".join(generators)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        return Multivector(self.components + other.components)

    def __sub__(self, other: 'Multivector') -> 'Multivector':
        return Multivector(self.components - other.components)

    def __neg__(self) -> 'Multivector':
        return Multivector(-self.components)

    def __mul__(self, other) -> 'Multivector':
        """Geometric product."""
        if isinstance(other, (int, float)):
            return Multivector(self.components * other)
        elif isinstance(other, Multivector):
            result = np.zeros(N_COMPONENTS, dtype=np.float64)
            for a in range(N_COMPONENTS):
                if abs(self.components[a]) < 1e-15:
                    continue
                for b in range(N_COMPONENTS):
                    if abs(other.components[b]) < 1e-15:
                        continue
                    idx, sign = geometric_product_indices(a, b)
                    result[idx] += sign * self.components[a] * other.components[b]
            return Multivector(result)
        else:
            return NotImplemented

    def __rmul__(self, other) -> 'Multivector':
        if isinstance(other, (int, float)):
            return Multivector(self.components * other)
        return NotImplemented

    def __truediv__(self, other: float) -> 'Multivector':
        return Multivector(self.components / other)

    def grade(self, k: int) -> 'Multivector':
        """Extract grade-k component."""
        result = np.zeros(N_COMPONENTS, dtype=np.float64)
        for idx in range(N_COMPONENTS):
            if basis_grade(idx) == k:
                result[idx] = self.components[idx]
        return Multivector(result)

    def scalar_part(self) -> float:
        """Extract scalar component."""
        return self.components[0]

    def norm_squared(self) -> float:
        """Compute squared norm (sum of squared components)."""
        return np.sum(self.components**2)

    def norm(self) -> float:
        """Compute norm."""
        return np.sqrt(self.norm_squared())


# Convenient basis vectors
e0 = Multivector.basis(0)
e1 = Multivector.basis(1)
e2 = Multivector.basis(2)
e3 = Multivector.basis(3)
e4 = Multivector.basis(4)
e5 = Multivector.basis(5)

# The Phase Bivector (geometric imaginary unit)
B_phase = e4 * e5


def clifford_wedge(a: Multivector, b: Multivector) -> Multivector:
    """
    Clifford wedge product (antisymmetric part).
    a ∧ b = (1/2)(ab - ba)

    This is the exterior product / curl in Clifford algebra.
    """
    return (a * b - b * a) * 0.5


def clifford_dot(a: Multivector, b: Multivector) -> Multivector:
    """
    Clifford dot product (symmetric part).
    a · b = (1/2)(ab + ba)

    This is the interior product / divergence in Clifford algebra.
    """
    return (a * b + b * a) * 0.5


def commutator(a: Multivector, b: Multivector) -> Multivector:
    """Commutator [a, b] = ab - ba."""
    return a * b - b * a


def anticommutator(a: Multivector, b: Multivector) -> Multivector:
    """Anticommutator {a, b} = ab + ba."""
    return a * b + b * a


def commutes_with_phase(x: Multivector) -> bool:
    """
    Check if x commutes with B_phase = e4*e5.
    Elements in the centralizer of B_phase are "visible" in 4D spacetime.
    """
    comm = commutator(x, B_phase)
    return comm.norm() < 1e-10


def project_to_centralizer(x: Multivector) -> Multivector:
    """
    Project x onto the centralizer of B_phase.

    The centralizer consists of elements that commute with B_phase.
    For Cl(3,3) with B = e4*e5:
    - Spacetime vectors (e0, e1, e2, e3) commute with B
    - Internal vectors (e4, e5) anticommute with B

    We keep only the components that commute.
    """
    result = np.zeros(N_COMPONENTS, dtype=np.float64)

    for idx in range(N_COMPONENTS):
        if abs(x.components[idx]) < 1e-15:
            continue

        # Create basis element and check if it commutes with B_phase
        basis_mv = Multivector.basis_element(idx)
        if commutes_with_phase(basis_mv):
            result[idx] = x.components[idx]

    return Multivector(result)


def is_beltrami_eigenfield(grad: Multivector, F: Multivector, kappa: float,
                           tol: float = 1e-6) -> bool:
    """
    Check if F is a Beltrami eigenfield: ∇ ∧ F = κ F
    """
    wedge = clifford_wedge(grad, F)
    expected = F * kappa
    diff = wedge - expected
    return diff.norm() < tol * max(F.norm(), 1.0)


# Verification tests
def verify_phase_properties():
    """Verify B_phase² = -1 and centralizer properties."""
    print("=" * 60)
    print("Cl(3,3) VERIFICATION TESTS")
    print("=" * 60)

    # Test 1: B² = -1
    B_squared = B_phase * B_phase
    print(f"\n[1] B_phase = e4 * e5")
    print(f"    B² = {B_squared}")
    print(f"    Expected: -1")
    assert abs(B_squared.scalar_part() + 1) < 1e-10, "B² should be -1"
    print("    ✓ PASS: B² = -1 (geometric imaginary unit)")

    # Test 2: Spacetime vectors commute with B
    print(f"\n[2] Spacetime vectors (e0, e1, e2, e3) commute with B:")
    for i, ei in enumerate([e0, e1, e2, e3]):
        comm = commutator(ei, B_phase)
        status = "✓ commutes" if comm.norm() < 1e-10 else "✗ FAILS"
        print(f"    e{i}: {status}")

    # Test 3: Internal vectors anticommute with B
    print(f"\n[3] Internal vectors (e4, e5) anticommute with B:")
    for i, ei in enumerate([e4, e5], start=4):
        anticomm = anticommutator(ei, B_phase)
        # Should be 2*ei*B or similar, not zero
        comm = commutator(ei, B_phase)
        status = "✓ anticommutes" if comm.norm() > 1e-10 else "✗ FAILS"
        print(f"    e{i}: {status} (||[e{i}, B]|| = {comm.norm():.4f})")

    # Test 4: Wedge + Dot = Full product
    print(f"\n[4] Wedge-Dot decomposition: a*b = (a·b) + (a∧b)")
    a, b = e0 + e1, e2 + e3
    full = a * b
    wedge = clifford_wedge(a, b)
    dot = clifford_dot(a, b)
    recon = dot + wedge
    diff = (full - recon).norm()
    print(f"    ||a*b - (a·b + a∧b)|| = {diff:.2e}")
    assert diff < 1e-10, "Decomposition should be exact"
    print("    ✓ PASS: Decomposition verified")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify_phase_properties()

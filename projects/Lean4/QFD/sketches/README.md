# Proof Sketches - NOT Formally Verified

**⚠️ WARNING: These files contain `sorry` placeholders and are NOT complete proofs.**

## Status

These files are **proof sketches** - they demonstrate the structure and strategy for formal proofs, but contain gaps filled with `sorry` tactics. They should **NOT** be cited as "formally verified" in publications.

## Contents

1. **AdjointStability.lean** (2 `sorry` placeholders)
   - Goal: Prove QFD adjoint guarantees positive-definite kinetic energy
   - Missing: Blade square normalization lemmas
   - Status: Type-correct scaffold, not verified

2. **SpacetimeEmergence.lean** (3 `sorry` placeholders)
   - Goal: Prove centralizer of B = e₄ ∧ e₅ is Cl(3,1)
   - Missing: Clifford algebra manipulation proofs
   - Status: Type-correct scaffold, not verified

3. **BivectorClasses.lean** (5 `sorry` placeholders)
   - Goal: Prove bivector trichotomy (rotors vs boosts)
   - Missing: Bivector algebra lemmas
   - Status: Type-correct scaffold, not verified

## What These Are Good For

✅ **Design documents** for future complete proofs
✅ **Type signatures** showing what needs to be proven
✅ **Proof strategies** documented in comments
✅ **Learning resource** for Lean 4 proof structure

## What These Are NOT

❌ **NOT formal proofs** (they compile, but prove nothing)
❌ **NOT publication-ready** (misleading to call them "verified")
❌ **NOT mathematically certain** (gaps could contain errors)

## Path to Completion

To convert these to rigorous proofs:

1. **AdjointStability.lean** (~40 hours)
   - Prove `blade_square I ∈ {-1, +1}` for all basis blades
   - Construct sum-of-squares decomposition
   - Verify using Mathlib real number lemmas

2. **SpacetimeEmergence.lean** (~50 hours)
   - Derive Clifford product rules from quadratic form
   - Prove associativity chains explicitly
   - Match approach used in EmergentAlgebra_Heavy.lean

3. **BivectorClasses.lean** (~60 hours)
   - Import or prove bivector square formula
   - Construct topological connectivity arguments
   - Verify rotor/boost distinction

## For Publication

**DO NOT reference these files in journal submissions.**

Instead, cite:
- `QFD/EmergentAlgebra_Heavy.lean` - Complete proof (0 `sorry`)

See `/LEAN_PROOF_STATUS.md` for publication guidelines.

---

**Last Updated:** December 21, 2025
**Status:** Proof intent only - NOT VERIFIED

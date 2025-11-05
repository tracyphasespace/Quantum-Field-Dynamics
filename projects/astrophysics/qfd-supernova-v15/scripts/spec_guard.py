#!/usr/bin/env python3
"""
Spec guard: Enforce V15 spec compliance by scanning for forbidden patterns.

V15 spec mandates α-space only; no ΛCDM triplet (1+z) terms in source code.
Docs and tests are allowed to reference (1+z) for explanation.
"""
import sys
import pathlib
import re

def main():
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    bad = []

    for p in ROOT.rglob("*.py"):
        # Skip docs and tests (allowed to reference (1+z) for explanation)
        if "docs" in p.parts or "tests" in p.parts:
            continue

        # Skip scripts directory itself (this file is in scripts/)
        if "scripts" in p.parts and p.name == "spec_guard.py":
            continue

        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Check for (1+z) or (1 + z) patterns
        if re.search(r"\(1\s*\+\s*z\)", txt):
            bad.append(str(p.relative_to(ROOT)))

    if bad:
        print("❌ Spec guard FAILED: Forbidden '(1+z)' pattern found in:")
        for f in bad:
            print(f"   - {f}")
        print()
        print("V15 spec mandates α-space only. Use α_pred(z; k_J, η′, ξ) instead.")
        print("If you need (1+z) for docs/explanation, move it to docs/ or tests/.")
        sys.exit(1)

    print("✓ Spec guard OK: No forbidden (1+z) patterns in source code")
    return 0

if __name__ == "__main__":
    sys.exit(main())

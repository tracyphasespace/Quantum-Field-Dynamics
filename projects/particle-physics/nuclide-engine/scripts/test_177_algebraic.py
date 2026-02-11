#!/usr/bin/env python3
"""
Test: Does N = 177 Connect to QFD Constants?
=============================================

The diameter ceiling test found N_max = 177 for Z = 114-118.
This script exhaustively tests whether 177 (or related numbers like
A_max = 295, Z_ceiling = 118, N/Z = 1.50) can be expressed in terms
of the QFD constants alpha and beta.

Constants:
  alpha = 1/137.036  (fine structure, CODATA)
  beta  = 3.043233053  (Golden Loop: 1/alpha = 2*pi^2*(e^beta/beta) + 1)

Derived constants tested:
  1/alpha = 137.036
  beta^2 = 9.261
  e^beta = 20.969
  pi, e, ln(2), sqrt(2), sqrt(3), phi (golden ratio)
  and all products/quotients up to 3 operations deep.

Provenance: QFD_DERIVED (constants from Golden Loop)
"""

import math
import itertools
import sys
import os

# Import QFD shared constants
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA

# ═══════════════════════════════════════════════════════════════
# QFD Constants (imported from qfd/shared_constants.py)
# ═══════════════════════════════════════════════════════════════
PI = math.pi
E_NUM = math.e
PHI = (1 + math.sqrt(5)) / 2  # golden ratio

# Derived
INV_ALPHA = 1.0 / ALPHA  # 137.036
BETA_SQ = BETA ** 2        # 9.261
EXP_BETA = math.exp(BETA)  # 20.969
LN2 = math.log(2)
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)

# ═══════════════════════════════════════════════════════════════
# Target numbers to match
# ═══════════════════════════════════════════════════════════════
TARGETS = {
    'N_max = 177': 177,
    'A_max = 295': 295,
    'Z_ceiling = 118': 118,
    'N/Z = 1.500': 1.500,
    'A/Z = 2.500': 2.500,
    'N_max/Z*(295) = 177/112.6': 177 / 112.6,
    'Core slope = 0.670': 0.670,  # dN_excess/dZ average
    'Core slope = 2/3': 2.0/3.0,
}


def test_algebraic():
    print("=" * 80)
    print("  ALGEBRAIC TEST: Does N = 177 Connect to QFD Constants?")
    print("=" * 80)

    # Build a library of simple expressions from (alpha, beta, pi, e)
    # Level 0: base constants
    base = {
        'alpha': ALPHA,
        '1/alpha': INV_ALPHA,
        'beta': BETA,
        'beta^2': BETA_SQ,
        'beta^3': BETA ** 3,
        'e^beta': EXP_BETA,
        'pi': PI,
        'e': E_NUM,
        'pi^2': PI ** 2,
        'pi^3': PI ** 3,
        'e^2': E_NUM ** 2,
        'ln2': LN2,
        'sqrt2': SQRT2,
        'sqrt3': SQRT3,
        'phi': PHI,
        '2': 2.0,
        '3': 3.0,
        '4': 4.0,
        '1': 1.0,
    }

    # Level 1: products and quotients of base pairs
    level1 = {}
    base_items = list(base.items())
    for (n1, v1), (n2, v2) in itertools.product(base_items, base_items):
        if n1 == n2:
            continue
        # Product
        name = f"{n1}*{n2}"
        val = v1 * v2
        if 0.001 < abs(val) < 1e6:
            level1[name] = val
        # Quotient
        if abs(v2) > 1e-10:
            name = f"{n1}/{n2}"
            val = v1 / v2
            if 0.001 < abs(val) < 1e6:
                level1[name] = val

    # Level 2: base * level1 and base + level1
    level2 = {}
    for (n1, v1) in base_items:
        for (n2, v2) in list(level1.items())[:500]:  # limit combinatorics
            # Product
            name = f"{n1}*({n2})"
            val = v1 * v2
            if 0.001 < abs(val) < 1e6:
                level2[name] = val
            # Sum
            name = f"{n1}+({n2})"
            val = v1 + v2
            if 0.001 < abs(val) < 1e6:
                level2[name] = val
            # Difference
            name = f"{n1}-({n2})"
            val = v1 - v2
            if 0.001 < abs(val) < 1e6:
                level2[name] = val

    # Also try: integer * base, integer * level1
    for n_int in range(1, 30):
        for (n1, v1) in base_items:
            name = f"{n_int}*{n1}"
            val = n_int * v1
            if name not in level1 and 0.001 < abs(val) < 1e6:
                level1[name] = val
        for (n1, v1) in list(level1.items())[:200]:
            name = f"{n_int}*({n1})"
            val = n_int * v1
            if name not in level2 and 0.001 < abs(val) < 1e6:
                level2[name] = val

    # Also try powers: x^(1/2), x^(1/3), x^(2/3)
    for (n1, v1) in base_items:
        if v1 > 0:
            for exp_name, exp_val in [('1/2', 0.5), ('1/3', 1/3), ('2/3', 2/3),
                                       ('3/2', 1.5), ('4/3', 4/3)]:
                name = f"{n1}^({exp_name})"
                val = v1 ** exp_val
                if 0.001 < abs(val) < 1e6:
                    level1[name] = val

    # Combine all expressions
    all_expr = {}
    all_expr.update(base)
    all_expr.update(level1)
    all_expr.update(level2)

    print(f"\n  Testing {len(all_expr)} algebraic expressions against {len(TARGETS)} targets")

    # Special hand-crafted expressions to test
    special = {
        # N = 177 candidates
        '2*pi^2*(e^beta/beta)': 2 * PI**2 * (EXP_BETA / BETA),  # = 1/alpha - 1 = 136.036
        '1/alpha + beta^2*pi/e': INV_ALPHA + BETA_SQ * PI / E_NUM,
        'beta^2 * (1/alpha)^(1/2)': BETA_SQ * INV_ALPHA**0.5,
        '2*beta^3/alpha': 2 * BETA**3 / ALPHA,
        'e^beta * beta * pi / 4': EXP_BETA * BETA * PI / 4,
        '(1/alpha)*(1+beta/pi)': INV_ALPHA * (1 + BETA/PI),
        'beta * (1/alpha - beta^2)': BETA * (INV_ALPHA - BETA_SQ),
        '1/alpha + 2*beta^2*pi': INV_ALPHA + 2 * BETA_SQ * PI,
        '(1/alpha)^(4/3)': INV_ALPHA ** (4/3),
        '1/alpha + beta*pi*e': INV_ALPHA + BETA * PI * E_NUM,
        'pi^2 * e^beta / beta': PI**2 * EXP_BETA / BETA,  # half of 1/alpha formula
        '2*pi^3*beta': 2 * PI**3 * BETA,
        'e^beta * beta^2 / pi': EXP_BETA * BETA_SQ / PI,
        '(1/alpha) + beta^2*4': INV_ALPHA + BETA_SQ * 4,
        '(1/alpha) + 4*pi*beta': INV_ALPHA + 4 * PI * BETA,
        '(1/alpha) + beta*(pi+e)^2': INV_ALPHA + BETA * (PI + E_NUM)**2,
        'e^beta * (pi + beta)': EXP_BETA * (PI + BETA),
        '4 * pi * beta * e': 4 * PI * BETA * E_NUM,
        'e^beta / alpha^(2/3)': EXP_BETA / ALPHA**(2/3),
        'beta^4 * 2': BETA**4 * 2,
        'pi^2 * (1/alpha)^(1/2)': PI**2 * INV_ALPHA**0.5,
        'e * (1/alpha)^(1+alpha)': E_NUM * INV_ALPHA**(1+ALPHA),
        '(1/alpha + 1)*(1 + beta/pi)': (INV_ALPHA + 1) * (1 + BETA/PI),
        '(1/alpha)*(4/pi + alpha*beta)': INV_ALPHA * (4/PI + ALPHA*BETA),
        # 295 candidates
        '2/alpha + pi*beta': 2/ALPHA + PI * BETA,
        'e^beta * (2*pi + beta)': EXP_BETA * (2*PI + BETA),
        'pi*(1/alpha - beta)': PI * (INV_ALPHA - BETA),
        '(1/alpha)*e^(pi*alpha)': INV_ALPHA * math.exp(PI * ALPHA),
        # 118 candidates
        '1/alpha - pi*beta/e': INV_ALPHA - PI * BETA / E_NUM,
        'beta^3 * 4 + pi': BETA**3 * 4 + PI,
        'e^beta * beta / pi': EXP_BETA * BETA / PI,
        'pi^2 * beta * 4 / pi': PI * BETA * 4,
        '4*beta*(pi + 1)': 4 * BETA * (PI + 1),
        '2*pi*beta*e': 2 * PI * BETA * E_NUM,
        # N/Z = 1.5 candidates
        '3/2': 1.5,
        'beta/2': BETA / 2,
        'pi/2': PI / 2,
        'e/2 - alpha': E_NUM / 2 - ALPHA,
        'beta/e + alpha': BETA / E_NUM + ALPHA,
        '1 + 1/e': 1 + 1/E_NUM,
        # 2/3 candidates
        '2/(beta+alpha)': 2 / (BETA + ALPHA),
        'pi/(beta*e/2 + 1)': PI / (BETA * E_NUM / 2 + 1),
        '1 - 1/beta': 1 - 1/BETA,
        '2*alpha*pi^2': 2 * ALPHA * PI**2,
        'beta/(beta + pi/2)': BETA / (BETA + PI/2),
        # Related
        '(1/alpha)*beta/e^(1/2)': INV_ALPHA * BETA / E_NUM**0.5,
    }
    all_expr.update(special)

    # Now test each target
    for target_name, target_val in TARGETS.items():
        print(f"\n{'─'*70}")
        print(f"  TARGET: {target_name} = {target_val}")
        print(f"{'─'*70}")

        # Find best matches
        matches = []
        for expr_name, expr_val in all_expr.items():
            if abs(expr_val) < 1e-10:
                continue
            pct_err = abs(expr_val - target_val) / abs(target_val) * 100
            if pct_err < 5.0:  # within 5%
                matches.append((expr_name, expr_val, pct_err))

        matches.sort(key=lambda x: x[2])

        if matches:
            print(f"\n  {'Expression':>45s} {'Value':>12s} {'Error':>8s}")
            print(f"  {'-'*68}")
            for name, val, err in matches[:20]:
                marker = " ***" if err < 0.5 else " **" if err < 1.0 else " *" if err < 2.0 else ""
                print(f"  {name:>45s} {val:>12.4f} {err:>7.2f}%{marker}")
        else:
            print(f"  No matches within 5%")

    # ═══════════════════════════════════════════════════════════════
    # Special focus: 177 as related to 1/alpha = 137.036
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  SPECIAL FOCUS: Relationship between N=177 and 1/alpha=137.036")
    print("=" * 80)

    ratio_177_137 = 177 / INV_ALPHA
    diff_177_137 = 177 - INV_ALPHA

    print(f"\n  177 / (1/alpha)  = {ratio_177_137:.6f}")
    print(f"  177 - (1/alpha)  = {diff_177_137:.4f}")
    print(f"  177 - 137        = 40")
    print(f"  177 + 118        = 295  (= A_max, trivially)")
    print(f"  177 / 118        = {177/118:.6f}")
    print(f"  177 / 3          = 59")
    print(f"  177 mod 4        = {177 % 4}")
    print(f"  177 mod beta     = {177 % BETA:.4f}")
    print(f"  177 / pi^2       = {177 / PI**2:.4f}")
    print(f"  177 / beta^2     = {177 / BETA_SQ:.4f}")

    # Test ratio candidates
    print(f"\n  Testing 177/(1/alpha) = {ratio_177_137:.6f}:")
    ratio_candidates = [
        (1 + BETA / PI, '1 + beta/pi'),
        (1 + 1 / E_NUM, '1 + 1/e'),
        (4 / PI, '4/pi'),
        (PI / E_NUM, 'pi/e'),
        (BETA / E_NUM, 'beta/e'),
        (E_NUM / BETA, 'e/beta'),
        (1 + ALPHA * PI * BETA, '1 + alpha*pi*beta'),
        (1 + 2 * PI * ALPHA, '1 + 2*pi*alpha'),
        (1 + PI**2 * ALPHA, '1 + pi^2*alpha'),
        (PHI, 'phi'),
        (SQRT2, 'sqrt(2)'),
        (math.log(BETA), 'ln(beta)'),
        ((PI + BETA) / PI, '(pi+beta)/pi'),
        (1 + BETA**2 / (4 * PI * E_NUM), '1 + beta^2/(4*pi*e)'),
        (1 + 4 / (PI * BETA), '1 + 4/(pi*beta)'),
    ]

    print(f"  {'Expression':>30s} {'Value':>10s} {'Error':>8s}")
    print(f"  {'-'*52}")
    for val, name in sorted(ratio_candidates, key=lambda x: abs(x[0] - ratio_177_137)):
        err = abs(val - ratio_177_137) / ratio_177_137 * 100
        marker = " ***" if err < 0.5 else " **" if err < 1.0 else ""
        print(f"  {name:>30s} {val:>10.6f} {err:>7.2f}%{marker}")

    # Test difference candidates
    print(f"\n  Testing 177 - (1/alpha) = {diff_177_137:.4f}:")
    diff_candidates = [
        (4 * PI * BETA, '4*pi*beta'),
        (BETA * PI * E_NUM, 'beta*pi*e'),
        (2 * PI * E_NUM, '2*pi*e'),
        (4 * BETA**2, '4*beta^2'),
        (PI**2 * 4, '4*pi^2'),
        (E_NUM**3 + 1, 'e^3 + 1'),
        (2 * E_NUM**2 * BETA / PI, '2*e^2*beta/pi'),
        (BETA**3 / PI * 4, '4*beta^3/pi'),
        (EXP_BETA * 2, '2*e^beta'),
        (PI**2 * BETA + E_NUM, 'pi^2*beta + e'),
        (10 * PI, '10*pi'),
        (10 * E_NUM + 2 * PI, '10*e + 2*pi'),
        (BETA**2 * PI + PI, 'beta^2*pi + pi'),
        (8 * BETA + PI, '8*beta + pi'),
    ]

    print(f"  {'Expression':>30s} {'Value':>10s} {'Error':>8s}")
    print(f"  {'-'*52}")
    for val, name in sorted(diff_candidates, key=lambda x: abs(x[0] - diff_177_137)):
        err = abs(val - diff_177_137) / abs(diff_177_137) * 100
        marker = " ***" if err < 0.5 else " **" if err < 1.0 else ""
        print(f"  {name:>30s} {val:>10.4f} {err:>7.2f}%{marker}")

    # ═══════════════════════════════════════════════════════════════
    # Focus: 177 as integer close to specific expressions
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  INTEGER PROXIMITY: What expression gives exactly 177?")
    print("=" * 80)

    # Try: floor/ceil/round of various expressions
    expressions = [
        ('round(1/alpha + 4*beta^2)', round(INV_ALPHA + 4 * BETA_SQ)),
        ('round(1/alpha + beta*pi*e/2)', round(INV_ALPHA + BETA * PI * E_NUM / 2)),
        ('round(e^beta * (pi+beta))', round(EXP_BETA * (PI + BETA))),
        ('round(2*pi^3*beta)', round(2 * PI**3 * BETA)),
        ('round(4*pi*beta*e)', round(4 * PI * BETA * E_NUM)),
        ('round(beta^4*2)', round(BETA**4 * 2)),
        ('round(pi^2*sqrt(1/alpha))', round(PI**2 * INV_ALPHA**0.5)),
        ('round(1/alpha * beta/e^(1/3))', round(INV_ALPHA * BETA / E_NUM**(1/3))),
        ('round(1/alpha * (1+beta/pi))', round(INV_ALPHA * (1 + BETA/PI))),
        ('round(1/alpha * 4/pi)', round(INV_ALPHA * 4/PI)),
        ('round((1/alpha)^(4/3))', round(INV_ALPHA ** (4/3))),
        ('round(e^beta*beta^2/pi)', round(EXP_BETA * BETA_SQ / PI)),
        ('round(1/alpha + beta^2*4)', round(INV_ALPHA + BETA_SQ * 4)),
        ('round(1/alpha + beta^2*pi)', round(INV_ALPHA + BETA_SQ * PI)),
        ('round(1/alpha + e^beta/pi)', round(INV_ALPHA + EXP_BETA / PI)),
        ('round(beta^2*(1/alpha)^(1/3))', round(BETA_SQ * INV_ALPHA**(1/3))),
        ('round(e^(beta+1))', round(math.exp(BETA + 1))),
        ('round(1/alpha + 4*pi*beta)', round(INV_ALPHA + 4 * PI * BETA)),
        ('round(1/alpha + 8*beta+pi)', round(INV_ALPHA + 8*BETA + PI)),
        ('round(1/alpha + 2*e^2)', round(INV_ALPHA + 2 * E_NUM**2)),
        ('round(1/alpha + e^3)', round(INV_ALPHA + E_NUM**3)),
        ('round(pi*beta*(1/alpha)^(1/3))', round(PI * BETA * INV_ALPHA**(1/3))),
        ('round(2*(1/alpha)*(1-1/(2*beta)))', round(2 * INV_ALPHA * (1 - 1/(2*BETA)))),
    ]

    print(f"\n  {'Expression':>45s} {'Value':>8s} {'= 177?':>8s}")
    print(f"  {'-'*65}")
    hits = []
    for name, val in expressions:
        match = "YES" if val == 177 else ""
        if val == 177:
            hits.append(name)
        if abs(val - 177) <= 5:
            print(f"  {name:>45s} {val:>8d} {match:>8s}")

    if hits:
        print(f"\n  HITS: {hits}")

    # ═══════════════════════════════════════════════════════════════
    # Brute force: N = a * (1/alpha) + b, solve for b
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  DECOMPOSITION: 177 = f(1/alpha) + correction")
    print("=" * 80)

    # 177 = 1 * (1/alpha) + 39.964 → what is 39.964?
    correction = 177 - INV_ALPHA
    print(f"\n  177 = (1/alpha) + {correction:.4f}")
    print(f"  Correction {correction:.4f} candidates:")

    corr_candidates = [
        (4 * PI * BETA, '4*pi*beta', 4 * PI * BETA),
        (BETA**2 * PI + PI, '(beta^2+1)*pi', BETA_SQ * PI + PI),
        (EXP_BETA * 2, '2*exp(beta)', EXP_BETA * 2),
        (PI**2 * 4, '4*pi^2', 4 * PI**2),
        (4 * BETA**2, '4*beta^2', 4 * BETA_SQ),
        (E_NUM * PI * BETA, 'e*pi*beta', E_NUM * PI * BETA),
        (10 * PI, '10*pi', 10 * PI),
        (BETA**2 * PI, 'beta^2*pi', BETA_SQ * PI),
        (E_NUM**3 + E_NUM, 'e^3 + e', E_NUM**3 + E_NUM),
        (PI**2 * BETA + BETA, 'pi^2*beta + beta', PI**2 * BETA + BETA),
        (8 * BETA + PI, '8*beta + pi', 8 * BETA + PI),
        (PI * (4 + BETA), 'pi*(4+beta)', PI * (4 + BETA)),
        (BETA * (PI + E_NUM), 'beta*(pi+e)', BETA * (PI + E_NUM)),
        (4 * E_NUM * BETA, '4*e*beta', 4 * E_NUM * BETA),
        (2 * PI * E_NUM, '2*pi*e', 2 * PI * E_NUM),
        (13 * PI, '13*pi', 13 * PI),
        (13 * BETA, '13*beta', 13 * BETA),
        (7 * BETA + PI*E_NUM, '7*beta + pi*e', 7 * BETA + PI * E_NUM),
    ]

    print(f"  {'Expression':>25s} {'Value':>10s} {'Error':>8s}")
    print(f"  {'-'*47}")
    for val, name, _ in sorted(corr_candidates, key=lambda x: abs(x[0] - correction)):
        err = abs(val - correction) / abs(correction) * 100
        marker = " ***" if err < 0.5 else " **" if err < 1.0 else ""
        print(f"  {name:>25s} {val:>10.4f} {err:>7.2f}%{marker}")

    # ═══════════════════════════════════════════════════════════════
    # Also try: 177 = n * beta^2 for some n
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  FACTORIZATION: 177 in terms of beta^2, pi^2, etc.")
    print("=" * 80)

    factorizations = [
        (177 / BETA_SQ, 'beta^2'),
        (177 / PI**2, 'pi^2'),
        (177 / E_NUM**2, 'e^2'),
        (177 / (PI * BETA), 'pi*beta'),
        (177 / (PI * E_NUM), 'pi*e'),
        (177 / (BETA * E_NUM), 'beta*e'),
        (177 / EXP_BETA, 'exp(beta)'),
        (177 / INV_ALPHA, '1/alpha'),
        (177 / (2 * PI**2), '2*pi^2'),
        (177 / (BETA**3), 'beta^3'),
        (177 / (4 * PI), '4*pi'),
        (177 / (2 * PI * E_NUM), '2*pi*e'),
        (177 / (4 * BETA), '4*beta'),
    ]

    print(f"\n  177 / X = ?")
    print(f"  {'X':>15s} {'177/X':>10s} {'Nearest int':>12s} {'Residual':>10s}")
    print(f"  {'-'*52}")
    for quotient, name in factorizations:
        nearest = round(quotient)
        resid = quotient - nearest
        marker = " ***" if abs(resid) < 0.02 else " **" if abs(resid) < 0.05 else ""
        print(f"  {name:>15s} {quotient:>10.4f} {nearest:>12d} {resid:>+10.4f}{marker}")


if __name__ == '__main__':
    test_algebraic()

# Predicted Half-Lives for All AME2020 Isotopes

**Date:** 2026-01-02

**Total isotopes:** 3530

## Prediction Models

### Alpha Decay
```
log₁₀(t₁/₂) = -24.14 + 67.05/√Q + 2.56*|ΔN|
```

### Beta- Decay
```
log₁₀(t₁/₂) = 9.35 + -0.63*log(Q) + -0.61*|ΔN|
```

### Beta+ Decay
```
log₁₀(t₁/₂) = 11.39 + -23.12*log(Q) + 0.00*|ΔN|
```

## Decay Mode Distribution

| Mode | Count | Percentage |
|------|-------|------------|
| stable | 242 | 6.9% |
| alpha | 550 | 15.6% |
| beta- | 1420 | 40.2% |
| beta+ | 1318 | 37.3% |

## Examples

### Very Long-Lived Alpha Emitters

| Isotope | Q (MeV) | ΔN | Predicted t₁/₂ (years) |
|---------|---------|----|-----------------------|
| Pt-198 | 0.11 | 1 | 2.15e+176 |
| Hg-202 | 0.13 | 1 | 1.02e+154 |
| Ho-165 | 0.14 | 1 | 4.61e+150 |
| Tl-205 | 0.16 | 1 | 1.16e+141 |
| Tb-157 | 0.18 | 1 | 2.49e+129 |

### Fast Beta- Emitters

| Isotope | Q (MeV) | ΔN | Predicted t₁/₂ (sec) |
|---------|---------|----|-----------------------|
| Tl-210 | 5.48 | -3 | 1.09e+07 |
| Ta-188 | 4.76 | -3 | 1.20e+07 |
| Hg-207 | 4.55 | -3 | 1.23e+07 |
| Au-204 | 4.30 | -3 | 1.28e+07 |
| Fr-226 | 3.85 | -3 | 1.37e+07 |

### Predicted Stable Isotopes

Total: 242 isotopes

Examples: H-1, H-2, He-3, He-4, He-5, Li-6, Li-7, Be-7, Be-8, Be-9, B-10, B-11, C-12, C-13, N-14, N-15, O-16, O-17, O-18, F-19

## Full Dataset

See `predicted_halflives_all_isotopes.csv` for complete predictions.

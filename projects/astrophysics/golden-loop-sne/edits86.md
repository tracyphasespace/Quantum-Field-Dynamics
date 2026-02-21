# edits86.md — Barrier Physics: Independent Gate Architecture (Research Direction)

**Target**: QFD_Edition_v10.4.md
**Source**: Barrier physics research (`barrier-physics/tune_tools.py`, `BARRIER_RESULTS.md`)
**Date**: 2026-02-21
**Upstream dependencies**: edits82 (§8.7.4 daughter-pull model should be applied first)
**Status**: SPEC — LOW PRIORITY (heuristic research, not canonical)

**Theme**: The v8 landscape predictor achieves 76.6% mode accuracy (GS) using geometric gates with hard thresholds (pf > 0.5 for alpha, pf > 1.74 for SF). A natural question is whether the gates themselves can be derived from physical triggers rather than coordinate thresholds. This section documents a research exploration that replaces the monolithic decision tree with eight independent binary gates — each asking a single physical YES/NO question — then resolving overlaps with precision filters. The result is 83.3% mode accuracy (+6.7% vs v8 baseline) with 2 fitted parameters. We present this to show opportunity, not to claim a solved problem. The gates and filters interact in ways that are not yet well understood, and improving one mode's gate often degrades another. This is heuristic territory that requires further research.

**Strategy**: One edit — insert a subsection after the Three Zones discussion (end of §8.7) and before §8.8, framing this as a research direction that demonstrates the potential of physics-based triggers but honestly documents the competition between gates.

**Why**: The current geometric gates (§8.7.4) use hard thresholds: pf ≥ 0.5 for alpha, pf > 1.74 AND cf > 0.881 for SF, ε > 3.0 AND A < 50 for proton. These thresholds WORK (76.6%), but they are ad hoc — they don't derive from a physical mechanism. The barrier physics approach asks: can we compute WHETHER a decay channel is physically open, rather than checking whether a coordinate exceeds a threshold? The answer is a qualified yes: the alpha scission barrier is a computable function of (A, Z, pf, ε) and opens progressively with peanut deformation. But the resulting gates compete with each other in ways that prevent clean optimization.

**Key Results**:
- Mode accuracy: 83.3% (B-: 93.3%, B+: 89.3%, alpha: 73.7%, SF: 30.6%, n: 100%, p: 30.6%)
- Independent binary gates: each tool evaluated YES/NO on all 3111 ground-state nuclides
- Alpha scission barrier: B_eff = B_surf(A,4) - K_SHEAR*pf^2 - k_coul*K_COUL(A)*max(0,eps)
- Neutron trigger: core overflow (cf > 1.0) catches 100% of neutron emitters
- SF and alpha are landscape-degenerate at Z >= 98 — no topological variable separates them
- 2 fitted parameters: K_SHEAR = 2.0, k_coul = 3.0 (tuned on alpha territory only)
- Competition: improving SF recall from 31% to 92% costs 10% alpha accuracy (net negative)

---

## Edit 86-01: Insert Research Subsection — Barrier Physics and Independent Gate Architecture

**Priority**: LOW (research direction, not canonical result)
**Section**: After Three Zones table (end of §8.7 material), before §8.8
**Action**: Insert new subsection

**FIND**:
```
Zone 1 performs best (83.7% mode accuracy) because the physics is cleanest. Zone 3 is hardest (60.4%) because alpha, beta, and fission compete in the same geometric space.


### **8.8 Atomic Architecture: From Nucleus to Bond**
```

**REPLACE**:
```
Zone 1 performs best (83.7% mode accuracy) because the physics is cleanest. Zone 3 is hardest (60.4%) because alpha, beta, and fission compete in the same geometric space.


### **Research Direction: Barrier Physics and Independent Gates**

The geometric gates above use coordinate thresholds: pf ≥ 0.5 for alpha, pf > 1.74 for SF. These thresholds work, but they are prescriptive — they say WHERE a mode appears on the chart without explaining WHY. A natural question: can we compute whether a decay channel is physically open, rather than checking whether a coordinate exceeds a cutoff?

**Independent Binary Gates.** We replace the monolithic decision tree with eight independent tools. Each tool asks a single physical question — YES or NO — for every nuclide:

| Tool | Physical Question | Trigger | Recall | Precision |
|------|------------------|---------|--------|-----------|
| Neutron | Is the core overfull? | cf > 1.0, Z ≤ 9 | 100% | 59% |
| Proton | Is the proton unbound? | N/Z < 0.75 (light) | 37% | 45% |
| Beta- | Does the survival gradient point toward Z+1? | gain(Z+1,A) > 0 | 94.5% | 92.7% |
| Beta+ | Does the survival gradient point toward Z-1? | gain(Z-1,A) > 0 | 98.5% | 63.7% |
| Alpha | Is the scission barrier open? | B_eff ≤ 0 | 80.1% | 69.3% |
| SF | Is the neck ready to rupture? | pf > 1.74, cf > 0.881, even-even | 30.6% | 30.6% |
| Stable | Is there no gradient and no fracture risk? | No beta gain, pf < 0.3 | 39.7% | 74.5% |

Each tool is evaluated independently on all 3111 ground-state nuclides. When multiple gates fire simultaneously, a priority order resolves the conflict: neutron > proton > SF > alpha > beta > stable.

**The Alpha Scission Barrier.** The key physics in this approach is a computable scission barrier for alpha emission. Surface tension resists fracture; peanut deformation and Coulomb repulsion drive it:

> B_eff = max(0, B_surf(A,4) - K_SHEAR * pf^2 - k_coul * K_COUL(A) * max(0, epsilon))

where B_surf is the soliton surface cost of splitting mass A into fragments (A-4) and 4, and K_COUL(A) is the Coulomb coefficient from the valley formula. The barrier opens progressively with peanut factor:

| pf range | Barrier open (alpha predicted) | Observed alpha fraction |
|----------|-------------------------------|------------------------|
| 0.0 - 0.3 | 0% | 0% |
| 0.5 - 0.7 | 22% | 22% |
| 1.0 - 1.3 | 86% | 54% |
| 1.6 - 2.0 | 100% | 100% |

This is the correct qualitative behavior: alpha emission turns on smoothly as the peanut forms, not as a step function at a threshold.

**Gate Competition.** The combined predictor reaches 83.3% mode accuracy — a 6.7% improvement over the v8 landscape baseline. However, the gates interact in ways that resist clean optimization. The following table shows that within the limited search space, the best parameters for one mode degrade another:

| Configuration | Total | B- | B+ | Alpha | SF | p | n |
|---------------|-------|-----|-----|-------|------|------|------|
| v8 baseline | 77.1% | 86.0% | 83.2% | 70.5% | 30.6% | 10.2% | 68.8% |
| Best total accuracy | **83.3%** | 93.3% | 89.3% | 73.7% | 30.6% | 30.6% | 100% |
| Most balanced modes | 82.6% | 93.4% | 87.1% | 69.2% | 63.3% | 40.8% | 100% |
| Maximum SF recall | 81.9% | 93.4% | 87.1% | 60.9% | 91.8% | 40.8% | 100% |

The tradeoff is stark: pushing SF recall from 31% to 92% costs 13 percentage points of alpha accuracy, because SF and alpha are landscape-degenerate at Z ≥ 98. No combination of topological variables (epsilon, pf, cf, N/Z, parity) cleanly separates them. This is consistent with the Lagrangian separation — the landscape decides that fracture will occur, but the specific fracture mode (tip shedding vs. global bifurcation) requires dynamical information that geometry alone cannot provide.

**What This Tells Us.** Three findings are robust:

1. **Core overflow IS the neutron trigger.** cf > 1.0 catches all 16 neutron emitters with no false negatives. This is a physical gate, not a coordinate threshold.

2. **The alpha barrier opens continuously with pf.** The scission barrier formula reproduces the observed onset of alpha decay as a progressive opening, not a step function. The two fitted parameters (K_SHEAR = 2.0, k_coul = 3.0) are the only non-derived constants in the architecture.

3. **SF and alpha are topologically indistinguishable.** At Z ≥ 98 and pf > 1.7, both modes occupy the same region of configuration space. Every gate that catches more SF emitters also misclassifies alpha emitters. The distinction between tip shedding and neck rupture is dynamical, not topological — consistent with the Lagrangian separation L = T[pi, e] - V[beta].

These results show that the gate architecture has room to improve beyond the v8 landscape baseline, but the competition between gates — particularly the alpha/SF degeneracy and the alpha/B+ overlap in the peanut transition zone — sets a ceiling that heuristic tuning cannot breach. Breaking through this ceiling requires either dynamical variables (half-life information, Q-values) or a deeper geometric principle that distinguishes fracture modes. This remains an open problem.


### **8.8 Atomic Architecture: From Nucleus to Bond**
```

---

## Summary Table

| Edit | Section | Action | Priority | Lines |
|------|---------|--------|----------|-------|
| 86-01 | After Three Zones, before §8.8 | Insert barrier physics research subsection | LOW | ~50 |

## Execution Notes

1. This edit inserts AFTER the Three Zones table and BEFORE §8.8.
2. Search for "Zone 3 is hardest (60.4%) because alpha, beta, and fission compete" as the unique anchor.
3. The subsection is framed as "Research Direction" — not a canonical result. It honestly documents what works, what doesn't, and why the gates compete.
4. The 2 fitted parameters (K_SHEAR, k_coul) are the only non-derived constants. All other gates use QFD-derived variables (cf, pf, epsilon, survival score).
5. The gate competition table is the key deliverable — it shows that within a limited parameter search, the results are the best values achievable but the tools fight each other for accuracy.

## Cross-references

- §8.7.4 (Three Geometric Dimensions) — defines pf, cf, epsilon used in all gates
- Parent Push / Daughter Pull — the daughter-pull action achieves the same 76.6% ceiling
- §8.7.3 (Lagrangian Separation) — explains WHY SF and alpha are landscape-degenerate
- Appendix A.6 (Geometric Gates) — the v8 gates this research extends
- `barrier-physics/BARRIER_RESULTS.md` — full technical documentation
- `barrier-physics/tune_tools.py` — implementation and validation code

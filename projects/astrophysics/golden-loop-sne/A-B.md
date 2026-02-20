# Aharonov-Bohm: Potential versus Force

## Classical Picture

Forces are real, potentials are bookkeeping.  You can always
gauge-transform **A** → **A** + ∇χ without changing **B** = ∇×**A**.
The potential appears to be "just math."

## The A-B Experiment (1959)

An electron beam splits, passes on either side of a solenoid,
and recombines.  Outside the solenoid:

- **B** = 0 everywhere the electrons travel
- **E** = 0 everywhere the electrons travel
- **F** = 0 — no force acts on the electrons at any point

Yet the interference pattern **shifts** by a phase:

    Δφ = (e/ℏ) ∮ A · dl

The line integral of **A** around the solenoid is nonzero (it equals
the enclosed magnetic flux Φ), even though **B** = 0 on the path.
The electrons feel the potential directly, not through any force.

## What Each Description Captures

|                        | Force (F = qE + qv×B) | Potential (A, φ)                   |
|------------------------|------------------------|------------------------------------|
| Local                  | Yes                    | Yes                                |
| Observable classically | Yes                    | No (gauge freedom)                 |
| Observable quantum     | Yes                    | **Yes** (A-B phase)                |
| Topology-sensitive     | No — F=0 ⇒ nothing    | **Yes** — enclosed flux matters    |

## Why It Matters

The force picture is **local and differential** — what is happening
right here, right now.

The potential picture is **global and topological** — what is enclosed
by the path, regardless of local conditions.

Two configurations with identical **F** everywhere can have different
**A** (up to topology), and quantum mechanics tells the difference.

Differentiation loses topological information.  The potential is not
gauge junk — it is the more fundamental object.  Forces are derivatives
of potentials, and **derivatives kill global structure**.

This is why every gauge theory (QED, QCD, the full Standard Model) is
written in terms of potentials (connections), not forces (curvatures).
The connection **A** is the real thing; the field strength
**F** = d**A** is a derived quantity that misses global structure.

## Relevance to QFD

QFD describes the vacuum as a medium with nontrivial topology — vortex
filaments, soliton structure, topological winding.  The Kelvin wave
framework treats scattering as interaction with this topology:

- The **forward vertex** (σ ∝ E) is a coherent drag — the photon
  accumulates phase from the vacuum connection along its worldline,
  producing redshift without any local force event.

- The **non-forward vertex** (σ ∝ √E) excites Kelvin waves on vortex
  filaments — a topological interaction that depends on the filament
  winding number, not on a local field strength.

Both mechanisms are potential-level (connection-level) physics.
A force-level description would see F = 0 in the vacuum between
filaments and predict no interaction — exactly the mistake the
classical picture makes in the Aharonov-Bohm solenoid.

The A-B effect is the experimental proof that nature operates at the
potential level.  QFD takes this seriously: the vacuum connection is
physical, its topology is observable, and forces are an incomplete
derivative of the real structure.

## Rift Range Extension: Black Holes as Topological Solenoids

The Rift Abundance model (§11.3) treats black holes as engines that
process infalling matter through topological rifts, producing the
observed H/He ratio (75/25) via a four-step feedback loop:
compression, Boltzmann filtering, electron-capture suppression, and
alpha-dominance.  The classical picture limits the BH's processing
zone to the tidal disruption radius — the distance where gravitational
**force** (tidal stress) exceeds the soliton's self-binding.

The A-B principle extends this range.

### The BH as a Topological Defect

A QFD black hole is not a point mass in flat space.  It is a
topological defect in the vacuum — a region of maximal winding
number where the Cl(3,3) connection has a non-trivial holonomy.
Like the solenoid in the A-B experiment:

- **Inside** the BH: extreme curvature, topology change, rift
  processing (analogous to B ≠ 0 inside the solenoid)
- **Outside** the BH: the local force field falls off as 1/r²,
  but the **connection** retains the enclosed topological charge

A soliton (Q-ball nucleus) orbiting well outside the tidal radius
accumulates phase from the BH's vacuum connection:

    Δφ_grav = ∮ Γ · dl

where Γ is the gravitational connection (Christoffel symbols in GR;
the Cl(3,3) vacuum connection in QFD).  This phase accumulation is
**not** a tidal force — it is a potential-level interaction that
modifies the soliton's internal coherence without exerting a
classical pull.

### Three Consequences for Rift Processing

**1. Enlarged capture cross-section.**  Solitons feel the BH's
topological influence before reaching the Roche limit.  The
connection-level interaction pre-conditions infalling matter:
internal phase coherence degrades, the soliton's winding modes
soften, and disruption begins at larger radii than the force
picture predicts.  The effective rift zone is set by the
**connection radius** (where Δφ ~ π), not the tidal radius
(where F_tidal > F_binding).

**2. Coulomb Ejection Spring at potential level.**  The BH charges
positive (electron mobility 43-56× ion mobility → preferential
electron capture).  The resulting Coulomb potential φ = Q_BH/r
acts on stripped nuclei (bare protons, alpha particles) ejected
by the rift.  In the force picture, this repulsion falls off as
1/r² and becomes negligible at ~10³ Schwarzschild radii.  In the
potential picture, the phase coupling ∮ eA · dl remains significant
at much larger distances — the ejected nuclei carry a topological
imprint of the BH's charge that influences their recombination
chemistry far downstream.  This extends the zone where H is
preferentially retained over He.

**3. Topological protection of the attractor.**  The rift model's
remarkable insensitivity to its calibration constant (k ±50% →
H% ±5%) is a hallmark of topological protection.  In the force
picture, this robustness is unexplained — why should a 50%
change in coupling strength barely affect the output?  In the
A-B picture, the answer is clear: the rift's selectivity depends
on **enclosed winding number** (an integer), not on coupling
strength (a continuous parameter).  Small perturbations to k
change the force magnitude but cannot change the topology.  The
attractor is stable because it is topologically quantized.

### Quantitative Results: Three Nested Rift Zones

The hand-wavy scaling estimate above is replaced by a computed
result from the ψ-field tail framework (Appendix U.2).

**The d-independent ratio.**  Each BH has a ψ-field tail
δψ/ψ₀ = R_s/(ξ_QFD × r), and the gravitational potential
|Φ|/c² = R_s/r.  Both decay as 1/r.  At the L1 saddle point
between two equal-mass BHs at separation d, their ratio is
therefore **constant** — independent of d:

    (δψ_gap/ψ₀) / (|Φ_L1|/c²) = 2/ξ_QFD = 2/16.157 = 0.1238

where ξ_QFD = k_geom² × 5/6 = 16.157 (zero free parameters).

**Escape velocity reduction.**  The ψ-overlap at L1 raises the
vacuum field, softening soliton binding.  The fractional barrier
reduction is 2/ξ_QFD = 12.4%, giving:

    v_esc(connection) = v_esc(classical) × √(1 − 2/ξ_QFD)
                      = 0.936 × v_esc(classical)

The escape velocity is 6.4% lower than the force picture predicts,
independent of BH mass, separation, or rift type.

**Separation distance extension.**  Since barrier ∝ 1/d, the 12.4%
reduction extends the maximum escape separation by:

    d_escape = d_classical / (1 − 2/ξ_QFD) = 1.141 × d_classical

This produces **three nested processing zones**:

| Zone | Boundary | Mechanism | Volume |
|------|----------|-----------|--------|
| 1. Classical tidal | d ≤ 3.45 R_s | Tidal stress > soliton binding | 172 R_s³ |
| 2. Topological channel | d ≤ 8.30 R_s | ψ-overlap > η_topo threshold | 2,395 R_s³ |
| 3. A-B extended | d ≤ 9.47 R_s | Escape velocity lowered 6.4% | 3,560 R_s³ |

**Total volume ratio: Zone 3 / Zone 1 = 20.7×**
(= 13.9× from topological channel × 1.49 from A-B extension)

**Effect on H/He ratio.**  Despite the 12.4% barrier change, the
cosmic abundance barely moves:

| Scenario | k | Global H% |
|----------|---|-----------|
| Bare classical (no A-B) | 6.25 | 74.80% |
| Current calibration | 5.48 | 74.68% |
| With A-B on top | 4.80 | 74.54% |

Across ±25% barrier swing, H% shifts by only ±0.15 percentage
points.  The attractor is insensitive because the interior
stratification (onion-layer composition) dominates the selectivity.
This IS the quantitative proof of topological protection.

**What changes is the rate, not the ratio.**  The 21× volume
enhancement means each binary BH interaction processes 21× more
matter, relaxing the required interaction frequency for steady-state
hydrogen replenishment against stellar burndown.

### Summary

| Quantity             | Force Picture         | Connection Picture          |
|----------------------|-----------------------|-----------------------------|
| BH influence range   | d_tidal = 3.45 R_s    | d_AB = 9.47 R_s (2.7× farther) |
| Processing volume    | 172 R_s³              | 3,560 R_s³ (21× larger)    |
| Escape velocity      | v_esc(classical)      | 0.936 × v_esc (6.4% lower) |
| Barrier reduction    | —                     | 2/ξ_QFD = 12.4% (zero parameters) |
| H/He ratio           | 74.8%                 | 74.5% (±0.15%, protected)   |
| Attractor stability  | Unexplained           | Winding number quantization |

The Aharonov-Bohm principle does not just validate QFD's
connection-level vacuum physics.  Applied to the rift model,
it predicts that black holes process matter over a **21× larger
volume** than the force picture allows — and that the resulting
abundance ratios are topologically protected against perturbation.

**Computation:** `rift-abundance/ab_barrier_calculation.py`
**Book edit spec:** `edits74.md` (Appendix U.2.3 + §11.3 cross-reference)

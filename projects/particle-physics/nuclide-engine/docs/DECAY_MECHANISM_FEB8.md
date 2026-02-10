# Unified Decay Mechanism — Density Overflow, Not Energy Barriers

**Date**: 2026-02-08
**Status**: Theoretical framework derived from frozen core conjecture
**Depends on**: FROZEN_CORE_CONJECTURE.md (N_max = 2*pi*beta^3)
**Provenance**: Physical insight (user) + QFD framework

---

## 1. The Core Claim

The frozen core conjecture (N_max = 2*pi*beta^3, confirmed at 0.049%)
implies that neutron emission, alpha shedding, and spontaneous fission
are NOT three separate mechanisms with three separate energy barriers.

They are **one mechanism** — density overflow — at three severity levels.

The Psi field density ceiling (set by beta) determines all three. When the
neutral core exceeds its capacity, the system reorganizes at whatever scale
is necessary. The only variable is HOW MUCH needs to be shed.

---

## 2. Standard Picture (What This Replaces)

In competing physics (shell model, liquid drop, QM barrier penetration):

| Decay | Standard mechanism | Key parameter |
|-------|-------------------|--------------|
| Neutron emission | Unbound state — neutron has excess energy above the potential well | Separation energy S_n |
| Alpha decay | Quantum tunneling — He-4 tunnels through the Coulomb barrier | Geiger-Nuttall law, barrier height/width |
| Spontaneous fission | Liquid drop instability — fissility Z^2/A exceeds critical value | Fission barrier height |

Each mechanism has its own formalism, its own parameters, and its own
conceptual framework. There is no unified explanation for why these three
exist as distinct decay channels.

---

## 3. Frozen Core Picture (The Unified Mechanism)

### The setup

The soliton's neutral core has a maximum capacity of N_max = 2*pi*beta^3
winding states. The core grows at a constant rate of ~2/3 neutrons per
proton (dN_excess/dZ = 1 - 1/beta = 0.671) until it hits the wall.
Above A ~ 150, the core has deformed into a peanut — two density centers
sharing one outer charge shell.

### The mechanism

When the core is at or above capacity, it cannot hold what it has. The
system must reorganize. The reorganization happens at the scale set by
the severity of the overflow:

### 3.1 Neutron emission — minimal overflow

**Trigger**: One extra winding state doesn't fit. The core density at
that location exceeds the Psi field ceiling.

**Mechanism**: The excess neutral winding state is pushed out. There is
no barrier to overcome — the state is ejected because there is literally
no room at that density. The core is full.

**Where it dominates**: Neutron-rich light nuclei (A < 100) where the
core is slightly overfull but no peanut exists. Also the neutron drip
line at all masses — the boundary where N_max(Z) is reached.

**Timescale**: Fast (microseconds to milliseconds). The overflow is
immediate — the system barely sustains the extra state before ejecting it.

**QFD language**: The winding state has no topological home. The Psi
field density at the core boundary exceeds the ceiling, and the excess
state decouples.

### 3.2 Alpha shedding — moderate overflow (peanut regime)

**Trigger**: The peanut neck is thin enough that a minimum complete
soliton (He-4) can detach. The density overflow is not just one state
but a coherent He-4-sized region of the neck.

**Mechanism**: This is NOT quantum tunneling through a Coulomb barrier.
It is a topological pinch-off — the neck's winding coherence fails over
a He-4-sized region, and the minimum stable soliton separates. He-4 is
the shed fragment because it is the smallest soliton with a complete
topological winding (a closed geometric phase). You cannot shed half a
soliton.

**Where it dominates**: Heavy nuclei above A ~ 195 where the peanut
exists and the neck geometry permits pinch-off. Specifically, epsilon > 0
(proton-rich side of the valley) AND A > A_CRIT + WIDTH (well into the
peanut regime).

**Timescale**: Slow (seconds to gigayears). The neck must narrow
sufficiently for the coherence to fail over a He-4 region. Emergent
time within the soliton (set by beta) determines how long the metastable
neck can persist.

**QFD language**: Soliton shedding at the peanut neck. The topological
winding cannot maintain coherence over the neck-to-lobe junction for
a He-4-sized piece. The shed piece carries away a complete geometric
phase.

### 3.3 Spontaneous fission — catastrophic overflow

**Trigger**: The two peanut lobes are too far apart for the topological
winding to bridge the neck at all. The charge shell cannot maintain
coherence across the gap.

**Mechanism**: Complete topological bifurcation. The neck breaks entirely
and the soliton splits into two daughter solitons. This is "too far apart
to experience self-repulsion" — the topological self-interaction that
confines the soliton falls off with neck width, and below a critical
width the winding cannot sustain the bridge.

**Where it dominates**: The heaviest nuclei (A > 240, especially
even-even), where the peanut is maximally extended and the lobes are
at maximum separation.

**Timescale**: Variable (microseconds for superheavy, gigayears for
actinides). Depends on how far beyond the neck coherence limit the
system sits.

**QFD language**: Topological bifurcation. The Psi field's self-
interaction across the neck drops below the threshold for maintaining
a single soliton's winding.

---

## 4. Key Differences from Standard Picture

### 4.1 No energy barriers

In the standard picture, decay requires overcoming (or tunneling through)
an energy barrier. The barrier height determines the rate.

In the frozen core picture, there are no barriers. The core is
GEOMETRICALLY incapable of holding what it has. The "barrier" is actually
the topological winding strength at the neck/surface — how long the
system can sustain an over-capacity state before reorganizing.

### 4.2 The half-life is emergent, not fundamental

The apparent half-life is not a tunneling probability through a potential
barrier. It is the timescale of emergent time within the soliton — how
long the metastable over-capacity state persists before the topological
reorganization occurs. This timescale is set by beta (the same constant
that determines the capacity ceiling).

This explains why the zero-parameter clock works: log10(t_half) has
slopes of -pi*beta/e (beta-minus), -pi (beta-plus), and -e (alpha),
all involving only transcendental numbers and beta. The rates are
geometric properties of the reorganization, not tunneling probabilities.

### 4.3 No Coulomb barrier for alpha

The standard Geiger-Nuttall law relates alpha half-life to the Coulomb
barrier. In the frozen core picture, the "barrier" is the neck's
topological coherence. The Geiger-Nuttall correlation exists because
higher Z means a more extended peanut (larger charge shell), which means
a thinner neck, which means faster pinch-off. The correlation is real
but the mechanism is geometric, not electrostatic.

### 4.4 No fissility parameter for SF

The standard Z^2/A fissility parameter is a liquid-drop stability
criterion. In the frozen core picture, SF happens when the peanut lobes
exceed the winding's bridging distance. Z^2/A correlates because larger
Z^2/A means more protons pushing the charge shell apart (stretching the
peanut), but the actual threshold is topological: can the winding bridge
the neck?

### 4.5 Neutron emission is not "unbound"

In the standard picture, a neutron is unbound when the separation energy
S_n < 0. In the frozen core picture, the neutron is ejected because the
core exceeds its density ceiling. S_n < 0 is the CONSEQUENCE (the energy
accounting shows a surplus because the state can't fit), not the CAUSE
(the cause is geometric overflow).

---

## 5. The Unified Ordering

For a given nuclide with (Z, A) above the density ceiling:

```
  Core barely overfull?
    YES → neutron emission (shed one winding state)
    NO → is there a peanut?
      NO → beta decay (convert charge state, try to reach valley)
      YES → is the neck thin enough for He-4 pinch-off?
        YES → alpha shedding
        NO → is the neck beyond coherence?
          YES → spontaneous fission
          NO → metastable — wait (long half-life)
```

This decision tree is purely geometric. No energy calculations, no
barrier heights, no tunneling probabilities. The geometry of the
soliton (set by alpha and beta) determines which channel is available.

---

## 6. The Surface Tension Analogy

The mechanism is identical to a cup of water filled past the brim.

### The meniscus

Surface tension at the rim holds the water above the cup's nominal
capacity. The liquid forms a meniscus — bulging above the rim but not
spilling. In the soliton, the topological winding at the core boundary
(the "rim") holds winding states beyond the nominal capacity (N_max).
The system is metastable. The meniscus can hold.

### The overflow

When surface tension fails, the water doesn't spill one drop at a time
proportional to the excess. The meniscus breaks where it is weakest, and
the amount that spills is determined by the GEOMETRY of the break — the
area of the rim failure — not the volume of excess water.

In the soliton: the topological winding fails at the peanut neck (the
weakest point of the "rim"), and what comes out is a topologically
complete fragment (He-4, the minimum stable soliton). You overfill by
one winding state but lose four (an entire He-4) because the surface
tension doesn't fail at a point — it fails over a topologically complete
region. The minimum failure region IS He-4.

This is why the shed fragment is "mysterious" from a volume perspective.
The standard picture asks "how much excess energy is there?" (volume).
The correct question is "where does the surface tension fail, and what
is the minimum coherent piece that can detach?" (area/topology).

### The half-life is the meniscus lifetime

A cup filled to 101% holds for a long time — the surface tension is
barely strained. A cup at 110% breaks immediately. The meniscus
lifetime depends on how far above capacity you are.

In the soliton: the valley stress epsilon measures how far above
capacity the system sits. The clock formula log10(t_half) ~ -slope *
sqrt(|epsilon|) says: further from the valley = more overfull = shorter
meniscus lifetime. The slope itself (-pi*beta/e for beta-minus, -pi for
beta-plus, -e for alpha) is a geometric property of HOW the surface
tension fails — different failure modes (charge conversion vs shedding
vs bifurcation) have different geometric prefactors.

### The 2/3 core slope IS surface-area scaling

The core capacity grows as dN_excess/dZ = 1 - 1/beta ~ 2/3. This is
A^(2/3) scaling — surface area, not volume. The "cup" gets bigger as you
add protons, but the capacity is limited by the surface (the topological
winding at the core boundary), not by the volume (the space inside).

This confirms that the density ceiling is a SURFACE phenomenon: the
winding at the boundary of the core is what sets the maximum, not the
density of the interior. The interior can (and does) reach maximum
density, but whether the core holds together depends on the boundary's
topological coherence — its "surface tension."

---

## 7. Testable Predictions

### 7.1 Neutron emission at the drip line

**Prediction**: Neutron emission timescales should correlate with
N - N_max(Z), the overflow amount, not with S_n (separation energy).
If the mechanism is capacity overflow, the rate depends on HOW MUCH
the core exceeds capacity, not on the energy balance.

**Test**: Plot neutron emission half-lives vs (N - N_max_stable) for
each element chain. If the frozen core picture is correct, this should
be a better predictor than S_n alone.

### 7.2 Alpha decay and peanut geometry

**Prediction**: Alpha half-lives should correlate with the peanut's
neck width (derivable from epsilon and A), not just with Q_alpha and
the Coulomb barrier parameters.

**Test**: The daughter-magic correction (Section 7.1 of the V5 clock)
gave +43 R-squared points for alpha because daughter proximity to
N=126/Z=82 affects the neck geometry. Look for additional geometric
predictors beyond the existing epsilon + ln(A/Z) basis.

### 7.3 SF threshold

**Prediction**: The SF threshold should correspond to a critical peanut
aspect ratio (lobe separation / lobe diameter), not to a critical
fissility Z^2/A. These correlate but are not identical — the prediction
diverges for deformed nuclei where Z^2/A is below the standard threshold
but the peanut is still extended.

**Test**: The fission parity constraint (odd-N forbids symmetric fission)
already supports the topological picture. Look for additional cases where
the geometric and energy-barrier predictions diverge.

### 7.4 The N = 177 wall

**Prediction**: If the mechanism is density overflow, then all nuclides
with N > 177 should be extremely short-lived (the core immediately
ejects excess). The half-life should drop discontinuously at N = 177,
not gradually.

**Test**: Plot half-life vs N for Z = 110-118. Is there a sharp
transition at N = 177, or a gradual decline?

---

## 8. Connection to Existing Results

| Existing result | Connection to density overflow |
|----------------|-------------------------------|
| N_max = 2*pi*beta^3 | Sets the capacity ceiling |
| Core slope = 1 - 1/beta | Rate of core filling until ceiling |
| Alpha shell ratio 1.288 | Geometric spacing between density reorganization events |
| Zero-param clock slopes | Reorganization timescales from beta geometry |
| Daughter-magic correction | Neck geometry depends on daughter's proximity to resonance |
| Fission parity constraint | Topological constraint on neck splitting symmetry |
| Electron damping factor | Same Psi field density limits, different boundary |
| Band width peak at Z=60-85 | Maximum core flexibility before peanut locks in |

---

## 9. Notation

| Term | QFD meaning | Replaces (standard) |
|------|------------|-------------------|
| Density overflow | Core exceeds Psi field ceiling | "Unbound state", "above barrier" |
| Topological pinch-off | Neck coherence fails over He-4 region | "Coulomb barrier tunneling" |
| Topological bifurcation | Neck breaks entirely | "Fission barrier exceeded" |
| Emergent time | Metastable persistence timescale from beta | "Tunneling probability", "decay constant" |
| Frozen core | Core at maximum density, held by time dilation | "Ground state nucleus" |
| Melting | Core exceeds ceiling, reorganizes to lower density | "Excited state", "above threshold" |
| Capacity | N_max = 2*pi*beta^3 winding states | "Neutron separation energy" |

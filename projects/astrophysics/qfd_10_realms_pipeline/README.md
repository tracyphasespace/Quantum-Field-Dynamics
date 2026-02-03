# QFD 10 Realms Pipeline

## Golden Loop COMPLETE ✅

**α → β → (e, μ, τ)** - All three charged lepton masses reproduced using β = 3.043233053 from fine structure constant α with zero free coupling parameters.

### Quick Start (30 seconds)

```bash
python test_golden_loop_pipeline.py
```

**Results**: All three leptons with chi² < 10⁻⁹ in ~20 seconds.

### Documentation for Researchers

📘 **[REPLICATION_GUIDE.md](REPLICATION_GUIDE.md)** - Quick replication in 30 seconds
📊 **[GOLDEN_LOOP_RESULTS.md](GOLDEN_LOOP_RESULTS.md)** - Complete results and physics explanation
🔬 **[SOLVER_COMPARISON.md](SOLVER_COMPARISON.md)** - Four solver approaches compared (Phoenix, V22 Quartic, V22 Enhanced, Pipeline)

**Key Finding**: Same vacuum stiffness β reproduces electron, muon, and tau masses through Hill vortex geometric quantization. Demonstrates universal vacuum mechanics connecting electromagnetism (α) to inertia (mass).

---

## Realm 0 — CMB Baseline & Polarization Gate

Run first to anchor the thermalization zeropoint (T_CMB) and enforce vacuum polarization constraints (no birefringence, TB/EB≈0).

```bash
python scripts/run_realm0_cmb.py
```

Then proceed to Realm 3a (PPN gate) and onward.

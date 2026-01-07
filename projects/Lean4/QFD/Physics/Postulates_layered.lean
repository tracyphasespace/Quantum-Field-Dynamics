structure Core where
  lepton_conservation :
    ∀ {sys : PhysicalSystem},
      total_lepton_num sys.input = total_lepton_num sys.output
  mass_winding_rule :
    ∀ ⦃ℓ : Lepton⦄, winding_number ℓ > 0 → mass ℓ > base_mass

structure SolitonPostulates extends Core where
  topological_charge : QFD.Soliton.FieldConfig → ℤ
  noether_charge : QFD.Soliton.FieldConfig → ℝ
  topological_conservation :
    ∀ evolution : ℝ → QFD.Soliton.FieldConfig,
      (∀ t, ContinuousAt evolution t) →
      ∀ t1 t2 : ℝ,
        topological_charge (evolution t1) = topological_charge (evolution t2)
  zero_pressure_gradient :
    ∀ ϕ : QFD.Soliton.FieldConfig,
      QFD.Soliton.is_saturated ϕ →
        ∃ R : ℝ, ∀ r, r < R →
          HasDerivAt (fun r => QFD.Soliton.EnergyDensity ϕ r) 0 r

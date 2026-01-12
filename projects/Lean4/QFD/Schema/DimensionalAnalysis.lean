import Mathlib.Data.Real.Basic

noncomputable section

namespace QFD.Schema

/-!
# Dimensional Analysis System

A type-safe system for physical quantities to prevent dimensional errors
in the Grand Solver parameters.

## Dimensions
We track 4 fundamental dimensions:
* Length [L]
* Mass [M]
* Time [T]
* Charge [Q]
-/

structure Dimensions where
  length : ℤ
  mass : ℤ
  time : ℤ
  charge : ℤ
  deriving DecidableEq, Repr, Inhabited

def Dimensions.none : Dimensions := ⟨0, 0, 0, 0⟩

instance : Add Dimensions where
  add d1 d2 := ⟨d1.length + d2.length, d1.mass + d2.mass, d1.time + d2.time, d1.charge + d2.charge⟩

instance : Sub Dimensions where
  sub d1 d2 := ⟨d1.length - d2.length, d1.mass - d2.mass, d1.time - d2.time, d1.charge - d2.charge⟩

instance : Neg Dimensions where
  neg d := ⟨-d.length, -d.mass, -d.time, -d.charge⟩

/-- A physical quantity with value and dimensions. -/
structure Quantity (d : Dimensions) where
  val : ℝ

/-! ## Fundamental Units -/

def Unitless := Quantity Dimensions.none
def Length := Quantity ⟨1, 0, 0, 0⟩
def Mass := Quantity ⟨0, 1, 0, 0⟩
def Time := Quantity ⟨0, 0, 1, 0⟩
def Charge := Quantity ⟨0, 0, 0, 1⟩
def Velocity := Quantity ⟨1, 0, -1, 0⟩
def Energy := Quantity ⟨2, 1, -2, 0⟩
def Force := Quantity ⟨1, 1, -2, 0⟩
def Action := Quantity ⟨2, 1, -1, 0⟩ -- Angular Momentum / Planck
def Density := Quantity ⟨-3, 1, 0, 0⟩

/-! ## Operations -/

def Quantity.add {d : Dimensions} (a b : Quantity d) : Quantity d :=
  ⟨a.val + b.val⟩

def Quantity.sub {d : Dimensions} (a b : Quantity d) : Quantity d :=
  ⟨a.val - b.val⟩

def Quantity.mul {d1 d2 : Dimensions} (a : Quantity d1) (b : Quantity d2) : Quantity (d1 + d2) :=
  ⟨a.val * b.val⟩

def Quantity.div {d1 d2 : Dimensions} (a : Quantity d1) (b : Quantity d2) : Quantity (d1 - d2) :=
  ⟨a.val / b.val⟩

def Quantity.inv {d : Dimensions} (a : Quantity d) : Quantity (-d) :=
  ⟨1 / a.val⟩

instance {d : Dimensions} : Add (Quantity d) := ⟨Quantity.add⟩
instance {d : Dimensions} : Sub (Quantity d) := ⟨Quantity.sub⟩

-- Note: Mul and Div cannot be instances because result dimensions depend on inputs
-- Use Quantity.mul and Quantity.div directly, or define custom operators

/-! ## Constants -/

def zero {d : Dimensions} : Quantity d := ⟨0⟩

end QFD.Schema

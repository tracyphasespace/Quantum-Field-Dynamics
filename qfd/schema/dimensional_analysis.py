"""
QFD Schema: Dimensional Analysis

Type-safe physical quantities to prevent dimensional errors in Python code.
Mirrors the Lean implementation in QFD/Schema/DimensionalAnalysis.lean

References:
    - QFD/Schema/DimensionalAnalysis.lean: Type-safe Lean implementation
    - QFD/Schema/Couplings.lean: Parameter declarations with dimensions
"""

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True)
class Dimensions:
    """
    Physical dimensions tracking [L^l M^m T^t Q^q]

    Attributes:
        length: Length dimension exponent [L]
        mass: Mass dimension exponent [M]
        time: Time dimension exponent [T]
        charge: Charge dimension exponent [Q]

    Example:
        >>> velocity = Dimensions(length=1, mass=0, time=-1, charge=0)  # [L T⁻¹]
        >>> energy = Dimensions(length=2, mass=1, time=-2, charge=0)    # [M L² T⁻²]
    """
    length: int = 0
    mass: int = 0
    time: int = 0
    charge: int = 0

    def __add__(self, other: 'Dimensions') -> 'Dimensions':
        """Dimension addition (for multiplication of quantities)."""
        return Dimensions(
            length=self.length + other.length,
            mass=self.mass + other.mass,
            time=self.time + other.time,
            charge=self.charge + other.charge,
        )

    def __sub__(self, other: 'Dimensions') -> 'Dimensions':
        """Dimension subtraction (for division of quantities)."""
        return Dimensions(
            length=self.length - other.length,
            mass=self.mass - other.mass,
            time=self.time - other.time,
            charge=self.charge - other.charge,
        )

    def __neg__(self) -> 'Dimensions':
        """Dimension negation (for inversion of quantities)."""
        return Dimensions(
            length=-self.length,
            mass=-self.mass,
            time=-self.time,
            charge=-self.charge,
        )

    def __eq__(self, other: object) -> bool:
        """Check dimensional equality."""
        if not isinstance(other, Dimensions):
            return False
        return (
            self.length == other.length
            and self.mass == other.mass
            and self.time == other.time
            and self.charge == other.charge
        )

    def __str__(self) -> str:
        """Human-readable dimension string."""
        parts = []
        if self.length != 0:
            parts.append(f"L^{self.length}" if self.length != 1 else "L")
        if self.mass != 0:
            parts.append(f"M^{self.mass}" if self.mass != 1 else "M")
        if self.time != 0:
            parts.append(f"T^{self.time}" if self.time != 1 else "T")
        if self.charge != 0:
            parts.append(f"Q^{self.charge}" if self.charge != 1 else "Q")
        return " ".join(parts) if parts else "Unitless"


# ============================================================================
# Fundamental Dimension Definitions
# ============================================================================

UNITLESS = Dimensions(0, 0, 0, 0)
LENGTH = Dimensions(1, 0, 0, 0)
MASS = Dimensions(0, 1, 0, 0)
TIME = Dimensions(0, 0, 1, 0)
CHARGE = Dimensions(0, 0, 0, 1)

# Derived dimensions
VELOCITY = Dimensions(1, 0, -1, 0)       # [L T⁻¹]
ENERGY = Dimensions(2, 1, -2, 0)         # [M L² T⁻²]
FORCE = Dimensions(1, 1, -2, 0)          # [M L T⁻²]
ACTION = Dimensions(2, 1, -1, 0)         # [M L² T⁻¹] (Angular momentum / Planck)
DENSITY = Dimensions(-3, 1, 0, 0)        # [M L⁻³]
GRAVITATIONAL = Dimensions(3, -1, -2, 0) # [L³ M⁻¹ T⁻²] (G constant)


class Quantity:
    """
    Physical quantity with value and type-safe dimensions.

    Attributes:
        value: Numerical value (float or array)
        dims: Physical dimensions

    Example:
        >>> v = Quantity(3e8, VELOCITY)  # Speed of light
        >>> t = Quantity(1.0, TIME)      # One second
        >>> d = v * t                    # Distance
        >>> print(d.dims)
        L

    References:
        - QFD/Schema/DimensionalAnalysis.lean:39 (Quantity definition)
    """

    def __init__(self, value: Union[float, np.ndarray], dims: Dimensions):
        self.value = value
        self.dims = dims

    def __add__(self, other: 'Quantity') -> 'Quantity':
        """
        Add two quantities (must have same dimensions).

        Raises:
            DimensionalError: If dimensions don't match
        """
        if self.dims != other.dims:
            raise DimensionalError(
                f"Cannot add {self.dims} + {other.dims}"
            )
        return Quantity(self.value + other.value, self.dims)

    def __sub__(self, other: 'Quantity') -> 'Quantity':
        """
        Subtract two quantities (must have same dimensions).

        Raises:
            DimensionalError: If dimensions don't match
        """
        if self.dims != other.dims:
            raise DimensionalError(
                f"Cannot subtract {self.dims} - {other.dims}"
            )
        return Quantity(self.value - other.value, self.dims)

    def __mul__(self, other: 'Quantity') -> 'Quantity':
        """
        Multiply two quantities (dimensions add).

        References:
            - QFD/Schema/DimensionalAnalysis.lean:64 (Quantity.mul)
        """
        return Quantity(
            self.value * other.value,
            self.dims + other.dims
        )

    def __truediv__(self, other: 'Quantity') -> 'Quantity':
        """
        Divide two quantities (dimensions subtract).

        References:
            - QFD/Schema/DimensionalAnalysis.lean:67 (Quantity.div)
        """
        return Quantity(
            self.value / other.value,
            self.dims - other.dims
        )

    def __pow__(self, exponent: int) -> 'Quantity':
        """Raise quantity to integer power (dimensions scale)."""
        return Quantity(
            self.value ** exponent,
            Dimensions(
                length=self.dims.length * exponent,
                mass=self.dims.mass * exponent,
                time=self.dims.time * exponent,
                charge=self.dims.charge * exponent,
            )
        )

    def __str__(self) -> str:
        return f"{self.value} [{self.dims}]"

    def __repr__(self) -> str:
        return f"Quantity({self.value}, {self.dims})"

    def is_unitless(self) -> bool:
        """Check if quantity is dimensionless."""
        return self.dims == UNITLESS

    def has_dimensions(self, dims: Dimensions) -> bool:
        """Check if quantity has specific dimensions."""
        return self.dims == dims


class DimensionalError(Exception):
    """Raised when dimensional analysis detects an error."""
    pass


# ============================================================================
# Schema Parameter Parsing
# ============================================================================

def parse_schema_units(units_str: str) -> Dimensions:
    """
    Parse unit string from schema JSON to Dimensions.

    Args:
        units_str: Unit specification from schema
            Examples: "dimensionless", "MeV", "km/s/Mpc", "eV"

    Returns:
        Dimensions object

    Example:
        >>> parse_schema_units("dimensionless")
        Dimensions(0, 0, 0, 0)
        >>> parse_schema_units("MeV")
        Dimensions(2, 1, -2, 0)  # Energy
    """
    units_lower = units_str.lower().strip()

    # Map common schema units to dimensions
    unit_map = {
        "dimensionless": UNITLESS,
        "unitless": UNITLESS,
        "": UNITLESS,
        # Energy
        "mev": ENERGY,
        "ev": ENERGY,
        "gev": ENERGY,
        "j": ENERGY,
        "joule": ENERGY,
        # Mass
        "kg": MASS,
        "mev/c^2": MASS,  # Natural units
        "mev/c2": MASS,
        # Length
        "m": LENGTH,
        "fm": LENGTH,
        "km": LENGTH,
        # Time
        "s": TIME,
        "sec": TIME,
        # Velocity
        "m/s": VELOCITY,
        "km/s": VELOCITY,
        # Density
        "kg/m^3": DENSITY,
        "mev/fm^3": DENSITY,
        # Compound (Hubble)
        "km/s/mpc": Dimensions(0, 0, -1, 0),  # [T⁻¹]
        "(km/s)/mpc": Dimensions(0, 0, -1, 0),
    }

    if units_lower in unit_map:
        return unit_map[units_lower]
    else:
        raise ValueError(
            f"Unknown unit '{units_str}'. "
            "Add to dimensional_analysis.py:parse_schema_units()"
        )


def create_quantity_from_schema(value: float, units: str) -> Quantity:
    """
    Create a dimensionally-typed Quantity from schema parameter.

    Args:
        value: Numerical value
        units: Unit string from schema JSON

    Returns:
        Quantity with correct dimensions

    Example:
        >>> c1 = create_quantity_from_schema(0.496, "dimensionless")
        >>> c1.dims == UNITLESS
        True
    """
    dims = parse_schema_units(units)
    return Quantity(value, dims)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_expression(
    expr_name: str,
    expected_dims: Dimensions,
    actual_dims: Dimensions
) -> None:
    """
    Validate that an expression has expected dimensions.

    Args:
        expr_name: Name of expression for error message
        expected_dims: Required dimensions
        actual_dims: Computed dimensions

    Raises:
        DimensionalError: If dimensions don't match

    Example:
        >>> validate_expression("E = mc²", ENERGY, MASS + VELOCITY + VELOCITY)
        # Passes: [M L² T⁻²] == [M] + [L T⁻¹] + [L T⁻¹]
    """
    if expected_dims != actual_dims:
        raise DimensionalError(
            f"Expression '{expr_name}' has wrong dimensions:\n"
            f"  Expected: {expected_dims}\n"
            f"  Got:      {actual_dims}"
        )


def check_unitless(quantity: Quantity, context: str = "") -> None:
    """
    Verify that a quantity is dimensionless.

    Useful for exponents, dimensionless couplings, etc.

    Args:
        quantity: Quantity to check
        context: Description for error message

    Raises:
        DimensionalError: If not unitless
    """
    if not quantity.is_unitless():
        raise DimensionalError(
            f"{context}: Expected unitless, got {quantity.dims}"
        )


if __name__ == "__main__":
    # Self-test
    print("Testing QFD Dimensional Analysis")
    print("=" * 50)

    # Test 1: Create quantities
    c1 = Quantity(0.496, UNITLESS)
    c2 = Quantity(0.324, UNITLESS)
    print(f"✓ c1 = {c1}")
    print(f"✓ c2 = {c2}")

    # Test 2: Parse schema units
    dims_energy = parse_schema_units("MeV")
    print(f"✓ Parsed 'MeV' → {dims_energy}")
    assert dims_energy == ENERGY

    # Test 3: Dimensional arithmetic
    v = Quantity(3e8, VELOCITY)  # m/s
    t = Quantity(1.0, TIME)      # s
    d = v * t                     # distance
    print(f"✓ v·t = {d}")
    assert d.dims == LENGTH

    # Test 4: Error detection
    try:
        _ = c1 + v  # Should fail: unitless + velocity
        print("✗ FAILED: Should have caught dimensional error")
    except DimensionalError as e:
        print(f"✓ Caught error: {e}")

    # Test 5: Nuclear parameters
    A = Quantity(12, UNITLESS)  # Mass number (dimensionless integer)
    Q = c1 * (A ** (2/3)) + c2 * A
    print(f"✓ Q(A=12) = {Q}")
    assert Q.is_unitless()

    print("\n✅ All dimensional analysis tests passed!")

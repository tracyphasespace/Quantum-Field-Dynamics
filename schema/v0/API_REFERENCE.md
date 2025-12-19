# Consistency Checker API Reference

## Module: check_lean_json_consistency.py

### Overview

Automated tool to verify JSON parameter configurations comply with Lean4 type-safe schema definitions.

---

## Data Classes

### `Dimensions`

Physical dimensions for dimensional analysis.

```python
@dataclass
class Dimensions:
    length: int   # Power of Length dimension
    mass: int     # Power of Mass dimension
    time: int     # Power of Time dimension
    charge: int   # Power of Charge dimension
```

**Methods**:

#### `parse_lean(lean_str: str) -> Optional[Dimensions]`

Parse Lean4 dimension notation.

```python
# Example
dims = Dimensions.parse_lean("⟨1, 0, -1, 0⟩")
# Returns: Dimensions(length=1, mass=0, time=-1, charge=0)
# Represents: L¹T⁻¹ (velocity)
```

#### `to_string() -> str`

Convert to human-readable string.

```python
Dimensions(1, 0, -1, 0).to_string()
# Returns: "L T^-1"

Dimensions(0, 0, 0, 0).to_string()
# Returns: "dimensionless"
```

---

### `LeanParameter`

Parameter definition from Lean4 schema.

```python
@dataclass
class LeanParameter:
    name: str              # Parameter name (e.g., "c1")
    type_name: str         # Lean type (e.g., "Unitless", "Energy")
    dimensions: Optional[Dimensions]  # Physical dimensions if specified
    comment: str           # Inline comment from Lean file
```

**Example**:
```python
LeanParameter(
    name="c1",
    type_name="Unitless",
    dimensions=Dimensions(0, 0, 0, 0),
    comment="Surface term"
)
```

---

### `LeanConstraint`

Constraint/bounds from Lean4 schema.

```python
@dataclass
class LeanConstraint:
    param_name: str                          # Parameter name
    bounds: Optional[Tuple[float, float]]    # (min, max) if specified
    constraints: List[str]                   # Other constraints (e.g., ["positive"])
```

**Example**:
```python
LeanConstraint(
    param_name="c1",
    bounds=(0.5, 1.5),
    constraints=[]
)
```

---

### `JsonParameter`

Parameter definition from JSON config.

```python
@dataclass
class JsonParameter:
    name: str                                # Full name (e.g., "nuclear.c1")
    value: float                             # Initial/current value
    role: str                                # "coupling", "nuisance", "calibration"
    bounds: Optional[Tuple[float, float]]    # (min, max) if specified
    units: str                               # Unit string (e.g., "dimensionless")
    frozen: bool                             # Whether parameter is fixed
```

**Example**:
```python
JsonParameter(
    name="nuclear.c1",
    value=1.0,
    role="coupling",
    bounds=(0.5, 1.5),
    units="dimensionless",
    frozen=False
)
```

---

### `ConsistencyIssue`

A detected inconsistency between Lean4 and JSON.

```python
@dataclass
class ConsistencyIssue:
    severity: str           # "error", "warning", or "info"
    category: str           # Issue category
    message: str            # Human-readable description
    lean_detail: Optional[str]  # Details from Lean4 side
    json_detail: Optional[str]  # Details from JSON side
```

**Categories**:
- `missing_lean_definition` - JSON param not in Lean (error)
- `json_only_nuisance` - Nuisance param not in Lean (info)
- `bounds_mismatch` - JSON bounds don't contain Lean bounds (warning/error)
- `missing_bounds` - Lean has bounds but JSON doesn't (warning)
- `frozen_with_bounds` - Frozen param has non-trivial bounds (info)
- `units_naming` - Inconsistent units naming (info)
- `units_mismatch` - Units don't match Lean type (warning)
- `incomplete_json` - Missing Lean params in JSON (info)

---

## Parser Classes

### `LeanSchemaParser`

Extracts parameters and constraints from Lean4 schema files.

```python
class LeanSchemaParser:
    def __init__(self, lean_dir: Path)
```

**Methods**:

#### `parse_couplings() -> Dict[str, List[LeanParameter]]`

Parse `QFD/Schema/Couplings.lean` to extract parameter definitions.

```python
parser = LeanSchemaParser(Path("projects/Lean4"))
params = parser.parse_couplings()

# Returns:
# {
#   "NuclearParams": [
#     LeanParameter(name="c1", type_name="Unitless", ...),
#     LeanParameter(name="c2", type_name="Unitless", ...),
#     ...
#   ],
#   "CosmoParams": [...],
#   ...
# }
```

**Parsing Logic**:
1. Find all `structure XParams where` blocks
2. Extract parameter lines: `name : Type -- comment`
3. Parse dimensions from type if present: `Quantity ⟨1, 0, -1, 0⟩`
4. Group by structure name

#### `parse_constraints() -> Dict[str, List[LeanConstraint]]`

Parse `QFD/Schema/Constraints.lean` to extract bounds and constraints.

```python
parser = LeanSchemaParser(Path("projects/Lean4"))
constraints = parser.parse_constraints()

# Returns:
# {
#   "NuclearParams": [
#     LeanConstraint(param_name="c1", bounds=(0.5, 1.5), ...),
#     ...
#   ]
# }
```

**Supported Constraint Patterns**:
```lean
-- Range constraint
c1_range : 0.5 < p.c1.val ∧ p.c1.val < 1.5
-- Parsed as: bounds=(0.5, 1.5)

-- Positivity constraint
c1_positive : p.c1.val > 0
-- Parsed as: constraints=["positive"]
```

---

### `JsonSchemaParser`

Extracts parameters from JSON configuration files.

```python
class JsonSchemaParser:
    def __init__(self, json_path: Path)
```

**Methods**:

#### `parse_parameters() -> List[JsonParameter]`

Parse parameters from JSON config.

```python
parser = JsonSchemaParser(Path("experiments/ccl_fit_v1.json"))
params = parser.parse_parameters()
```

**Supported Formats**:

1. **Simple list bounds**:
```json
{
  "name": "nuclear.c1",
  "value": 1.0,
  "role": "coupling",
  "bounds": [0.5, 1.5],
  "units": "dimensionless"
}
```

2. **Dict bounds**:
```json
{
  "name": "c1",
  "init": 1.0,
  "bounds": {"min": 0.5, "max": 1.5}
}
```

3. **File references** (skipped):
```json
{
  "parameters": [
    "path/to/param.json"  // Skipped, not a dict
  ]
}
```

---

## Checker Classes

### `ConsistencyChecker`

Main consistency checking logic.

```python
class ConsistencyChecker:
    def __init__(self, lean_dir: Path, strict: bool = False)
```

**Parameters**:
- `lean_dir`: Root directory of Lean4 project
- `strict`: If True, bounds mismatches are errors instead of warnings

**Methods**:

#### `check_json_config(json_path: Path) -> List[ConsistencyIssue]`

Run all consistency checks on a JSON config.

```python
checker = ConsistencyChecker(Path("projects/Lean4"))
issues = checker.check_json_config(Path("experiments/ccl_fit_v1.json"))
```

**Checks Performed** (in order):

1. **Parameter Existence** (`_check_json_params_in_lean`)
   - Verifies all coupling params in JSON exist in Lean
   - Allows nuisance/calibration params to be JSON-only

2. **Bounds Consistency** (`_check_bounds_consistency`)
   - Checks JSON bounds contain Lean bounds
   - Warns about frozen params with non-trivial bounds

3. **Units Consistency** (`_check_units_consistency`)
   - Verifies JSON units match Lean type
   - Checks for naming inconsistencies

4. **Coverage** (`_check_lean_coverage`)
   - Reports Lean params missing from JSON

---

### `ConsistencyReporter`

Formats and displays consistency issues.

```python
class ConsistencyReporter:
    def __init__(self, issues: List[ConsistencyIssue])
```

**Methods**:

#### `print_report() -> int`

Print human-readable report and return exit code.

```python
reporter = ConsistencyReporter(issues)
exit_code = reporter.print_report()
# Returns: 0 if no errors, 1 if errors found
```

**Output Format**:
```
============================================================
Checking: ccl_fit_v1.json
============================================================

❌ ERRORS (1):
  [missing_lean_definition] JSON parameter 'foo' not found
    JSON:  role=coupling, units=eV

⚠️  WARNINGS (1):
  [bounds_mismatch] Parameter 'c2': JSON bounds don't contain Lean
    Lean4: Lean: (0.0, 0.1)
    JSON:  JSON: [0.1, 1.0]

ℹ️  INFO (1):
  [incomplete_json] JSON config incomplete
    Lean4: Missing: V4, g_c, k_J

============================================================
Summary: 1 errors, 1 warnings, 1 info

❌ FAILED: Critical consistency errors found
```

---

## Command-Line Interface

### Usage

```bash
python check_lean_json_consistency.py [OPTIONS] [CONFIG]
```

### Arguments

- `config` (optional): JSON config file to check
  - Default: `experiments/ccl_fit_v1.json`

### Options

- `--strict`: Treat bounds mismatches as errors instead of warnings
- `--all`: Check all JSON files in `experiments/` and `examples/`
- `--lean-dir PATH`: Path to Lean4 project root
  - Default: `../../projects/Lean4`

### Examples

```bash
# Check default config
python check_lean_json_consistency.py

# Check specific config
python check_lean_json_consistency.py experiments/my_fit.json

# Check all configs
python check_lean_json_consistency.py --all

# Strict mode (CI/CD)
python check_lean_json_consistency.py --all --strict
```

### Exit Codes

- `0`: All checks passed (warnings/info allowed)
- `1`: Errors found (blocking issues)
- `2`: Usage error (invalid arguments)

---

## Workflow Integration

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
cd schema/v0
python check_lean_json_consistency.py --all
if [ $? -ne 0 ]; then
    echo "❌ Schema consistency check failed"
    exit 1
fi
```

### CI/CD (GitHub Actions)

```yaml
- name: Check Schema Consistency
  run: |
    cd schema/v0
    python check_lean_json_consistency.py --all --strict
```

### Makefile

```makefile
check-schema:
	cd schema/v0 && python check_lean_json_consistency.py --all

check-schema-strict:
	cd schema/v0 && python check_lean_json_consistency.py --all --strict
```

---

## Extension Points

### Adding New Constraint Types

Edit `LeanSchemaParser.parse_constraints()`:

```python
# Add pattern for new constraint type
custom_pattern = r'(\w+)_custom\s*:\s*custom_logic'
for match in re.finditer(custom_pattern, constraint_body):
    # Extract and store
```

### Adding New Units Mappings

Edit `ConsistencyChecker._check_units_consistency()`:

```python
lean_to_json = {
    "Unitless": ["dimensionless", "unitless"],
    "Energy": ["eV", "MeV", "GeV", "J"],
    "CustomType": ["custom_unit"],  # Add here
}
```

### Custom Issue Severity

Edit `ConsistencyChecker` methods:

```python
severity = "error" if self.strict else "warning"
# Customize based on category, parameter name, etc.
```

---

## Limitations

### Current

1. **Lean Parsing**: Regex-based, may miss complex Lean syntax
2. **File References**: Doesn't resolve external parameter files
3. **Derived Parameters**: Doesn't check parameter relationships
4. **Dimensional Checking**: Only validates units strings, not arithmetic

### Future Enhancements

1. **Lean AST Parsing**: Use Lean's built-in parser for robustness
2. **Reference Resolution**: Follow file references recursively
3. **Constraint Translation**: Convert Lean `Prop`s to JSON validators
4. **Dimensional Arithmetic**: Validate dimension consistency in formulas

---

## Troubleshooting

### "Lean schema not found"

```
FileNotFoundError: Lean4 schema not found: .../QFD/Schema/Couplings.lean
```

**Solution**: Specify correct `--lean-dir`:
```bash
python check_lean_json_consistency.py --lean-dir /path/to/Lean4
```

### "KeyError: 'value'"

```
KeyError: 'value'
```

**Solution**: Ensure parameters have either `value` or `init` field:
```json
{"name": "c1", "value": 1.0}  // OR
{"name": "c1", "init": 1.0}
```

### "Bounds mismatch" warnings

```
⚠️  Parameter 'c2': JSON bounds do not contain Lean bounds
```

**Solution**: Adjust JSON bounds to contain Lean bounds:
```json
{
  "bounds": [lean_lo, lean_hi]  // Must contain Lean bounds
}
```

---

## Testing

### Unit Tests (recommended)

```python
import pytest
from check_lean_json_consistency import *

def test_dimensions_parse():
    dims = Dimensions.parse_lean("⟨1, 0, -1, 0⟩")
    assert dims.length == 1
    assert dims.time == -1

def test_bounds_containment():
    checker = ConsistencyChecker(Path("..."))
    # Test bounds checking logic
```

### Integration Tests

```bash
# Should pass
python check_lean_json_consistency.py experiments/ccl_fit_v1.json
assert $? -eq 0

# Should fail (deliberately bad config)
python check_lean_json_consistency.py tests/bad_bounds.json
assert $? -eq 1
```

---

## References

- **Main Documentation**: `README_CONSISTENCY.md`
- **Analysis**: `LEAN_JSON_CONSISTENCY.md`
- **Lean Schema**: `projects/Lean4/QFD/Schema/`
- **JSON Schema**: `schema/v0/*.schema.json`

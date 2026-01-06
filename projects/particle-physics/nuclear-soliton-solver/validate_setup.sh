#!/bin/bash
# Quick validation script to verify setup

echo "====================================================================="
echo "QFD Nuclear Soliton Solver - Setup Validation"
echo "====================================================================="
echo ""

# Check directory structure
echo "Checking directory structure..."
for dir in src data docs results; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ (missing)"
    fi
done
echo ""

# Check core files
echo "Checking core files..."
files=(
    "src/qfd_solver.py"
    "src/qfd_metaopt_ame2020.py"
    "src/qfd_regional_calibration.py"
    "src/analyze_isotopes.py"
    "data/ame2020_system_energies.csv"
    "requirements.txt"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file (missing)"
    fi
done
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
if command -v python3 &> /dev/null; then
    echo "  ✓ python3 found: $(python3 --version)"

    # Check for key modules
    for module in numpy pandas torch scipy; do
        if python3 -c "import $module" 2>/dev/null; then
            echo "  ✓ $module installed"
        else
            echo "  ✗ $module not installed (run: pip install -r requirements.txt)"
        fi
    done
else
    echo "  ✗ python3 not found"
fi
echo ""

# Check AME2020 data
echo "Checking AME2020 data..."
if [ -f "data/ame2020_system_energies.csv" ]; then
    lines=$(wc -l < data/ame2020_system_energies.csv)
    echo "  ✓ AME2020 data: $lines lines (3558 isotopes + header)"

    # Show first few isotopes
    echo ""
    echo "  Sample data (first 5 isotopes):"
    head -6 data/ame2020_system_energies.csv | tail -5 | while read line; do
        echo "    $line"
    done
else
    echo "  ✗ AME2020 data file missing"
fi
echo ""

# Summary
echo "====================================================================="
echo "Setup validation complete!"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Validate Trial 32: python src/qfd_regional_calibration.py --validate-only"
echo "  3. Optimize heavy region: python src/qfd_regional_calibration.py --region heavy"
echo "====================================================================="

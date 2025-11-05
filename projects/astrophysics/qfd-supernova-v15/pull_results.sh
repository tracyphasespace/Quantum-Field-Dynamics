#!/bin/bash
# pull_results.sh - Sync latest publication figures and results to local machine
#
# Usage: ./pull_results.sh [--local-dir /path/to/destination]
#
# This script:
# 1. Pulls latest changes from the working branch
# 2. Lists all available figures and results
# 3. Copies them to a convenient local directory (default: ~/QFD_Results)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default local directory
LOCAL_DIR="${HOME}/QFD_Results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./pull_results.sh [--local-dir /path/to/destination]"
            echo ""
            echo "Options:"
            echo "  --local-dir PATH    Destination directory (default: ~/QFD_Results)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  QFD V15 Results Sync Script${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "${GREEN}✓${NC} Current branch: ${YELLOW}${CURRENT_BRANCH}${NC}"

# Pull latest changes
echo -e "\n${BLUE}[1/4]${NC} Pulling latest changes from remote..."
git fetch origin "${CURRENT_BRANCH}"
git pull origin "${CURRENT_BRANCH}" || {
    echo -e "${YELLOW}⚠${NC}  Already up to date or conflicts present"
}

# Show latest commit
LATEST_COMMIT=$(git log -1 --oneline)
echo -e "${GREEN}✓${NC} Latest commit: ${LATEST_COMMIT}"

# Create local directory
echo -e "\n${BLUE}[2/4]${NC} Creating local results directory..."
mkdir -p "${LOCAL_DIR}"/{figures,reports,validation_plots}
echo -e "${GREEN}✓${NC} Created: ${LOCAL_DIR}"

# Copy publication figures
echo -e "\n${BLUE}[3/4]${NC} Copying publication figures..."
FIGURES_SRC="results/mock_stage3/figures"
if [ -d "${FIGURES_SRC}" ]; then
    cp -r "${FIGURES_SRC}"/* "${LOCAL_DIR}/figures/" 2>/dev/null || true
    FIG_COUNT=$(find "${LOCAL_DIR}/figures" -name "*.png" | wc -l)
    echo -e "${GREEN}✓${NC} Copied ${FIG_COUNT} figures to ${LOCAL_DIR}/figures/"
else
    echo -e "${YELLOW}⚠${NC}  No mock_stage3 figures found"
fi

# Copy validation plots
echo -e "\n${BLUE}[3.5/4]${NC} Copying validation plots..."
if [ -d "validation_plots" ]; then
    cp validation_plots/*.png "${LOCAL_DIR}/validation_plots/" 2>/dev/null || true
    VAL_COUNT=$(find "${LOCAL_DIR}/validation_plots" -name "*.png" | wc -l)
    echo -e "${GREEN}✓${NC} Copied ${VAL_COUNT} validation plots to ${LOCAL_DIR}/validation_plots/"
fi

# Copy reports and data
echo -e "\n${BLUE}[4/4]${NC} Copying reports and data files..."
REPORTS_SRC="results/mock_stage3/reports"
if [ -d "${REPORTS_SRC}" ]; then
    cp "${REPORTS_SRC}"/*.csv "${LOCAL_DIR}/reports/" 2>/dev/null || true
    REPORT_COUNT=$(find "${LOCAL_DIR}/reports" -name "*.csv" | wc -l)
    echo -e "${GREEN}✓${NC} Copied ${REPORT_COUNT} CSV reports to ${LOCAL_DIR}/reports/"
else
    echo -e "${YELLOW}⚠${NC}  No reports found"
fi

# Copy Stage 3 results CSV if it exists
if [ -f "results/mock_stage3/stage3_results.csv" ]; then
    cp "results/mock_stage3/stage3_results.csv" "${LOCAL_DIR}/"
    echo -e "${GREEN}✓${NC} Copied stage3_results.csv"
fi

# Copy posterior samples if they exist
if [ -f "results/mock_stage3/posterior_samples.csv" ]; then
    cp "results/mock_stage3/posterior_samples.csv" "${LOCAL_DIR}/"
    echo -e "${GREEN}✓${NC} Copied posterior_samples.csv"
fi

# Copy documentation
echo -e "\n${BLUE}[Bonus]${NC} Copying documentation..."
cp PUBLICATION_FIGURES_SUMMARY.md "${LOCAL_DIR}/" 2>/dev/null || true
cp docs/PUBLICATION_TEMPLATE.md "${LOCAL_DIR}/" 2>/dev/null || true
cp docs/CODE_VERIFICATION.md "${LOCAL_DIR}/" 2>/dev/null || true
cp HOTFIX_VALIDATION.md "${LOCAL_DIR}/" 2>/dev/null || true
echo -e "${GREEN}✓${NC} Copied documentation files"

# Create index file
echo -e "\n${BLUE}[Final]${NC} Creating results index..."
cat > "${LOCAL_DIR}/INDEX.md" << EOF
# QFD V15 Results - Local Copy

**Synced on**: $(date)
**Git branch**: ${CURRENT_BRANCH}
**Latest commit**: ${LATEST_COMMIT}

## Directory Structure

\`\`\`
${LOCAL_DIR}/
├── figures/                    # Publication figures (Fig 4, 5, 6, 8)
├── validation_plots/           # Validation plots (Fig 1, 2, 3)
├── reports/                    # Per-survey diagnostic CSVs
├── stage3_results.csv          # Stage 3 residuals (300 SNe)
├── posterior_samples.csv       # MCMC samples (2000 draws)
├── PUBLICATION_FIGURES_SUMMARY.md
├── PUBLICATION_TEMPLATE.md
├── CODE_VERIFICATION.md
└── HOTFIX_VALIDATION.md
\`\`\`

## Available Figures

### Publication Figures (300 DPI)
EOF

# List all figures
for fig in "${LOCAL_DIR}"/figures/*.png; do
    if [ -f "$fig" ]; then
        SIZE=$(du -h "$fig" | cut -f1)
        BASENAME=$(basename "$fig")
        echo "- **${BASENAME}** (${SIZE})" >> "${LOCAL_DIR}/INDEX.md"
    fi
done

cat >> "${LOCAL_DIR}/INDEX.md" << EOF

### Validation Plots
EOF

for fig in "${LOCAL_DIR}"/validation_plots/*.png; do
    if [ -f "$fig" ]; then
        SIZE=$(du -h "$fig" | cut -f1)
        BASENAME=$(basename "$fig")
        echo "- **${BASENAME}** (${SIZE})" >> "${LOCAL_DIR}/INDEX.md"
    fi
done

cat >> "${LOCAL_DIR}/INDEX.md" << EOF

## Quick Links

**View figures**: Open \`${LOCAL_DIR}/figures/\` in your file browser

**Read summary**: Open \`PUBLICATION_FIGURES_SUMMARY.md\` in your markdown viewer

**Review data**: Load CSVs from \`reports/\` directory in Excel/Python/R

## Re-sync

To pull latest updates, run:
\`\`\`bash
cd $(pwd)
./pull_results.sh
\`\`\`

---
*Generated by pull_results.sh*
EOF

echo -e "${GREEN}✓${NC} Created INDEX.md"

# Final summary
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ SUCCESS!${NC} Results synced to: ${YELLOW}${LOCAL_DIR}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "📊 What's available:"
echo "   - $(find "${LOCAL_DIR}/figures" -name "*.png" 2>/dev/null | wc -l) publication figures"
echo "   - $(find "${LOCAL_DIR}/validation_plots" -name "*.png" 2>/dev/null | wc -l) validation plots"
echo "   - $(find "${LOCAL_DIR}/reports" -name "*.csv" 2>/dev/null | wc -l) CSV reports"
echo ""
echo "📂 Open results folder:"
echo "   ${LOCAL_DIR}"
echo ""
echo "📖 Read the index:"
echo "   cat ${LOCAL_DIR}/INDEX.md"
echo ""
echo "🔄 To re-sync later, just run: ./pull_results.sh"
echo ""

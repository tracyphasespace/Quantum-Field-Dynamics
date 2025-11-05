#!/bin/bash
# Quick morning status checker for V15 pipeline

echo "================================================================================"
echo "V15 PIPELINE MORNING STATUS CHECK"
echo "================================================================================"
echo ""

# Check if Stage 2 is still running
if pgrep -f "stage2_mcmc_optimized.py" > /dev/null; then
    echo "🔄 Stage 2 is STILL RUNNING"
    CURRENT_STEP=$(grep -oP '\d+(?=/5000)' stage2_mcmc.log | tail -1)
    PERCENT=$((CURRENT_STEP * 100 / 5000))
    echo "   Progress: $CURRENT_STEP/5,000 steps ($PERCENT%)"
    REMAINING=$((5000 - CURRENT_STEP))
    TIME_REMAINING=$((REMAINING * 4 / 3600))
    echo "   Time remaining: ~$TIME_REMAINING hours"
    echo ""
else
    echo "✅ Stage 2 COMPLETED"
    
    # Check if Stage 2 results exist
    if [ -d "results/v15_stage2_mcmc" ] && [ -f "results/v15_stage2_mcmc/chain.h5" ]; then
        echo "   ✅ Stage 2 results found"
    else
        echo "   ⚠️  Stage 2 results missing - may have failed"
    fi
    echo ""
    
    # Check Stage 3 status
    if [ -d "results/v15_stage3_hubble" ]; then
        echo "✅ Stage 3 COMPLETED"
        echo "   ✅ Full pipeline complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Review Hubble diagram: ls results/v15_stage3_hubble/"
        echo "  2. Analyze results: python analyze_final_results.py"
        echo "  3. Check OVERNIGHT_STATUS.md for details"
    else
        echo "⏭️  Stage 3 NOT STARTED"
        echo "   Run manually: python stage3_hubble_optimized.py \\"
        echo "                   --stage1-results results/v15_stage1_production \\"
        echo "                   --stage2-results results/v15_stage2_mcmc \\"
        echo "                   --out results/v15_stage3_hubble"
    fi
fi

echo ""
echo "================================================================================"
echo "For detailed status, see: OVERNIGHT_STATUS.md"
echo "================================================================================"

#!/bin/bash
# Quick status checker for V15 pipeline

echo "================================================================================"
echo "V15 PIPELINE STATUS"
echo "================================================================================"
echo ""

# Stage 1
completed=$(ls -d results/v15_stage1_production/*/ 2>/dev/null | wc -l)
total=5468
pct=$((completed * 100 / total))

if pgrep -f "stage1_optimize.py" > /dev/null; then
    status1="RUNNING"
else
    status1="COMPLETE"
fi

echo "STAGE 1: Per-SN Optimization [$status1]"
echo "  Progress: $completed/$total ($pct%)"

if [ -f stage1_production.log ]; then
    last_sn=$(tail -5 stage1_production.log | grep "Optimizing SNID" | tail -1 | sed 's/.*SNID=\([0-9]*\).*/\1/')
    if [ ! -z "$last_sn" ]; then
        echo "  Current: SNID $last_sn"
    fi

    # Show last few results
    echo "  Recent results:"
    tail -10 stage1_production.log | grep -E "SNID=" | sed 's/^/    /'
fi

echo ""

# Stage 2
if [ -d results/v15_stage2_mcmc ]; then
    if pgrep -f "stage2_mcmc.py" > /dev/null; then
        status2="RUNNING"
    else
        status2="COMPLETE"
    fi
    echo "STAGE 2: Global MCMC [$status2]"

    if [ -f stage2_mcmc.log ]; then
        echo "  Recent output:"
        tail -5 stage2_mcmc.log | sed 's/^/    /'
    fi
else
    echo "STAGE 2: Global MCMC [WAITING]"
fi

echo ""

# Stage 3
if [ -d results/v15_stage3_hubble ]; then
    if pgrep -f "stage3_hubble.py" > /dev/null; then
        status3="RUNNING"
    else
        status3="COMPLETE"
    fi
    echo "STAGE 3: Hubble Diagram [$status3]"

    if [ -f results/v15_stage3_hubble/summary.json ]; then
        echo "  Results:"
        python3 << 'PYEOF'
import json
with open('results/v15_stage3_hubble/summary.json') as f:
    data = json.load(f)
print(f"    SNe: {data['n_sne']}")
print(f"    QFD RMS: {data['statistics']['qfd_rms']:.3f} mag")
print(f"    ΛCDM RMS: {data['statistics']['lcdm_rms']:.3f} mag")
if data['statistics']['qfd_rms'] < data['statistics']['lcdm_rms']:
    pct = (1 - data['statistics']['qfd_rms']/data['statistics']['lcdm_rms'])*100
    print(f"    ✅ QFD better by {pct:.1f}%")
else:
    pct = (data['statistics']['qfd_rms']/data['statistics']['lcdm_rms'] - 1)*100
    print(f"    ⚠️  ΛCDM better by {pct:.1f}%")
PYEOF
    fi
else
    echo "STAGE 3: Hubble Diagram [WAITING]"
fi

echo ""
echo "================================================================================"
echo ""
echo "Log files:"
echo "  Stage 1: stage1_production.log"
echo "  Stage 2: stage2_mcmc.log"
echo "  Stage 3: stage3_hubble.log"
echo "  Pipeline monitor: pipeline_monitor.log"
echo ""

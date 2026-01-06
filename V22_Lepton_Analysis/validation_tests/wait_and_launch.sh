#!/bin/bash
# Wait for current run to finish, then launch 3-lepton

echo "Waiting for current run (PID 210376) to complete..."
while ps -p 210376 > /dev/null 2>&1; do
    sleep 30
done

echo ""
echo "Current run finished! Waiting 10 seconds for cleanup..."
sleep 10

echo "Launching 3-lepton fit..."
nohup python t3b_three_lepton_overnight.py > results/V22/logs/t3b_three_lepton.log 2>&1 &
new_pid=$!
echo $new_pid > t3b_three_lepton.pid

echo ""
echo "3-lepton run started with PID: $new_pid"
echo "Monitor with: tail -f results/V22/logs/t3b_three_lepton.log"
echo ""

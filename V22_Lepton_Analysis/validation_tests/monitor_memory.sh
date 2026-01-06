#!/bin/bash
# Monitor memory usage of t3b script

echo "Memory Monitor for t3b_restart_4gb.py"
echo "======================================"
echo ""
echo "Watching for python process running t3b_restart_4gb.py..."
echo "Press Ctrl+C to stop"
echo ""
printf "%-20s %10s %10s %10s\n" "TIME" "RSS_MB" "VSZ_MB" "CPU%"
echo "--------------------------------------------------------"

while true; do
    # Find the python process running our script
    pid=$(pgrep -f "t3b_restart_4gb.py" | head -1)

    if [ -z "$pid" ]; then
        echo "$(date '+%H:%M:%S')    No process found - waiting..."
    else
        # Get memory stats
        stats=$(ps -p $pid -o rss=,vsz=,pcpu= 2>/dev/null)
        if [ $? -eq 0 ]; then
            rss_kb=$(echo $stats | awk '{print $1}')
            vsz_kb=$(echo $stats | awk '{print $2}')
            cpu=$(echo $stats | awk '{print $3}')

            rss_mb=$((rss_kb / 1024))
            vsz_mb=$((vsz_kb / 1024))

            # Color code if memory exceeds thresholds
            if [ $rss_mb -gt 3800 ]; then
                # Red if over 3.8GB (danger zone)
                printf "\033[0;31m%-20s %10d %10d %10s\033[0m\n" "$(date '+%H:%M:%S')" "$rss_mb" "$vsz_mb" "$cpu"
            elif [ $rss_mb -gt 3500 ]; then
                # Yellow if over 3.5GB (warning)
                printf "\033[0;33m%-20s %10d %10d %10s\033[0m\n" "$(date '+%H:%M:%S')" "$rss_mb" "$vsz_mb" "$cpu"
            else
                # Green if under 3.5GB (good)
                printf "\033[0;32m%-20s %10d %10d %10s\033[0m\n" "$(date '+%H:%M:%S')" "$rss_mb" "$vsz_mb" "$cpu"
            fi
        fi
    fi

    sleep 10
done

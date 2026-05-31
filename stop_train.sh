#!/bin/bash

# Stop the UAV training task started by the matching nohup launch script.
# It targets the train.py process first, then cleans up the wrapper pipeline.

set -u

TRAIN_PATTERN="python xuance-master/examples/train.py"
WRAPPER_PATTERN="xuance-master/examples/train.py 2>&1"
LOG_FILE="train_log.txt"

echo "=================================================="
echo "Stop training time: $(date "+%Y-%m-%d %H:%M:%S")"
echo "=================================================="

train_pids=$(pgrep -f "$TRAIN_PATTERN" || true)
wrapper_pids=$(pgrep -f "$WRAPPER_PATTERN" || true)

all_pids=$(printf "%s\n%s\n" "$train_pids" "$wrapper_pids" | awk 'NF' | sort -u)

if [ -z "$all_pids" ]; then
    echo "No running training process found."
    exit 0
fi

echo "Found related processes:"
for pid in $all_pids; do
    ps -p "$pid" -o pid,ppid,stat,cmd --no-headers
done

echo "Sending SIGTERM..."
for pid in $all_pids; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
done

sleep 5

still_alive=""
for pid in $all_pids; do
    if kill -0 "$pid" 2>/dev/null; then
        still_alive="$still_alive $pid"
    fi
done

if [ -n "$still_alive" ]; then
    echo "Processes still alive; sending SIGKILL:$still_alive"
    for pid in $still_alive; do
        kill -9 "$pid" 2>/dev/null || true
    done
else
    echo "Training process stopped."
fi

if [ -f "$LOG_FILE" ]; then
    {
        echo "=================================================="
        echo "Training manually stopped: $(date "+%Y-%m-%d %H:%M:%S")"
        echo "=================================================="
    } >> "$LOG_FILE"
    echo "Stop record appended to $LOG_FILE."
fi

echo "Done."

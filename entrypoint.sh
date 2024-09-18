#!/bin/bash

set -e

# Start Rasa server in the background
rasa run --enable-api --model "$MODEL_PATH" --port "$RASA_PORT" &

# Wait for Rasa server to start (adjust sleep time if needed)
sleep 300

cd action
# Run the benchmark script
python -m benchmark.bin check

# Capture the completion time
COMPLETION_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Set the output
echo "time=$COMPLETION_TIME" >> $GITHUB_OUTPUT

# Stop the Rasa server
pkill -f "rasa run"

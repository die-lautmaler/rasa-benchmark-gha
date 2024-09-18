#!/bin/bash

set -e

echo "Current working directory: $(pwd)"
echo "Contents of current directory:"
ls -la

# Start Rasa server in the background
echo "Starting Rasa server with model path: $MODEL_PATH"
rasa run --enable-api --model "$MODEL_PATH" --port "$RASA_PORT" &

# Wait for Rasa server to start (adjust sleep time if needed)
sleep 500

# Run the benchmark script
echo "Running benchmark script" 
python -m pip install -r requirements.txt
python -m benchmark.bin check --threshold $THRESHOLD --nlu-data-dir $TEST_DATA_PATH

# Capture the completion time
COMPLETION_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Set the output
echo "time=$COMPLETION_TIME" >> $GITHUB_OUTPUT

# Stop the Rasa server
pkill -f "rasa run"
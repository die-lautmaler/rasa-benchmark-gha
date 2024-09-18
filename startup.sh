#!/bin/bash

# Start Rasa server in the background
rasa run --enable-api --model ${model_path} --endpoints endpoints.yml --cors "*" &

# Wait for 300 seconds
sleep 300

# Run the Python script
python -m benchmark.bin check
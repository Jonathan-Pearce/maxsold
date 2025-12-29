#!/bin/bash

# Run ML Pipeline in background
echo "Starting ML Pipeline in background..."
cd /workspaces/maxsold

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the fast pipeline
nohup python model_pipeline_fast.py > model_pipeline.log 2>&1 &
PID=$!

echo "Pipeline started with PID: $PID"
echo "Monitor progress: tail -f model_pipeline.log"
echo "Check if running: ps -p $PID"

#!/bin/bash
# Script to run the ML model pipeline

echo "Starting MaxSold ML Pipeline..."
cd /workspaces/maxsold
python3 model_pipeline.py

echo ""
echo "Pipeline execution complete!"
echo "Check data/models/output/ for visualizations"
echo "Check data/models/ for saved model"

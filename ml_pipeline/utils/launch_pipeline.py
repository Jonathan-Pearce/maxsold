#!/usr/bin/env python3
"""
Launcher script - runs model pipeline in subprocess
"""
import subprocess
import sys
from pathlib import Path

def run_pipeline(script_name='model_pipeline_fast.py'):
    """Run the model pipeline"""
    script_path = Path(__file__).parent / script_name
    
    print(f"Launching {script_name}...")
    print("="*80)
    
    # Run the script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("SUCCESS! Pipeline completed.")
        print("="*80)
        print("\nCheck outputs:")
        print("  - Model: data/models/xgboost_model.pkl")
        print("  - Plots: data/models/output/")
        print("  - Metrics: data/models/output/metrics_summary.txt")
    else:
        print("\n" + "="*80)
        print(f"ERROR: Pipeline failed with code {result.returncode}")
        print("="*80)
        return result.returncode
    
    return 0

if __name__ == '__main__':
    script = sys.argv[1] if len(sys.argv) > 1 else 'model_pipeline_fast.py'
    exit_code = run_pipeline(script)
    sys.exit(exit_code)

"""
Quick test script to verify the model pipeline setup
"""
import sys
import pandas as pd
from pathlib import Path

# Get the repository root (2 levels up from this script)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
print("=" * 80)
print("VERIFYING ML PIPELINE SETUP")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✓ numpy")
    import pandas as pd
    print("   ✓ pandas")
    import matplotlib.pyplot as plt
    print("   ✓ matplotlib")
    import seaborn as sns
    print("   ✓ seaborn")
    import sklearn
    print("   ✓ scikit-learn")
    import xgboost as xgb
    print("   ✓ xgboost")
    import joblib
    print("   ✓ joblib")
    print("\n   All required packages are installed!")
except ImportError as e:
    print(f"\n   ✗ Error importing: {e}")
    sys.exit(1)

# Test data loading
print("\n2. Testing data loading...")
try:
    data_path = REPO_ROOT / 'data' / 'final_data' / 'maxsold_final_dataset.parquet'
    df = pd.read_parquet(data_path)
    print(f"   ✓ Successfully loaded dataset")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check for target column
    if 'current_bid' in df.columns:
        print(f"   ✓ Target column 'current_bid' found")
        print(f"   Target statistics:")
        print(f"     - Mean: ${df['current_bid'].mean():.2f}")
        print(f"     - Median: ${df['current_bid'].median():.2f}")
        print(f"     - Min: ${df['current_bid'].min():.2f}")
        print(f"     - Max: ${df['current_bid'].max():.2f}")
    else:
        print("   ✗ Target column 'current_bid' not found")
        sys.exit(1)
    
    # Check for bid_count column
    if 'bid_count' in df.columns:
        print(f"   ✓ 'bid_count' column found (will be excluded from model)")
    
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("SETUP VERIFICATION COMPLETE - Ready to run model_pipeline.py")
print("=" * 80)
print("\nTo run the full pipeline, execute:")
print("  python model_pipeline.py")

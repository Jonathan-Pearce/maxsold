#!/usr/bin/env python3
"""
Quick Start Script for MaxSold Feature Engineering Pipeline

This script provides a simple way to run the pipeline with common configurations.
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80 + "\n")

def main():
    print_header("MaxSold Feature Engineering Pipeline - Quick Start")
    
    print("Select an option:")
    print("1. Run full pipeline (download → transform → upload)")
    print("2. Run pipeline (skip download, use existing data)")
    print("3. Run pipeline (skip upload, local processing only)")
    print("4. Test modules only (no data processing)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print_header("Running Full Pipeline")
        print("This will:")
        print("  • Download 3 raw datasets from Kaggle")
        print("  • Apply feature engineering transformations")
        print("  • Save engineered datasets locally")
        print("  • Upload 3 engineered datasets to Kaggle")
        print("  • Merge all datasets into final dataset")
        print("  • Upload final dataset to Kaggle")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([sys.executable, "run_pipeline.py"])
        else:
            print("Cancelled.")
    
    elif choice == "2":
        print_header("Running Pipeline (Skip Download)")
        print("This will use existing raw data in ./data/raw/")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([sys.executable, "run_pipeline.py", "--skip-download"])
        else:
            print("Cancelled.")
    
    elif choice == "3":
        print_header("Running Pipeline (Local Only)")
        print("This will process data locally without uploading to Kaggle")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([sys.executable, "run_pipeline.py", "--skip-upload"])
        else:
            print("Cancelled.")
    
    elif choice == "4":
        print_header("Testing Modules")
        print("This will test all feature engineering modules")
        
        subprocess.run([sys.executable, "test_modules.py"])
    
    elif choice == "5":
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please run again.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

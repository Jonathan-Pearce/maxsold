#!/usr/bin/env python3
"""
Download and read a dataset from Kaggle using the Kaggle API.

Prerequisites:
1. Install kaggle package: pip install kaggle
2. Get API credentials from https://www.kaggle.com/settings/account
3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars

Usage:
    python download_kaggle_dataset.py
"""

import os
import pandas as pd
from pathlib import Path
import kaggle


def setup_kaggle_credentials(username=None, key=None):
    """
    Setup Kaggle API credentials from environment variables or parameters.
    
    Args:
        username: Kaggle username (optional if already configured)
        key: Kaggle API key (optional if already configured)
    """
    if username and key:
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        print("✓ Kaggle credentials set from parameters")
    elif 'KAGGLE_USERNAME' in os.environ and 'KAGGLE_KEY' in os.environ:
        print("✓ Kaggle credentials found in environment variables")
    else:
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        if kaggle_json.exists():
            print(f"✓ Using Kaggle credentials from {kaggle_json}")
        else:
            print("⚠ No Kaggle credentials found!")
            print("Please either:")
            print("  1. Download kaggle.json from https://www.kaggle.com/settings/account")
            print(f"     and place it in {kaggle_dir}")
            print("  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            print("  3. Pass username and key to setup_kaggle_credentials()")
            raise ValueError("Kaggle credentials not configured")


def download_kaggle_dataset(dataset_name, download_path='./kaggle_data'):
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset_name: Dataset identifier (e.g., 'username/dataset-name')
        download_path: Local directory to download to
    
    Returns:
        Path to downloaded dataset directory
    """
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading dataset: {dataset_name}")
    print(f"Destination: {download_path}")
    
    # Download dataset
    kaggle.api.dataset_download_files(
        dataset_name,
        path=download_path,
        unzip=True,
        quiet=False
    )
    
    print(f"✓ Dataset downloaded successfully")
    
    return download_path


def list_dataset_files(dataset_path):
    """List all files in the downloaded dataset"""
    dataset_path = Path(dataset_path)
    files = list(dataset_path.glob('*'))
    
    print(f"\nFiles in dataset ({len(files)} total):")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    return files


def read_dataset_file(file_path, file_type='auto'):
    """
    Read a dataset file into a pandas DataFrame.
    
    Args:
        file_path: Path to the file
        file_type: File type ('csv', 'parquet', 'json', 'excel', or 'auto')
    
    Returns:
        pandas DataFrame
    """
    file_path = Path(file_path)
    
    if file_type == 'auto':
        file_type = file_path.suffix.lower().lstrip('.')
    
    print(f"\nReading file: {file_path.name}")
    print(f"File type: {file_type}")
    
    readers = {
        'csv': pd.read_csv,
        'parquet': pd.read_parquet,
        'json': pd.read_json,
        'xlsx': pd.read_excel,
        'xls': pd.read_excel,
    }
    
    if file_type not in readers:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    df = readers[file_type](file_path)
    
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def search_kaggle_datasets(search_term, max_results=10):
    """
    Search for datasets on Kaggle.
    
    Args:
        search_term: Search query
        max_results: Maximum number of results to return
    
    Returns:
        List of dataset information
    """
    print(f"\nSearching Kaggle for: '{search_term}'")
    
    datasets = kaggle.api.dataset_list(
        search=search_term,
        page=1,
        max_size=max_results
    )
    
    print(f"\nFound {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset.ref}")
        print(f"   Title: {dataset.title}")
        print(f"   Size: {dataset.size}")
        print(f"   Downloads: {dataset.downloadCount}")
        print(f"   Updated: {dataset.lastUpdated}")
    
    return datasets


def main():
    """Example usage"""
    
    # Setup credentials (optional - will use default locations if not provided)
    try:
        setup_kaggle_credentials()
        # Or explicitly set:
        # setup_kaggle_credentials(username='your_username', key='your_api_key')
    except ValueError as e:
        print(f"\nError: {e}")
        return
    
    # Example 1: Search for datasets
    print("\n" + "="*80)
    print("SEARCHING FOR DATASETS")
    print("="*80)
    search_kaggle_datasets("house prices", max_results=5)
    
    # Example 2: Download a specific dataset
    print("\n" + "="*80)
    print("DOWNLOADING DATASET")
    print("="*80)
    
    # Replace with your desired dataset
    dataset_name = "vikrishnan/boston-house-prices"
    download_path = "./data/kaggle_datasets"
    
    try:
        # Download dataset
        dataset_dir = download_kaggle_dataset(dataset_name, download_path)
        
        # List files
        files = list_dataset_files(dataset_dir)
        
        # Read the first CSV file found
        csv_files = [f for f in files if f.suffix.lower() == '.csv']
        if csv_files:
            df = read_dataset_file(csv_files[0])
            
            # Display dataset info
            print("\n" + "="*80)
            print("DATASET PREVIEW")
            print("="*80)
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            print(f"\nColumn info:")
            print(df.info())
            
            print(f"\nBasic statistics:")
            print(df.describe())
        else:
            print("\nNo CSV files found in dataset")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
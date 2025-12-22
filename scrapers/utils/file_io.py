"""File I/O utilities for saving and loading data"""

import pandas as pd
import json
from typing import Any, Dict, List
from pathlib import Path
import sys


def save_to_parquet(data: List[Dict[str, Any]], output_path: str, schema_config: Dict[str, Any] = None):
    """
    Save data to parquet file with optional schema transformations
    
    Args:
        data: List of dictionaries to save
        output_path: Path to output parquet file
        schema_config: Optional dict with keys:
            - numeric_cols: List of column names to convert to numeric
            - boolean_cols: List of column names to convert to boolean
            - datetime_cols: List of column names to convert to datetime
    """
    if not data:
        print("No data to save.", file=sys.stderr)
        return
    
    df = pd.DataFrame(data)
    
    # Apply schema transformations if provided
    if schema_config:
        # Convert numeric columns
        for col in schema_config.get('numeric_cols', []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        for col in schema_config.get('boolean_cols', []):
            if col in df.columns:
                df[col] = df[col].astype('boolean')
        
        # Convert datetime columns
        for col in schema_config.get('datetime_cols', []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\nSaved {len(df)} records to {output_path}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f} KB")


def load_from_parquet(parquet_path: str, columns: List[str] = None) -> pd.DataFrame:
    """
    Load data from parquet file
    
    Args:
        parquet_path: Path to parquet file
        columns: Optional list of specific columns to load
        
    Returns:
        DataFrame with loaded data
    """
    try:
        if columns:
            df = pd.read_parquet(parquet_path, columns=columns)
        else:
            df = pd.read_parquet(parquet_path)
        return df
    except Exception as e:
        print(f"Error loading parquet file {parquet_path}: {e}", file=sys.stderr)
        raise


def save_to_json(data: Any, output_path: str, indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save (should be JSON-serializable)
        output_path: Path to output JSON file
        indent: Indentation level for pretty printing
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    print(f"Saved data to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f} KB")


def load_from_json(json_path: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}", file=sys.stderr)
        raise

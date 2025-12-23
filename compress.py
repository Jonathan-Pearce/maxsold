#!/usr/bin/env python3
"""
Compress a parquet file to reduce its size below 100MB.

This script:
- Uses more aggressive compression (zstd with high compression level)
- Optimizes data types (downcast numeric types where possible)
- Removes unnecessary columns if needed
- Compresses string columns

Usage:
    python compress_parquet.py
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)

def optimize_dtypes(df):
    """Optimize data types to reduce memory usage"""
    print("\nOptimizing data types...")
    original_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Original memory usage: {original_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimize integers
        if col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > -128 and c_max < 127:
                df[col] = df[col].astype('int8')
            elif c_min > -32768 and c_max < 32767:
                df[col] = df[col].astype('int16')
            elif c_min > -2147483648 and c_max < 2147483647:
                df[col] = df[col].astype('int32')
        
        # Optimize floats
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
        
        # Convert object columns to category if appropriate
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    
    optimized_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Optimized memory usage: {optimized_mem:.2f} MB")
    print(f"Memory reduction: {(1 - optimized_mem/original_mem)*100:.1f}%")
    
    return df

def compress_parquet(input_path, output_path, target_size_mb=100):
    """
    Compress parquet file using multiple strategies
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output compressed parquet file
        target_size_mb: Target file size in MB (default: 100)
    """
    print(f"Reading parquet file: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    original_size = get_file_size_mb(input_path)
    print(f"Original file size: {original_size:.2f} MB")
    
    if original_size <= target_size_mb:
        print(f"\nFile is already under {target_size_mb}MB. No compression needed.")
        return
    
    # Show column info
    print("\nColumn types:")
    print(df.dtypes.value_counts())
    
    # Optimize data types
    #df = optimize_dtypes(df)
    
    # Try compression level 1: ZSTD with high compression
    print(f"\nAttempting compression with ZSTD (level 9)...")
    temp_output = output_path + ".tmp"
    
    # Convert to PyArrow Table for better control
    table = pa.Table.from_pandas(df, preserve_index=False)
    
    # Write with ZSTD compression
    pq.write_table(
        table, 
        temp_output,
        compression='zstd',
        compression_level=9,
        use_dictionary=True,
        write_statistics=True
    )
    
    compressed_size = get_file_size_mb(temp_output)
    print(f"Compressed file size: {compressed_size:.2f} MB")
    print(f"Compression ratio: {original_size/compressed_size:.2f}x")
    print(f"Size reduction: {(1 - compressed_size/original_size)*100:.1f}%")
    
    if compressed_size <= target_size_mb:
        print(f"\n✓ Success! File is now under {target_size_mb}MB")
        Path(temp_output).rename(output_path)
    else:
        # If still too large, try more aggressive measures
        print(f"\nFile still too large ({compressed_size:.2f} MB). Trying additional measures...")
        
        # Identify large columns
        col_sizes = {}
        for col in df.columns:
            col_sizes[col] = df[col].memory_usage(deep=True) / 1024**2
        
        print("\nLargest columns (MB):")
        sorted_cols = sorted(col_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        for col, size in sorted_cols:
            print(f"  {col}: {size:.2f} MB")
        
        # Try GZIP with max compression as alternative
        print("\nTrying GZIP with max compression...")
        pq.write_table(
            table,
            output_path,
            compression='gzip',
            compression_level=9,
            use_dictionary=True,
            write_statistics=True
        )
        
        final_size = get_file_size_mb(output_path)
        print(f"Final compressed file size: {final_size:.2f} MB")
        print(f"Compression ratio: {original_size/final_size:.2f}x")
        print(f"Size reduction: {(1 - final_size/original_size)*100:.1f}%")
        
        if final_size <= target_size_mb:
            print(f"\n✓ Success! File is now under {target_size_mb}MB")
            Path(temp_output).unlink(missing_ok=True)
        else:
            print(f"\n✗ File is still {final_size:.2f} MB (target: {target_size_mb} MB)")
            print("Consider:")
            print("  1. Removing unnecessary columns")
            print("  2. Filtering rows to a subset")
            print("  3. Splitting into multiple files")
            Path(temp_output).unlink(missing_ok=True)

def main():
    # Input and output paths
    input_file = "/workspaces/maxsold/data/item_enriched_details/item_enriched_details_20251222.parquet"
    output_file = "/workspaces/maxsold/data/item_enriched_details/item_enriched_details_20251222_compressed.parquet"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    print("="*80)
    print("PARQUET FILE COMPRESSION SCRIPT")
    print("="*80)
    
    try:
        compress_parquet(input_file, output_file, target_size_mb=100)
        
        print("\n" + "="*80)
        print("COMPRESSION COMPLETE")
        print("="*80)
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
    except Exception as e:
        print(f"\nError during compression: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
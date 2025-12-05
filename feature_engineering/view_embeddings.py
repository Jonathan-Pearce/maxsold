#!/usr/bin/env python3
# filepath: scripts/inspect_embeddings.py
import argparse
import numpy as np
import os
import textwrap

def inspect_npy(path, rows=5):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # Use mmap_mode='r' to avoid loading huge arrays fully into memory
    arr = np.load(path, mmap_mode='r')
    # Ensure it's an ndarray
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    print(f"Path: {path}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    try:
        print(f"Total elements: {arr.size}")
        print(f"Estimated bytes (nbytes): {arr.nbytes}")
    except Exception:
        pass
    print()
    # Print first few rows in a readable way
    r = max(0, int(rows))
    print(f"First {r} rows (or entries):")
    if arr.ndim == 1:
        for i, v in enumerate(arr[:r]):
            print(f"[{i}] {v!r}")
    else:
        # Format rows to avoid overwhelming the terminal
        for i, row in enumerate(arr[:r]):
            # Limit line length for readability
            row_str = np.array2string(row, precision=6, threshold=20, max_line_width=120)
            print(f"[{i}] {row_str}")
    # If there are more rows, print a short summary
    if arr.shape[0] > r:
        print(f"\n... ({arr.shape[0] - r} more rows omitted)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a .npy embeddings file: print shape and first rows"
    )
    parser.add_argument("path", help="Path to the .npy file")
    parser.add_argument("--rows", "-n", type=int, default=5, help="Number of rows to show (default: 5)")
    args = parser.parse_args()
    inspect_npy(args.path, rows=args.rows)
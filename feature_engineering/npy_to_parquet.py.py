#!/usr/bin/env python3
# filepath: scripts/npy_to_parquet.py
"""
Convert a 2D .npy embeddings matrix into a Parquet file with columns
'emb_0', 'emb_1', ... 'emb_{D-1}'.

Usage:
    python3 scripts/npy_to_parquet.py /path/to/description_embeddings.npy /path/to/output.parquet
"""

import argparse
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Tuple

def load_npy_memmap(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path, mmap_mode='r')
    if arr.ndim == 1:
        # make into shape (n,1)
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, found ndim={arr.ndim}")
    return arr

def chunked_parquet_write(arr: np.ndarray, out_path: str, chunk_size: int = 10000, compression: str = 'snappy'):
    n_rows, n_cols = arr.shape
    col_names = [f"emb_{i}" for i in range(n_cols)]

    # Prepare writer using first chunk to infer schema
    start = 0
    end = min(chunk_size, n_rows)
    first_chunk = pd.DataFrame(arr[start:end], columns=col_names)
    table = pa.Table.from_pandas(first_chunk, preserve_index=False)
    writer = pq.ParquetWriter(out_path, table.schema, compression=compression)

    writer.write_table(table)
    start = end

    while start < n_rows:
        end = min(start + chunk_size, n_rows)
        chunk_df = pd.DataFrame(arr[start:end], columns=col_names)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        writer.write_table(table)
        start = end

    writer.close()

def inspect_npy(path: str) -> Tuple[int,int,str]:
    arr = np.load(path, mmap_mode='r')
    if arr.ndim == 1:
        shape = (arr.shape[0], 1)
    else:
        shape = arr.shape
    return shape, str(arr.dtype)

def main():
    parser = argparse.ArgumentParser(description="Convert .npy embeddings to Parquet (chunked)")
    parser.add_argument("input_npy", help="Path to .npy embeddings file (2D array)")
    parser.add_argument("output_parquet", help="Path to output .parquet file")
    parser.add_argument("--chunk-size", "-c", type=int, default=10000, help="Rows per write chunk (default: 10000)")
    parser.add_argument("--compression", type=str, default="snappy", help="Parquet compression (snappy, gzip, zstd, none)")
    parser.add_argument("--preview-rows", type=int, default=5, help="Print shape and first N rows preview (default: 5)")
    args = parser.parse_args()

    inp = args.input_npy
    out = args.output_parquet
    chunk_size = max(1, args.chunk_size)

    print(f"Loading (memmap) from: {inp}")
    arr = load_npy_memmap(inp)
    n_rows, n_cols = arr.shape
    print(f"Array shape: {n_rows} rows x {n_cols} cols, dtype={arr.dtype}")

    # Preview first few rows
    pr = max(0, args.preview_rows)
    if pr > 0:
        preview = arr[:pr]
        print(f"\nPreview first {pr} rows:")
        for i, row in enumerate(preview):
            row_str = np.array2string(row, precision=6, threshold=20, max_line_width=200)
            print(f"[{i}] {row_str}")

    # Ensure output dir exists
    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nWriting Parquet to: {out} (chunk_size={chunk_size}, compression={args.compression})")
    chunked_parquet_write(arr, out, chunk_size=chunk_size, compression=args.compression)
    print("Done.")

if __name__ == "__main__":
    main()
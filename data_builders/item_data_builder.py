"""
Merge auction_details, auction_search, and items_details datasets using auction_id as the join key.
Strategy: Merge auction files first, then merge onto items.
"""

import pandas as pd
from pathlib import Path
import sys
import gc

def merge_datasets():
    """Merge three datasets on auction_id"""
    
    # Define file paths
    items_path = Path('/workspaces/maxsold/data/engineered_data/items_details/items_details_20251201_with_embeddings.parquet')
    auction_details_path = Path('/workspaces/maxsold/data/engineered_data/auction_details/auction_details_20251201_engineered.parquet')
    auction_search_path = Path('/workspaces/maxsold/data/engineered_data/auction_search/auction_search_20251201_engineered.parquet')
    
    # Output path
    output_dir = Path('/workspaces/maxsold/data/final_data/item_details')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'items_merged_20251201.parquet'
    
    print("="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    # Load auction_details
    print(f"\n1. Loading auction_details from:\n   {auction_details_path}")
    if not auction_details_path.exists():
        print(f"ERROR: File not found: {auction_details_path}")
        sys.exit(1)
    
    df_auction_details = pd.read_parquet(auction_details_path)
    print(f"   Shape: {df_auction_details.shape}")
    print(f"   Columns: {len(df_auction_details.columns)}")
    
    # Identify auction_id column in auction_details
    auction_details_id_col = None
    for col in ['auction_id', 'amAuctionId', 'auctionId', 'id']:
        if col in df_auction_details.columns:
            auction_details_id_col = col
            break
    
    if not auction_details_id_col:
        print(f"   Available columns: {df_auction_details.columns.tolist()[:10]}...")
        print("ERROR: Could not find auction_id column in auction_details")
        sys.exit(1)
    
    print(f"   Using join key: '{auction_details_id_col}'")
    print(f"   Data type: {df_auction_details[auction_details_id_col].dtype}")
    print(f"   Unique auctions: {df_auction_details[auction_details_id_col].nunique():,}")
    
    # Standardize the auction_id column name
    auction_id_col = 'auction_id'
    if auction_details_id_col != auction_id_col:
        df_auction_details = df_auction_details.rename(columns={auction_details_id_col: auction_id_col})
        print(f"   Renamed '{auction_details_id_col}' to '{auction_id_col}'")
    
    # Convert to string
    df_auction_details[auction_id_col] = df_auction_details[auction_id_col].astype(str)
    
    # Load auction_search
    print(f"\n2. Loading auction_search from:\n   {auction_search_path}")
    if not auction_search_path.exists():
        print(f"ERROR: File not found: {auction_search_path}")
        sys.exit(1)
    
    df_auction_search = pd.read_parquet(auction_search_path)
    print(f"   Shape: {df_auction_search.shape}")
    print(f"   Columns: {len(df_auction_search.columns)}")
    
    # Identify auction_id column in auction_search
    auction_search_id_col = None
    for col in ['amAuctionId', 'auction_id', 'auctionId']:
        if col in df_auction_search.columns:
            auction_search_id_col = col
            break
    
    if not auction_search_id_col:
        print(f"   Available columns: {df_auction_search.columns.tolist()[:10]}...")
        print("ERROR: Could not find auction_id column in auction_search")
        sys.exit(1)
    
    print(f"   Using join key: '{auction_search_id_col}'")
    print(f"   Data type: {df_auction_search[auction_search_id_col].dtype}")
    print(f"   Unique auctions: {df_auction_search[auction_search_id_col].nunique():,}")
    
    # Rename to standardize
    if auction_search_id_col != auction_id_col:
        df_auction_search = df_auction_search.rename(columns={auction_search_id_col: auction_id_col})
        print(f"   Renamed '{auction_search_id_col}' to '{auction_id_col}'")
    
    # Convert to string
    df_auction_search[auction_id_col] = df_auction_search[auction_id_col].astype(str)
    
    # Check for overlapping columns between auction datasets
    print("\n3. Checking for duplicate columns between auction datasets...")
    details_cols = set(df_auction_details.columns)
    search_cols = set(df_auction_search.columns)
    overlap = (details_cols & search_cols) - {auction_id_col}
    
    if overlap:
        print(f"   Overlapping columns: {overlap}")
        print(f"   Adding '_search' suffix to {len(overlap)} columns in auction_search")
        rename_dict = {col: f"{col}_search" for col in overlap}
        df_auction_search = df_auction_search.rename(columns=rename_dict)
    
    # Merge auction_details with auction_search first
    print(f"\n4. Merging auction_details with auction_search on '{auction_id_col}'...")
    df_auctions_combined = df_auction_details.merge(
        df_auction_search,
        on=auction_id_col,
        how='outer',  # Use outer to keep all auctions from both sources
        suffixes=('_details', '_search')
    )
    print(f"   Combined auction data shape: {df_auctions_combined.shape}")
    print(f"   Unique auctions in combined: {df_auctions_combined[auction_id_col].nunique():,}")
    
    # Free memory
    del df_auction_details, df_auction_search
    gc.collect()
    
    # Load items_details
    print(f"\n5. Loading items_details from:\n   {items_path}")
    if not items_path.exists():
        print(f"ERROR: File not found: {items_path}")
        sys.exit(1)
    
    df_items = pd.read_parquet(items_path)
    print(f"   Shape: {df_items.shape}")
    print(f"   Columns: {len(df_items.columns)}")
    
    # Identify auction_id column in items
    items_id_col = None
    for col in ['auction_id', 'amAuctionId', 'auctionId']:
        if col in df_items.columns:
            items_id_col = col
            break
    
    if not items_id_col:
        print(f"   Available columns: {df_items.columns.tolist()[:10]}...")
        print("ERROR: Could not find auction_id column in items_details")
        sys.exit(1)
    
    print(f"   Using join key: '{items_id_col}'")
    print(f"   Data type: {df_items[items_id_col].dtype}")
    print(f"   Unique auctions in items: {df_items[items_id_col].nunique():,}")
    
    # Rename to standardize
    if items_id_col != auction_id_col:
        df_items = df_items.rename(columns={items_id_col: auction_id_col})
        print(f"   Renamed '{items_id_col}' to '{auction_id_col}'")
    
    # Convert to string
    df_items[auction_id_col] = df_items[auction_id_col].astype(str)
    
    # Check for overlapping columns
    print("\n6. Checking for duplicate columns between items and combined auctions...")
    items_cols = set(df_items.columns)
    auctions_cols = set(df_auctions_combined.columns)
    overlap = (items_cols & auctions_cols) - {auction_id_col}
    
    if overlap:
        print(f"   Overlapping columns: {overlap}")
        print(f"   Adding '_auction' suffix to {len(overlap)} columns in auction data")
        rename_dict = {col: f"{col}_auction" for col in overlap}
        df_auctions_combined = df_auctions_combined.rename(columns=rename_dict)
    
    # Merge items with combined auction data
    print(f"\n7. Merging items with combined auction data on '{auction_id_col}'...")
    df_final = df_items.merge(
        df_auctions_combined,
        on=auction_id_col,
        how='left',  # Left join to keep all items
        suffixes=('', '_auction')
    )
    print(f"   Final shape: {df_final.shape}")
    
    # Free memory
    del df_items, df_auctions_combined
    gc.collect()
    
    # Summary statistics
    print("\n" + "="*70)
    print("MERGE SUMMARY")
    print("="*70)
    print(f"Final merged dataset:          {df_final.shape[0]:,} rows × {df_final.shape[1]:,} cols")
    
    # Check merge quality
    print(f"\n8. Checking merge quality...")
    null_counts = df_final.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if len(cols_with_nulls) > 0:
        print(f"   Columns with null values: {len(cols_with_nulls)}")
        print(f"   Top 5 columns with most nulls:")
        for col, count in cols_with_nulls.head(5).items():
            pct = 100 * count / len(df_final)
            print(f"     {col}: {count:,} ({pct:.1f}%)")
    
    # Save merged dataset
    print(f"\n9. Saving merged dataset to:\n   {output_path}")
    df_final.to_parquet(output_path, index=False, compression='snappy')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Display sample (exclude embedding columns for readability)
    print("\n10. Sample of merged data (first 3 rows, selected columns):")
    sample_cols = [auction_id_col]
    for col in df_final.columns:
        if len(sample_cols) < 15:
            if col not in sample_cols and not any(col.endswith(f'_emb_{i}') for i in range(200)):
                sample_cols.append(col)
    
    print(df_final[sample_cols].head(3).to_string())
    
    print("\n" + "="*70)
    print("MERGE COMPLETE")
    print("="*70)
    
    return df_final


if __name__ == "__main__":
    try:
        df_merged = merge_datasets()
    except MemoryError as e:
        print(f"\nMemory Error: {e}")
        print("Try closing other applications or using a machine with more RAM")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMerge interrupted by user")
        sys.exit(1)
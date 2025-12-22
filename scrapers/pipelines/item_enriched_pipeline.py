"""
Item enriched details pipeline
Handles batch fetching of AI-generated enriched item details and saves to parquet
Supports both parallel and sequential processing
"""

import sys
import pandas as pd
import glob
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from extractors.item_enriched import fetch_enriched_details, extract_enriched_data
from utils.file_io import save_to_parquet, load_from_parquet
from utils.config import DEFAULT_OUTPUT_DIRS

# Thread-safe lock for printing
print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)


def fetch_single_item_enriched(item_id: str, index: int, total: int) -> tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch enriched details for a single item (designed for parallel execution)
    
    Args:
        item_id: The item ID to fetch
        index: Current index in batch
        total: Total number of items
        
    Returns:
        Tuple of (index, enriched data dictionary or None, error message or None)
    """
    
    if not item_id:
        return index, None, "Missing item_id"
    
    try:
        data = fetch_enriched_details(item_id)
        result = extract_enriched_data(data, item_id)
        
        if result:
            thread_safe_print(f"[{index}/{total}] ✓ Item {item_id}: enriched data extracted")
        else:
            thread_safe_print(f"[{index}/{total}] ℹ Item {item_id}: No data")
        
        return index, result, None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            thread_safe_print(f"[{index}/{total}] ℹ Item {item_id}: Not found (404)")
            return index, None, None  # 404 is not an error, just no data
        error_msg = f"Item {item_id}: HTTP {e.response.status_code}"
        thread_safe_print(f"[{index}/{total}] ✗ {error_msg}", file=sys.stderr)
        return index, None, error_msg
    except Exception as e:
        error_msg = f"Item {item_id}: {str(e)}"
        thread_safe_print(f"[{index}/{total}] ✗ {error_msg}", file=sys.stderr)
        return index, None, error_msg


def fetch_enriched_for_multiple_items_parallel(
    item_ids: List[str], 
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch enriched details for multiple items using parallel requests
    
    Args:
        item_ids: List of item IDs to fetch
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of enriched data dictionaries
    """
    all_results = []
    errors = []
    total = len(item_ids)
    
    thread_safe_print(f"\nStarting parallel fetch with {max_workers} workers...")
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(fetch_single_item_enriched, item_id, i+1, total): item_id 
            for i, item_id in enumerate(item_ids)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_id):
            try:
                index, result, error = future.result()
                if error:
                    errors.append(error)
                if result:
                    all_results.append(result)
            except Exception as e:
                thread_safe_print(f"✗ Unexpected error: {e}", file=sys.stderr)
                errors.append(str(e))
    
    # Print error summary
    if errors:
        thread_safe_print(f"\n⚠ Encountered {len(errors)} errors during fetch")
        if len(errors) <= 10:
            for err in errors:
                thread_safe_print(f"  - {err}", file=sys.stderr)
        else:
            for err in errors[:10]:
                thread_safe_print(f"  - {err}", file=sys.stderr)
            thread_safe_print(f"  ... and {len(errors) - 10} more errors", file=sys.stderr)
    
    return all_results


def fetch_enriched_for_multiple_items(item_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch enriched details for multiple items (sequential)
    
    Args:
        item_ids: List of item IDs to fetch
        
    Returns:
        List of enriched data dictionaries
    """
    all_results = []
    
    for i, item_id in enumerate(item_ids, 1):
        if not item_id:
            continue
        
        print(f"[{i}/{len(item_ids)}] Processing item {item_id}...")
        try:
            data = fetch_enriched_details(item_id)
            result = extract_enriched_data(data, item_id)
            if result:
                all_results.append(result)
                print(f"  ✓ Extracted enriched data")
            else:
                print(f"  ℹ No data found")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  ℹ Item not found (404)")
            else:
                print(f"  ✗ Error fetching enriched details: HTTP {e.response.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ Error fetching enriched details: {e}", file=sys.stderr)
    
    return all_results


def load_item_ids_from_parquet(parquet_path: str, limit: Optional[int] = None) -> List[str]:
    """
    Load item IDs from parquet file
    
    Args:
        parquet_path: Path to parquet file (supports glob patterns)
        limit: Optional limit on number of items to load
        
    Returns:
        List of item IDs
    """
    try:
        # Handle glob patterns
        files = glob.glob(parquet_path)
        if not files:
            print(f"No files found matching: {parquet_path}", file=sys.stderr)
            return []
        
        # Read the most recent file if multiple matches
        parquet_file = sorted(files)[-1]
        print(f"Reading items from: {parquet_file}")
        
        df = load_from_parquet(parquet_file)
        
        # Try different possible column names for item ID
        id_column = None
        for col in ['id', 'amLotId', 'item_id', 'lotId']:
            if col in df.columns:
                id_column = col
                break
        
        if id_column is None:
            print(f"Could not find item ID column in {parquet_file}", file=sys.stderr)
            print(f"Available columns: {', '.join(df.columns.tolist())}", file=sys.stderr)
            return []
        
        # Apply limit if specified
        if limit and limit > 0:
            df = df.head(limit)
            print(f"Limited to first {limit} items")
        
        # Extract unique item IDs
        item_ids = df[id_column].dropna().astype(str).unique().tolist()
        
        print(f"Loaded {len(item_ids)} unique item IDs from column '{id_column}'")
        return item_ids
        
    except Exception as e:
        print(f"Error reading parquet file: {e}", file=sys.stderr)
        return []


def run_item_enriched_pipeline(
    item_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None,
    max_workers: int = 10,
    limit: Optional[int] = None,
    sequential: bool = False
):
    """
    Run the full item enriched details pipeline: fetch + save
    
    Args:
        item_ids: List of item IDs to fetch (optional if input_parquet provided)
        output_path: Path to save output parquet (default: auto-generated with timestamp)
        input_parquet: Path to parquet file to load item IDs from
        max_workers: Maximum number of parallel workers
        limit: Optional limit on number of items to process
        sequential: If True, use sequential processing instead of parallel
    """
    # Determine source of item IDs
    if input_parquet:
        item_ids = load_item_ids_from_parquet(input_parquet, limit=limit)
    elif item_ids is None or len(item_ids) == 0:
        # Default: load from most recent items details
        default_path = f"{DEFAULT_OUTPUT_DIRS['item_details']}/items_details_{datetime.now().strftime('%Y%m%d')}.parquet"
        print(f"No item IDs provided, loading from {default_path}...")
        item_ids = load_item_ids_from_parquet(default_path, limit=limit)
    
    if not item_ids:
        print("No item IDs to process.", file=sys.stderr)
        return
    
    output_path = output_path or f"{DEFAULT_OUTPUT_DIRS['item_enriched']}/item_enriched_details_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Enriched Item Details Pipeline")
    print("=" * 60)
    print(f"Items to fetch: {len(item_ids)}")
    print(f"Mode: {'Sequential' if sequential else f'Parallel ({max_workers} workers)'}")
    if len(item_ids) <= 10:
        print(f"Item IDs: {', '.join(item_ids)}")
    else:
        print(f"  Showing first 10 of {len(item_ids)} item IDs:")
        print(f"  {', '.join(item_ids[:10])}")
        print(f"  ... and {len(item_ids) - 10} more")
    print("=" * 60)
    
    # Fetch all enriched details
    start_time = datetime.now()
    
    if sequential:
        results = fetch_enriched_for_multiple_items(item_ids)
    else:
        results = fetch_enriched_for_multiple_items_parallel(item_ids, max_workers=max_workers)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱ Fetch completed in {elapsed:.2f} seconds ({len(item_ids)/elapsed:.2f} items/sec)")
    
    if not results:
        print("No enriched item data retrieved.", file=sys.stderr)
        return
    
    # Define schema transformations
    schema_config = {
        'numeric_cols': ['brands_count', 'categories_count', 'items_count', 'attributes_count', 'numItems'],
        'boolean_cols': ['singleKeyItem'],
        'datetime_cols': []
    }
    
    # Save to parquet
    save_to_parquet(results, output_path, schema_config)
    
    # Print sample
    df = pd.read_parquet(output_path)
    print(f"\nSummary:")
    print(f"  Total items: {len(df)}")
    if 'amAuctionId' in df.columns:
        print(f"  Unique auctions: {df['amAuctionId'].nunique()}")
    if 'title' in df.columns:
        print(f"  Items with title: {df['title'].notna().sum()}")
    if 'description' in df.columns:
        print(f"  Items with description: {df['description'].notna().sum()}")
    if 'brand' in df.columns:
        print(f"  Items with brand: {df['brand'].notna().sum()}")
    if 'categories' in df.columns:
        print(f"  Items with categories: {df['categories'].notna().sum()}")
    if 'attributes' in df.columns:
        print(f"  Items with attributes: {df['attributes'].notna().sum()}")
    
    print("\nSample enriched items (first 5 rows):")
    display_cols = ['item_id', 'amAuctionId', 'title', 'brand', 'condition', 'brands_count', 'categories_count']
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head(5).to_string())

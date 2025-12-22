"""
Bid history pipeline
Handles batch fetching of bid history and saves to parquet
Supports both parallel and sequential processing
"""

import sys
import pandas as pd
import glob
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from extractors.bid_history import fetch_item_bid_history, extract_bids_from_json
from utils.file_io import save_to_parquet, load_from_parquet
from utils.config import DEFAULT_OUTPUT_DIRS

# Thread-safe lock for printing
print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)


def fetch_single_item_bids(item: Dict[str, str], index: int, total: int) -> tuple[int, List[Dict[str, Any]], Optional[str]]:
    """
    Fetch bids for a single item (designed for parallel execution)
    
    Args:
        item: Dictionary with auction_id and item_id
        index: Current index in batch
        total: Total number of items
        
    Returns:
        Tuple of (index, list of bid dictionaries, error message or None)
    """
    auction_id = item.get("auction_id", "")
    item_id = item.get("item_id", "")
    
    if not auction_id or not item_id:
        return index, [], f"Missing auction_id or item_id"
    
    try:
        data = fetch_item_bid_history(auction_id, item_id)
        bids = extract_bids_from_json(data, auction_id, item_id)
        
        if bids:
            thread_safe_print(f"[{index}/{total}] ✓ Auction {auction_id}, Item {item_id}: {len(bids)} bids")
        else:
            thread_safe_print(f"[{index}/{total}] ℹ Auction {auction_id}, Item {item_id}: No bids")
        
        return index, bids, None
    except Exception as e:
        error_msg = f"Auction {auction_id}, Item {item_id}: {str(e)}"
        thread_safe_print(f"[{index}/{total}] ✗ {error_msg}", file=sys.stderr)
        return index, [], error_msg


def fetch_bids_for_multiple_items_parallel(
    items: List[Dict[str, str]], 
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch bid history for multiple auction/item pairs using parallel requests
    
    Args:
        items: List of dictionaries with auction_id and item_id
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of all bid dictionaries
    """
    all_bids = []
    errors = []
    total = len(items)
    
    thread_safe_print(f"\nStarting parallel fetch with {max_workers} workers...")
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(fetch_single_item_bids, item, i+1, total): item 
            for i, item in enumerate(items)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_item):
            try:
                index, bids, error = future.result()
                if error:
                    errors.append(error)
                if bids:
                    all_bids.extend(bids)
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
    
    return all_bids


def fetch_bids_for_multiple_items(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Fetch bid history for multiple auction/item pairs (sequential)
    
    Args:
        items: List of dictionaries with auction_id and item_id
        
    Returns:
        List of all bid dictionaries
    """
    all_bids = []
    
    for i, item in enumerate(items, 1):
        auction_id = item.get("auction_id", "")
        item_id = item.get("item_id", "")
        
        if not auction_id or not item_id:
            continue
        
        print(f"[{i}/{len(items)}] Processing auction {auction_id}, item {item_id}...")
        try:
            data = fetch_item_bid_history(auction_id, item_id)
            bids = extract_bids_from_json(data, auction_id, item_id)
            if bids:
                all_bids.extend(bids)
                print(f"  ✓ Extracted {len(bids)} bids")
            else:
                print(f"  ℹ No bids found")
        except Exception as e:
            print(f"  ✗ Error fetching bids: {e}", file=sys.stderr)
    
    return all_bids


def load_items_from_parquet(parquet_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load auction_id and item_id pairs from parquet file
    
    Args:
        parquet_path: Path to parquet file (supports glob patterns)
        limit: Optional limit on number of items to load
        
    Returns:
        List of dictionaries with auction_id and item_id
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
        
        required_cols = ['auction_id', 'id']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"Missing columns {missing_cols} in {parquet_file}", file=sys.stderr)
            print(f"Available columns: {', '.join(df.columns.tolist())}", file=sys.stderr)
            return []
        
        # Optionally filter to items with bids
        if 'bid_count' in df.columns:
            original_count = len(df)
            df = df[df['bid_count'] > 0]
            print(f"Filtered to {len(df)} items with bids (from {original_count} total)")
        
        # Apply limit if specified
        if limit and limit > 0:
            df = df.head(limit)
            print(f"Limited to first {limit} items")
        
        # Create list of auction_id/item_id pairs
        items = []
        for _, row in df.iterrows():
            items.append({
                "auction_id": str(row["auction_id"]),
                "item_id": str(row["id"])
            })
        
        print(f"Loaded {len(items)} items from {parquet_file}")
        return items
        
    except Exception as e:
        print(f"Error reading parquet file: {e}", file=sys.stderr)
        return []


def run_bid_history_pipeline(
    items: Optional[List[Dict[str, str]]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None,
    max_workers: int = 10,
    limit: Optional[int] = None,
    sequential: bool = False
):
    """
    Run the full bid history pipeline: fetch + save
    
    Args:
        items: List of dicts with auction_id and item_id (optional if input_parquet provided)
        output_path: Path to save output parquet (default: auto-generated with timestamp)
        input_parquet: Path to parquet file to load items from
        max_workers: Maximum number of parallel workers
        limit: Optional limit on number of items to process
        sequential: If True, use sequential processing instead of parallel
    """
    # Determine source of items
    if input_parquet:
        items = load_items_from_parquet(input_parquet, limit=limit)
    elif items is None or len(items) == 0:
        # Default: load from most recent items details
        default_path = f"{DEFAULT_OUTPUT_DIRS['item_details']}/items_details_{datetime.now().strftime('%Y%m%d')}.parquet"
        print(f"No items provided, loading from {default_path}...")
        items = load_items_from_parquet(default_path, limit=limit)
    
    if not items:
        print("No items to process.", file=sys.stderr)
        return
    
    output_path = output_path or f"{DEFAULT_OUTPUT_DIRS['bid_history']}/bid_history_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Bid History Pipeline")
    print("=" * 60)
    print(f"Items to fetch: {len(items)}")
    print(f"Mode: {'Sequential' if sequential else f'Parallel ({max_workers} workers)'}")
    if len(items) <= 5:
        for item in items:
            print(f"  Auction {item['auction_id']}, Item {item['item_id']}")
    else:
        print(f"  Showing first 5 of {len(items)} items:")
        for item in items[:5]:
            print(f"  Auction {item['auction_id']}, Item {item['item_id']}")
        print(f"  ... and {len(items) - 5} more")
    print("=" * 60)
    
    # Fetch all bids
    start_time = datetime.now()
    
    if sequential:
        bids = fetch_bids_for_multiple_items(items)
    else:
        bids = fetch_bids_for_multiple_items_parallel(items, max_workers=max_workers)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱ Fetch completed in {elapsed:.2f} seconds ({len(items)/elapsed:.2f} items/sec)")
    
    if not bids:
        print("No bid history data retrieved.", file=sys.stderr)
        return
    
    # Define schema transformations
    schema_config = {
        'numeric_cols': ['amount', 'bid_sequence'],
        'boolean_cols': ['isproxy'],
        'datetime_cols': ['time_of_bid']
    }
    
    # Save to parquet
    save_to_parquet(bids, output_path, schema_config)
    
    # Print sample
    df = pd.read_parquet(output_path)
    print(f"\nSummary:")
    print(f"  Unique auctions: {df['auction_id'].nunique()}")
    print(f"  Unique items: {df['item_id'].nunique()}")
    print(f"  Total bids: {len(df)}")
    if 'isproxy' in df.columns:
        print(f"  Proxy bids: {df['isproxy'].sum()}")
    if 'amount' in df.columns:
        print(f"  Total bid amount: ${df['amount'].sum():,.2f}")
    
    print("\nSample bids (first 10 rows):")
    display_cols = ['auction_id', 'item_id', 'bid_sequence', 'time_of_bid', 'amount', 'isproxy']
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head(10).to_string())

"""
Item details pipeline
Handles batch fetching of item/lot details and saves to parquet
"""

import sys
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from extractors.item_details import fetch_auction_items, extract_items_from_json
from utils.file_io import save_to_parquet, load_from_parquet
from utils.config import DEFAULT_OUTPUT_DIRS


def fetch_items_for_multiple_auctions(auction_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch items for multiple auctions
    
    Args:
        auction_ids: List of auction IDs to fetch items for
        
    Returns:
        List of all item dictionaries
    """
    all_items = []
    
    for i, auction_id in enumerate(auction_ids, 1):
        print(f"[{i}/{len(auction_ids)}] Processing auction {auction_id}...")
        try:
            data = fetch_auction_items(auction_id)
            items = extract_items_from_json(data, auction_id)
            if items:
                all_items.extend(items)
                print(f"  ✓ Extracted {len(items)} items")
            else:
                print(f"  ✗ No items found", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ Error fetching auction {auction_id}: {e}", file=sys.stderr)
    
    return all_items


def load_auction_ids_from_parquet(parquet_path: str) -> List[str]:
    """
    Load unique auction IDs from parquet file
    
    Args:
        parquet_path: Path to parquet file with amAuctionId column
        
    Returns:
        List of unique auction IDs
    """
    try:
        df = load_from_parquet(parquet_path)
        
        if 'amAuctionId' not in df.columns:
            print(f"Column 'amAuctionId' not found in {parquet_path}", file=sys.stderr)
            print(f"Available columns: {', '.join(df.columns.tolist())}", file=sys.stderr)
            return []
        
        # Get unique auction IDs, drop nulls, convert to string
        auction_ids = df['amAuctionId'].dropna().unique().astype(str).tolist()
        
        print(f"Loaded {len(auction_ids)} unique auction IDs from {parquet_path}")
        return auction_ids
        
    except FileNotFoundError:
        print(f"File not found: {parquet_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error reading parquet file: {e}", file=sys.stderr)
        return []


def run_item_details_pipeline(
    auction_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None
):
    """
    Run the full item details pipeline: fetch + save
    
    Args:
        auction_ids: List of auction IDs to fetch items for (optional if input_parquet provided)
        output_path: Path to save output parquet (default: auto-generated with timestamp)
        input_parquet: Path to parquet file to load auction IDs from
    """
    # Determine source of auction IDs
    if input_parquet:
        auction_ids = load_auction_ids_from_parquet(input_parquet)
    elif auction_ids is None or len(auction_ids) == 0:
        # Default: load from most recent auction search
        default_path = f"{DEFAULT_OUTPUT_DIRS['auction_search']}/auction_search_{datetime.now().strftime('%Y%m%d')}.parquet"
        print(f"No auction IDs provided, loading from {default_path}...")
        auction_ids = load_auction_ids_from_parquet(default_path)
    
    if not auction_ids:
        print("No auction IDs to process.", file=sys.stderr)
        return
    
    output_path = output_path or f"{DEFAULT_OUTPUT_DIRS['item_details']}/items_details_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Items Details Pipeline")
    print("=" * 60)
    print(f"Auctions to fetch: {len(auction_ids)}")
    if len(auction_ids) <= 10:
        print(f"Auction IDs: {', '.join(auction_ids)}")
    else:
        print(f"Auction IDs: {', '.join(auction_ids[:10])} ... (showing first 10)")
    print("=" * 60)
    
    # Fetch all items
    items = fetch_items_for_multiple_auctions(auction_ids)
    
    if not items:
        print("No items data retrieved.", file=sys.stderr)
        return
    
    # Define schema transformations
    schema_config = {
        'numeric_cols': ['viewed', 'minimum_bid', 'starting_bid', 'current_bid', 
                        'proxy_bid', 'bid_count', 'buyer_premium'],
        'boolean_cols': ['taxable', 'bidding_extended'],
        'datetime_cols': ['start_time', 'end_time']
    }
    
    # Save to parquet
    save_to_parquet(items, output_path, schema_config)
    
    # Print sample
    df = pd.read_parquet(output_path)
    print(f"\nSummary:")
    print(f"  Unique auctions: {df['auction_id'].nunique()}")
    print(f"  Total items: {len(df)}")
    if 'bid_count' in df.columns:
        print(f"  Items with bids: {df[df['bid_count'] > 0]['bid_count'].count()}")
        print(f"  Total bids: {df['bid_count'].sum():.0f}")
    
    print("\nSample items (first 5 rows):")
    display_cols = ['auction_id', 'id', 'title', 'current_bid', 'bid_count', 'viewed']
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head(5).to_string())

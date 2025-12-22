"""
Auction details pipeline
Handles batch fetching of auction details and saves to parquet
"""

import sys
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from extractors.auction_details import fetch_auction_details, extract_auction_from_json
from utils.file_io import save_to_parquet, load_from_parquet
from utils.config import DEFAULT_OUTPUT_DIRS


def fetch_multiple_auctions(auction_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch details for multiple auctions
    
    Args:
        auction_ids: List of auction IDs to fetch
        
    Returns:
        List of auction detail dictionaries
    """
    all_auctions = []
    
    for i, auction_id in enumerate(auction_ids, 1):
        print(f"[{i}/{len(auction_ids)}] Processing auction {auction_id}...")
        try:
            data = fetch_auction_details(auction_id)
            auction_row = extract_auction_from_json(data, auction_id)
            if auction_row:
                all_auctions.append(auction_row)
                print(f"  ✓ Extracted: {auction_row.get('title', 'N/A')}")
            else:
                print(f"  ✗ Failed to extract auction data", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ Error fetching auction {auction_id}: {e}", file=sys.stderr)
    
    return all_auctions


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


def run_auction_details_pipeline(
    auction_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None
):
    """
    Run the full auction details pipeline: fetch + save
    
    Args:
        auction_ids: List of auction IDs to fetch (optional if input_parquet provided)
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
    
    output_path = output_path or f"{DEFAULT_OUTPUT_DIRS['auction_details']}/auction_details_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Auction Details Pipeline")
    print("=" * 60)
    print(f"Auctions to fetch: {len(auction_ids)}")
    if len(auction_ids) <= 10:
        print(f"Auction IDs: {', '.join(auction_ids)}")
    else:
        print(f"Auction IDs: {', '.join(auction_ids[:10])} ... (showing first 10)")
    print("=" * 60)
    
    # Fetch all auctions
    auctions = fetch_multiple_auctions(auction_ids)
    
    if not auctions:
        print("No auction data retrieved.", file=sys.stderr)
        return
    
    # Define schema transformations
    schema_config = {
        'numeric_cols': ['extended_bidding_interval', 'extended_bidding_threshold', 'catalog_lots'],
        'boolean_cols': ['extended_bidding'],
        'datetime_cols': ['starts', 'ends', 'last_item_closes', 'pickup_time']
    }
    
    # Save to parquet
    save_to_parquet(auctions, output_path, schema_config)
    
    # Print sample
    df = pd.read_parquet(output_path)
    print("\nAuction data (first 5 rows):")
    print(df[['auction_id', 'title', 'starts', 'ends']].head(5).to_string())

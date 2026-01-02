import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
from datetime import datetime
import utils.json_extractors

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

API_URL = "https://maxsold.maxsold.com/msapi/auctions/items"

OUT_DIR_DEFAULT = "data/auction_details"
AUCTION_SEARCH_DEFAULT = f"data/raw_data/auction_search/auction_search_20251201.parquet"


def fetch_auction_details(auction_id: str, timeout: int = 30) -> Any:
    """Fetch auction details from MaxSold API"""
    params = {"auctionid": auction_id, "limit": 1000}
    
    print(f"Fetching auction {auction_id}...")
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_multiple_auctions(auction_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch details for multiple auctions"""
    all_auctions = []
    
    for i, auction_id in enumerate(auction_ids, 1):
        print(f"[{i}/{len(auction_ids)}] Processing auction {auction_id}...")
        try:
            data = fetch_auction_details(auction_id)
            auction_row = utils.json_extractors.extract_auction_from_json(data, auction_id)
            if auction_row:
                all_auctions.append(auction_row)
                print(f"  ✓ Extracted: {auction_row.get('title', 'N/A')}")
            else:
                print(f"  ✗ Failed to extract auction data", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ Error fetching auction {auction_id}: {e}", file=sys.stderr)
    
    return all_auctions


def load_auction_ids_from_parquet(parquet_path: str) -> List[str]:
    """Load unique auction IDs from parquet file"""
    try:
        df = pd.read_parquet(parquet_path)
        
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


def save_to_parquet(auctions: List[Dict[str, Any]], output_path: str):
    """Save auction data to parquet file"""
    if not auctions:
        print("No auction data to save.", file=sys.stderr)
        return
    
    df = pd.DataFrame(auctions)
    
    # Convert numeric columns
    numeric_cols = ['extended_bidding_interval', 'extended_bidding_threshold', 'catalog_lots']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean
    if 'extended_bidding' in df.columns:
        df['extended_bidding'] = df['extended_bidding'].astype('boolean')
    
    # Convert datetime columns
    datetime_cols = ['starts', 'ends', 'last_item_closes', 'pickup_time']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\nSaved {len(df)} auction(s) to {output_path}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f} KB")


def main(
    auction_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None
):
    """Main function to fetch and save auction data"""
    
    # Determine source of auction IDs
    if input_parquet:
        auction_ids = load_auction_ids_from_parquet(input_parquet)
    elif auction_ids is None or len(auction_ids) == 0:
        # Default: load from default parquet file
        print(f"No auction IDs provided, loading from default parquet file...")
        auction_ids = load_auction_ids_from_parquet(AUCTION_SEARCH_DEFAULT)
    
    if not auction_ids:
        print("No auction IDs to process.", file=sys.stderr)
        return
    
    output_path = output_path or f"{OUT_DIR_DEFAULT}/auction_details_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Auction Details Scraper")
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
    
    # Save to parquet
    save_to_parquet(auctions, output_path)
    
    # Print sample
    df = pd.read_parquet(output_path)
    print("\nAuction data (first 5 rows):")
    print(df[['auction_id', 'title', 'starts', 'ends']].head(5).to_string())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape MaxSold auction details API")
    parser.add_argument("auction_ids", nargs="*", help="Auction ID(s) to fetch (optional if using --input-parquet)")
    parser.add_argument("-o", "--output", help="Output parquet file path")
    parser.add_argument("-i", "--input-file", help="Read auction IDs from text file (one per line)")
    parser.add_argument("-p", "--input-parquet", help="Read auction IDs from parquet file (amAuctionId column)")
    
    args = parser.parse_args()
    
    # Get auction IDs from args, file, or parquet
    auction_ids = []
    input_parquet = None
    
    if args.input_parquet:
        input_parquet = args.input_parquet
    elif args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                auction_ids = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.auction_ids:
        auction_ids = args.auction_ids
    else:
        # No arguments provided, use default parquet file
        input_parquet = AUCTION_SEARCH_DEFAULT
    
    main(auction_ids=auction_ids, output_path=args.output, input_parquet=input_parquet)
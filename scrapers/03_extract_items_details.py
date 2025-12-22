"""
MaxSold Items Details Scraper
This script is now a thin wrapper around the refactored pipeline.
For reusable extraction logic, import from scrapers.extractors.item_details
For pipeline with file I/O, import from scrapers.pipelines.item_details_pipeline
"""

from typing import List, Optional
from pipelines.item_details_pipeline import run_item_details_pipeline


def fetch_auction_items(auction_id: str, timeout: int = 30) -> Any:
    """Fetch auction items from MaxSold API"""
    params = {"auctionid": auction_id}
    
    print(f"Fetching items for auction {auction_id}...")
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_items_from_json(data: Any, auction_id: str) -> List[Dict[str, Any]]:
    """Extract item details from JSON response"""
    # Navigate to items array
    items = None
    
    if isinstance(data, dict):
        # Try common paths
        if "auction" in data and isinstance(data["auction"], dict):
            auction_obj = data["auction"]
            if "items" in auction_obj and isinstance(auction_obj["items"], list):
                items = auction_obj["items"]
        
        # Try direct items key
        if items is None and "items" in data and isinstance(data["items"], list):
            items = data["items"]
        
        # Try other common paths
        if items is None:
            for key in ["results", "data", "lots", "auctionItems"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
    
    if not items:
        print(f"Could not locate items array in response for auction {auction_id}", file=sys.stderr)
        return []
    
    extracted_items = []
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        # Extract fields
        row = {
            "id": item.get("id", ""),
            "auction_id": auction_id,
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "taxable": item.get("taxable", None),
            "viewed": item.get("viewed", None),
            "minimum_bid": item.get("minimum_bid", None),
            "starting_bid": item.get("starting_bid", None),
            "current_bid": item.get("current_bid", None),
            "proxy_bid": item.get("proxy_bid", None),
            "start_time": item.get("start_time", ""),
            "end_time": item.get("end_time", ""),
            "lot_number": item.get("lot_number", ""),
            "bid_count": item.get("bid_count", None),
            "bidding_extended": item.get("bidding_extended", None),
            "buyer_premium": item.get("buyer_premium", None),
            "timezone": item.get("timezone", ""),
        }
        
        extracted_items.append(row)
    
    return extracted_items


def fetch_items_for_multiple_auctions(auction_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch items for multiple auctions"""
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


def save_to_parquet(items: List[Dict[str, Any]], output_path: str):
    """Save items data to parquet file"""
    if not items:
        print("No items data to save.", file=sys.stderr)
        return
    
    df = pd.DataFrame(items)
    
    # Convert numeric columns
    numeric_cols = ['viewed', 'minimum_bid', 'starting_bid', 'current_bid', 
                    'proxy_bid', 'bid_count', 'buyer_premium']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean columns
    bool_cols = ['taxable', 'bidding_extended']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('boolean')
    
    # Convert datetime columns
    datetime_cols = ['start_time', 'end_time']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\nSaved {len(df)} item(s) to {output_path}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f} KB")
    
    # Print summary stats
    print(f"\nSummary:")
    print(f"  Unique auctions: {df['auction_id'].nunique()}")
    print(f"  Total items: {len(df)}")
    print(f"  Items with bids: {df[df['bid_count'] > 0]['bid_count'].count()}")
    print(f"  Total bids: {df['bid_count'].sum():.0f}")


def main(
    auction_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None
):
    """Main function to fetch and save items data"""
    run_item_details_pipeline(
        auction_ids=auction_ids,
        output_path=output_path,
        input_parquet=input_parquet
    )


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Scrape MaxSold items details API")
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
    
    main(auction_ids=auction_ids, output_path=args.output, input_parquet=input_parquet)
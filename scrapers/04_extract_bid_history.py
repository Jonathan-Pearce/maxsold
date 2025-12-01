import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

API_URL = "https://maxsold.maxsold.com/msapi/auctions/items"

OUT_DIR_DEFAULT = "/data/bid_history"
ITEMS_PARQUET_DEFAULT = "data/item_details/items_details_20251201.parquet"

# Thread-safe lock for printing
print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)


def fetch_item_bid_history(auction_id: str, item_id: str, timeout: int = 30) -> Any:
    """Fetch item bid history from MaxSold API"""
    params = {
        "auctionid": auction_id,
        "itemid": item_id
    }
    
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_bids_from_json(data: Any, auction_id: str, item_id: str) -> List[Dict[str, Any]]:
    """Extract bid history from JSON response"""
    # Navigate to bid history array
    bids = None
    
    # Try to find the item and its bid history
    if isinstance(data, dict):
        # Try common paths for auction/items structure
        auction_obj = None
        if "auction" in data and isinstance(data["auction"], dict):
            auction_obj = data["auction"]
        
        # Find items array
        items_list = None
        if auction_obj and "items" in auction_obj and isinstance(auction_obj["items"], list):
            items_list = auction_obj["items"]
        elif "items" in data and isinstance(data["items"], list):
            items_list = data["items"]
        
        # Find the specific item
        if items_list:
            for item in items_list:
                if not isinstance(item, dict):
                    continue
                item_id_str = str(item.get("id", ""))
                if item_id_str == str(item_id):
                    # Found the item, look for bid history
                    for key in ["bid_history", "bidHistory", "bids", "bid_history_list"]:
                        if key in item and isinstance(item[key], list):
                            bids = item[key]
                            break
                    # Sometimes bid_history is nested: [[{...}, ...]]
                    if bids and isinstance(bids[0], list):
                        bids = bids[0]
                    break
    
    if not bids:
        return []
    
    extracted_bids = []
    
    for i, bid in enumerate(bids, start=1):
        if not isinstance(bid, dict):
            continue
        
        # Extract fields with various possible key names
        time_of_bid = (bid.get("time_of_bid") or bid.get("timeOfBid") or 
                       bid.get("time") or bid.get("bidTime") or 
                       bid.get("createdAt") or "")
        
        amount = (bid.get("amount") or bid.get("bidAmount") or 
                  bid.get("value") or bid.get("currentBid") or None)
        
        isproxy = (bid.get("isproxy") or bid.get("isProxy") or 
                   bid.get("proxy") or bid.get("is_proxy") or 
                   bid.get("isProxyBid") or False)
        
        row = {
            "auction_id": auction_id,
            "item_id": item_id,
            "bid_number": i,
            "time_of_bid": str(time_of_bid),
            "amount": amount,
            "isproxy": bool(isproxy),
        }
        
        extracted_bids.append(row)
    
    return extracted_bids


def fetch_single_item_bids(item: Dict[str, str], index: int, total: int) -> tuple[int, List[Dict[str, Any]], Optional[str]]:
    """Fetch bids for a single item (designed for parallel execution)"""
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
    """Fetch bid history for multiple auction/item pairs using parallel requests"""
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
    """Fetch bid history for multiple auction/item pairs (sequential - legacy)"""
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
    """Load auction_id and item_id pairs from parquet file"""
    try:
        import glob
        
        # Handle glob patterns
        files = glob.glob(parquet_path)
        if not files:
            print(f"No files found matching: {parquet_path}", file=sys.stderr)
            return []
        
        # Read the most recent file if multiple matches
        parquet_file = sorted(files)[-1]
        print(f"Reading items from: {parquet_file}")
        
        df = pd.read_parquet(parquet_file)
        
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


def save_to_parquet(bids: List[Dict[str, Any]], output_path: str):
    """Save bid history data to parquet file"""
    if not bids:
        print("No bid history data to save.", file=sys.stderr)
        return
    
    df = pd.DataFrame(bids)
    
    # Convert numeric columns
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    if 'bid_number' in df.columns:
        df['bid_number'] = pd.to_numeric(df['bid_number'], errors='coerce').astype('Int64')
    
    # Convert boolean
    if 'isproxy' in df.columns:
        df['isproxy'] = df['isproxy'].astype('boolean')
    
    # Convert datetime
    if 'time_of_bid' in df.columns:
        df['time_of_bid'] = pd.to_datetime(df['time_of_bid'], errors='coerce')
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\nSaved {len(df)} bid(s) to {output_path}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f} KB")
    
    # Print summary stats
    print(f"\nSummary:")
    print(f"  Unique auctions: {df['auction_id'].nunique()}")
    print(f"  Unique items: {df['item_id'].nunique()}")
    print(f"  Total bids: {len(df)}")
    if 'isproxy' in df.columns:
        print(f"  Proxy bids: {df['isproxy'].sum()}")
    if 'amount' in df.columns:
        print(f"  Total bid amount: ${df['amount'].sum():,.2f}")


def main(
    items: Optional[List[Dict[str, str]]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None,
    max_workers: int = 10,
    limit: Optional[int] = None,
    sequential: bool = False
):
    """Main function to fetch and save bid history data"""
    
    # Determine source of items
    if input_parquet:
        items = load_items_from_parquet(input_parquet, limit=limit)
    elif items is None or len(items) == 0:
        # Default: load from default parquet file
        print(f"No items provided, loading from default items parquet file...")
        items = load_items_from_parquet(ITEMS_PARQUET_DEFAULT, limit=limit)
    
    if not items:
        print("No items to process.", file=sys.stderr)
        return
    
    output_path = output_path or f"{OUT_DIR_DEFAULT}/bid_history_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Bid History Scraper")
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
    
    # Save to parquet
    save_to_parquet(bids, output_path)
    
    # Print sample
    df = pd.read_parquet(output_path)
    print("\nSample bids (first 10 rows):")
    display_cols = ['auction_id', 'item_id', 'bid_number', 'time_of_bid', 'amount', 'isproxy']
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head(10).to_string())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape MaxSold bid history API")
    parser.add_argument("--auction-id", help="Single auction ID")
    parser.add_argument("--item-id", help="Single item ID (requires --auction-id)")
    parser.add_argument("-o", "--output", help="Output parquet file path")
    parser.add_argument("-i", "--input-file", help="Read auction_id,item_id pairs from CSV file")
    parser.add_argument("-p", "--input-parquet", help="Read items from parquet file (auction_id and id columns)")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("-l", "--limit", type=int, help="Limit number of items to process")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing instead of parallel")
    
    args = parser.parse_args()
    
    # Get items from args, file, or parquet
    items = []
    input_parquet = None
    
    if args.input_parquet:
        input_parquet = args.input_parquet
    elif args.input_file:
        try:
            df = pd.read_csv(args.input_file)
            if 'auction_id' in df.columns and 'item_id' in df.columns:
                items = df[['auction_id', 'item_id']].to_dict('records')
                items = [{"auction_id": str(i["auction_id"]), "item_id": str(i["item_id"])} for i in items]
            else:
                print(f"CSV must have 'auction_id' and 'item_id' columns", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.auction_id and args.item_id:
        items = [{"auction_id": args.auction_id, "item_id": args.item_id}]
    elif args.auction_id:
        print("Error: --item-id is required when using --auction-id", file=sys.stderr)
        sys.exit(1)
    else:
        # No arguments provided, use default parquet file
        input_parquet = ITEMS_PARQUET_DEFAULT
    
    main(
        items=items, 
        output_path=args.output, 
        input_parquet=input_parquet,
        max_workers=args.workers,
        limit=args.limit,
        sequential=args.sequential
    )
"""
MaxSold Bid History Scraper
This script is now a thin wrapper around the refactored pipeline.
For reusable extraction logic, import from scrapers.extractors.bid_history
For pipeline with file I/O, import from scrapers.pipelines.bid_history_pipeline
"""

import sys
import pandas as pd
from typing import List, Dict, Optional
from pipelines.bid_history_pipeline import run_bid_history_pipeline


def main(
    items: Optional[List[Dict[str, str]]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None,
    max_workers: int = 10,
    limit: Optional[int] = None,
    sequential: bool = False
):
    """Main function to fetch and save bid history data"""
    run_bid_history_pipeline(
        items=items,
        output_path=output_path,
        input_parquet=input_parquet,
        max_workers=max_workers,
        limit=limit,
        sequential=sequential
    )


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
    
    main(
        items=items, 
        output_path=args.output, 
        input_parquet=input_parquet,
        max_workers=args.workers,
        limit=args.limit,
        sequential=args.sequential
    )

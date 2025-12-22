"""
MaxSold Enriched Item Details Scraper
This script is now a thin wrapper around the refactored pipeline.
For reusable extraction logic, import from scrapers.extractors.item_enriched
For pipeline with file I/O, import from scrapers.pipelines.item_enriched_pipeline
"""

import sys
from typing import List, Optional
from pipelines.item_enriched_pipeline import run_item_enriched_pipeline


def main(
    item_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None,
    max_workers: int = 10,
    limit: Optional[int] = None,
    sequential: bool = False
):
    """Main function to fetch and save enriched item details data"""
    run_item_enriched_pipeline(
        item_ids=item_ids,
        output_path=output_path,
        input_parquet=input_parquet,
        max_workers=max_workers,
        limit=limit,
        sequential=sequential
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape MaxSold enriched item details API")
    parser.add_argument("item_ids", nargs="*", help="Item ID(s) to fetch (optional if using --input-parquet)")
    parser.add_argument("-o", "--output", help="Output parquet file path")
    parser.add_argument("-i", "--input-file", help="Read item IDs from text file (one per line)")
    parser.add_argument("-p", "--input-parquet", help="Read items from parquet file (id or amLotId column)")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("-l", "--limit", type=int, help="Limit number of items to process")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing instead of parallel")
    
    args = parser.parse_args()
    
    # Get item IDs from args, file, or parquet
    item_ids = []
    input_parquet = None
    
    if args.input_parquet:
        input_parquet = args.input_parquet
    elif args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                item_ids = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.item_ids:
        item_ids = args.item_ids
    
    main(
        item_ids=item_ids, 
        output_path=args.output, 
        input_parquet=input_parquet,
        max_workers=args.workers,
        limit=args.limit,
        sequential=args.sequential
    )

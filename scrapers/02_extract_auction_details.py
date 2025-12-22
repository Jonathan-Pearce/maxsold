"""
MaxSold Auction Details Scraper
This script is now a thin wrapper around the refactored pipeline.
For reusable extraction logic, import from scrapers.extractors.auction_details
For pipeline with file I/O, import from scrapers.pipelines.auction_details_pipeline
"""

from typing import List, Optional
from pipelines.auction_details_pipeline import run_auction_details_pipeline


def main(
    auction_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    input_parquet: Optional[str] = None
):
    """Main function to fetch and save auction data"""
    run_auction_details_pipeline(
        auction_ids=auction_ids,
        output_path=output_path,
        input_parquet=input_parquet
    )


if __name__ == "__main__":
    import argparse
    import sys
    
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
        # No arguments provided, use default
        pass
    
    main(auction_ids=auction_ids, output_path=args.output, input_parquet=input_parquet)
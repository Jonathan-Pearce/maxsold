"""
MaxSold Auction Search Scraper
This script is now a thin wrapper around the refactored pipeline.
For reusable extraction logic, import from scrapers.extractors.auction_search
For pipeline with file I/O, import from scrapers.pipelines.auction_search_pipeline
"""

from typing import Optional
from pipelines.auction_search_pipeline import run_auction_search_pipeline


def main(
    output_path: Optional[str] = None,
    lat: float = 43.653226,
    lng: float = -79.3831843,
    radius_metres: int = 201168,
    country: str = "canada",
    days: int = 120,
    max_pages: Optional[int] = None
):
    """Main function to fetch and save sales data"""
    run_auction_search_pipeline(
        output_path=output_path,
        lat=lat,
        lng=lng,
        radius_metres=radius_metres,
        country=country,
        days=days,
        max_pages=max_pages
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape MaxSold sales search API")
    parser.add_argument("-o", "--output", help="Output parquet file path")
    parser.add_argument("--lat", type=float, default=43.653226, help="Latitude")
    parser.add_argument("--lng", type=float, default=-79.3831843, help="Longitude")
    parser.add_argument("--radius", type=int, default=201168, help="Radius in metres")
    parser.add_argument("--country", default="canada", help="Country filter")
    parser.add_argument("--days", type=int, default=180, help="Days back to search")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to fetch")
    
    args = parser.parse_args()
    
    main(
        output_path=args.output,
        lat=args.lat,
        lng=args.lng,
        radius_metres=args.radius,
        country=args.country,
        days=args.days,
        max_pages=args.max_pages
    )
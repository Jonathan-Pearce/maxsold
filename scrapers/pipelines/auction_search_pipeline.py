"""
Auction search pipeline
Handles batch fetching of auction search results and saves to parquet
"""

import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

from extractors.auction_search import fetch_sales_search, extract_sales_from_json
from utils.file_io import save_to_parquet
from utils.config import DEFAULT_OUTPUT_DIRS


def fetch_all_pages(
    lat: float = 43.653226,
    lng: float = -79.3831843,
    radius_metres: int = 201168,
    country: str = "canada",
    limit: int = 100,
    sale_state: str = "closed",
    days: int = 120,
    max_pages: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Fetch all pages of sales data
    
    Args:
        lat: Latitude for search center
        lng: Longitude for search center
        radius_metres: Search radius in metres
        country: Country filter
        limit: Number of results per page
        sale_state: State of sales to fetch
        days: Number of days back to search
        max_pages: Optional maximum number of pages to fetch
        
    Returns:
        List of all sales data
    """
    all_sales = []
    page_number = 0
    
    while True:
        try:
            data = fetch_sales_search(
                lat=lat,
                lng=lng,
                radius_metres=radius_metres,
                country=country,
                page_number=page_number,
                limit=limit,
                sale_state=sale_state,
                days=days,
                total=True
            )
        except Exception as e:
            print(f"Error fetching page {page_number}: {e}", file=sys.stderr)
            break
        
        sales = extract_sales_from_json(data)
        if not sales:
            print(f"No sales found on page {page_number}, stopping.")
            break
        
        all_sales.extend(sales)
        print(f"  Page {page_number}: extracted {len(sales)} sales (total: {len(all_sales)})")
        
        # Check if we have more pages
        total_available = data.get("total") if isinstance(data, dict) else None
        if total_available is not None:
            print(f"  API reports {total_available} total sales available")
            if len(all_sales) >= total_available:
                print("All sales fetched.")
                break
        
        # Check if this page had fewer results than limit (last page)
        if len(sales) < limit:
            print("Last page reached (partial results).")
            break
        
        # Max pages safety check
        if max_pages is not None and page_number >= max_pages - 1:
            print(f"Max pages ({max_pages}) reached.")
            break
        
        page_number += 1
    
    return all_sales


def run_auction_search_pipeline(
    output_path: Optional[str] = None,
    lat: float = 43.653226,
    lng: float = -79.3831843,
    radius_metres: int = 201168,
    country: str = "canada",
    days: int = 120,
    max_pages: Optional[int] = None
):
    """
    Run the full auction search pipeline: fetch + save
    
    Args:
        output_path: Path to save output parquet (default: auto-generated with timestamp)
        lat: Latitude for search center
        lng: Longitude for search center
        radius_metres: Search radius in metres
        country: Country filter
        days: Number of days back to search
        max_pages: Optional maximum number of pages to fetch
    """
    output_path = output_path or f"{DEFAULT_OUTPUT_DIRS['auction_search']}/auction_search_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Auction Search Pipeline")
    print("=" * 60)
    print(f"Location: ({lat}, {lng})")
    print(f"Radius: {radius_metres}m ({radius_metres/1000:.1f}km)")
    print(f"Country: {country}")
    print(f"Days: {days}")
    print(f"Sale state: closed")
    print("=" * 60)
    
    # Fetch all sales
    sales = fetch_all_pages(
        lat=lat,
        lng=lng,
        radius_metres=radius_metres,
        country=country,
        limit=100,
        sale_state="closed",
        days=days,
        max_pages=max_pages
    )
    
    if not sales:
        print("No sales data retrieved.", file=sys.stderr)
        return
    
    # Define schema transformations
    schema_config = {
        'numeric_cols': ['distanceMeters', 'totalBids', 'numberLots', 'lat', 'lng'],
        'boolean_cols': ['hasShipping'],
        'datetime_cols': ['openTime', 'closeTime']
    }
    
    # Save to parquet
    save_to_parquet(sales, output_path, schema_config)
    
    # Print sample
    import pandas as pd
    df = pd.read_parquet(output_path)
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())

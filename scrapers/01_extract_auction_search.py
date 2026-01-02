import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
import utils.json_extractors

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

API_URL = "https://api.maxsold.com/sales/search"

OUT_DIR_DEFAULT = "data/auction_search"


def fetch_sales_search(
    lat: float = 43.653226,
    lng: float = -79.3831843,
    radius_metres: int = 201168,
    country: str = "canada",
    page_number: int = 0,
    limit: int = 100,
    sale_state: str = "closed",
    days: int = 120,
    total: bool = True,
    timeout: int = 30
) -> Any:
    """Fetch sales search data from MaxSold API"""
    params = {
        "lat": lat,
        "lng": lng,
        "radiusMetres": radius_metres,
        "country": country,
        "pageNumber": page_number,
        "limit": limit,
        "saleState": sale_state,
        "days": days,
        "total": str(total).lower()
    }
    
    print(f"Fetching sales data: page={page_number}, limit={limit}...")
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


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
    """Fetch all pages of sales data"""
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
        
        sales = utils.json_extractors.extract_sales_from_json(data)
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


def save_to_parquet(sales: List[Dict[str, Any]], output_path: str):
    """Save sales data to parquet file"""
    if not sales:
        print("No sales to save.", file=sys.stderr)
        return
    
    df = pd.DataFrame(sales)
    
    # Convert numeric columns
    numeric_cols = ['distanceMeters', 'totalBids', 'numberLots', 'lat', 'lng']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean
    if 'hasShipping' in df.columns:
        df['hasShipping'] = df['hasShipping'].astype('boolean')
    
    # Convert datetime
    for col in ['openTime', 'closeTime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\nSaved {len(df)} sales to {output_path}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f} KB")


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
    from datetime import datetime
    
    output_path = output_path or f"{OUT_DIR_DEFAULT}/auction_search_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    print("MaxSold Auction Search Scraper")
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
    
    # Save to parquet
    save_to_parquet(sales, output_path)
    
    # Print sample
    df = pd.read_parquet(output_path)
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())


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
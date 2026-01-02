"""
Scraper to extract auction location data from MaxSold API and filter by distance from Toronto
"""
import requests
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Downtown Toronto coordinates
TORONTO_LAT = 43.653226
TORONTO_LNG = -79.383184
MAX_DISTANCE_METERS = 201168

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

API_BASE_URL = "https://api.maxsold.com/sales/am/"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth in meters
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    
    return c * r


def fetch_auction_data(auction_id: int, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch auction data from MaxSold API
    
    Args:
        auction_id: The auction ID to fetch
        timeout: Request timeout in seconds
    
    Returns:
        JSON response or None if request fails
    """
    url = f"{API_BASE_URL}{auction_id}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching auction {auction_id}: {e}")
        return None


def extract_auction_fields(data: Dict) -> Optional[Dict[str, any]]:
    """
    Extract required fields from auction data
    
    Args:
        data: JSON response from API
    
    Returns:
        Dictionary with extracted fields or None if required fields missing
    """
    try:
        # Extract amAuctionId
        am_auction_id = data.get('amAuctionId')
        
        # Extract lat, lng, and postalCode from nested approxLocation
        approx_location = data.get('approxLocation', {})
        lat_lng = approx_location.get('latLng', {})
        lat = lat_lng.get('lat')
        lng = lat_lng.get('lng')
        postal_code = approx_location.get('postalCode')
        
        # Check if required fields exist
        if am_auction_id is None or lat is None or lng is None:
            return None
        
        return {
            'amAuctionId': am_auction_id,
            'lat': lat,
            'lng': lng,
            'postalCode': postal_code
        }
    except (KeyError, TypeError, AttributeError) as e:
        print(f"Error extracting fields: {e}")
        return None


def process_single_auction(auction_id: int) -> Optional[Dict]:
    """
    Process a single auction: fetch, extract, calculate distance, and filter
    
    Args:
        auction_id: The auction ID to process
    
    Returns:
        Dictionary with auction data if within radius, None otherwise
    """
    # Fetch data
    data = fetch_auction_data(auction_id)
    
    if data is None:
        return None
    
    # Extract fields
    extracted = extract_auction_fields(data)
    
    if extracted is None:
        return None
    
    # Calculate distance
    distance = haversine_distance(
        extracted['lat'],
        extracted['lng'],
        TORONTO_LAT,
        TORONTO_LNG
    )
    
    #Filter by distance
    if distance <= MAX_DISTANCE_METERS:
        extracted['distance_from_toronto_meters'] = distance
        return extracted
    
    return None


def scrape_auctions(
    start_id: int = 1,
    end_id: int = 100000,
    output_path: str = "data/backfilling/auction_location_data.parquet",
    checkpoint_interval: int = 1000,
    delay_seconds: float = 0.01,
    max_workers: int = 10
) -> pd.DataFrame:
    """
    Scrape auction data from MaxSold API and filter by distance using parallel processing
    
    Args:
        start_id: Starting auction ID
        end_id: Ending auction ID (inclusive)
        output_path: Path to save the output parquet file
        checkpoint_interval: Save checkpoint every N records
        delay_seconds: Delay between requests to avoid rate limiting (per worker)
        max_workers: Number of parallel workers for concurrent requests
    
    Returns:
        DataFrame with filtered auction data
    """
    results = []
    results_lock = Lock()
    
    print(f"Starting parallel scraping from ID {start_id} to {end_id}")
    print(f"Using {max_workers} parallel workers")
    print(f"Filtering auctions within {MAX_DISTANCE_METERS}m of Toronto ({TORONTO_LAT}, {TORONTO_LNG})")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create list of auction IDs
    auction_ids = list(range(start_id, end_id + 1))
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_id = {executor.submit(process_single_auction, aid): aid for aid in auction_ids}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(auction_ids), desc="Scraping auctions") as pbar:
            for future in as_completed(future_to_id):
                auction_id = future_to_id[future]
                
                try:
                    result = future.result()
                    
                    if result is not None:
                        with results_lock:
                            results.append(result)
                            
                            if len(results) % 100 == 0:
                                tqdm.write(f"Found {len(results)} auctions within radius so far...")
                            
                            # Save checkpoint
                            if len(results) % checkpoint_interval == 0:
                                df_checkpoint = pd.DataFrame(results)
                                checkpoint_path = output_path.replace('.parquet', f'_checkpoint_{len(results)}.parquet')
                                df_checkpoint.to_parquet(checkpoint_path, index=False)
                                tqdm.write(f"Checkpoint saved: {checkpoint_path}")
                
                except Exception as e:
                    tqdm.write(f"Error processing auction {auction_id}: {e}")
                
                finally:
                    pbar.update(1)
                    # Rate limiting per request
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
    
    # Convert to DataFrame and save
    if results:
        df = pd.DataFrame(results)
        # Sort by amAuctionId for consistent ordering
        df = df.sort_values('amAuctionId').reset_index(drop=True)
        df.to_parquet(output_path, index=False)
        print(f"\n✓ Scraping complete! Saved {len(df)} records to {output_path}")
        print(f"\nSummary statistics:")
        print(f"  Total auctions found: {len(df)}")
        print(f"  Distance range: {df['distance_from_toronto_meters'].min():.2f}m - {df['distance_from_toronto_meters'].max():.2f}m")
        print(f"  Postal codes with data: {df['postalCode'].notna().sum()}")
        return df
    else:
        print("\nNo auctions found within the specified radius")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape MaxSold auction location data with parallel processing')
    parser.add_argument('--start-id', type=int, default=1, help='Starting auction ID')
    parser.add_argument('--end-id', type=int, default=100000, help='Ending auction ID')
    parser.add_argument('--output', type=str, default='data/backfilling/auction_location_data.parquet',
                        help='Output parquet file path')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between requests in seconds (per worker)')
    parser.add_argument('--checkpoint', type=int, default=1000, help='Checkpoint interval')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    scrape_auctions(
        start_id=args.start_id,
        end_id=args.end_id,
        output_path=args.output,
        checkpoint_interval=args.checkpoint,
        delay_seconds=args.delay,
        max_workers=args.workers
    )

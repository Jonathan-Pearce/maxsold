"""
Test and debug script for the auction scraper
"""
import json
import sys
from pathlib import Path

# Add parent directory to path to import the scraper module
sys.path.insert(0, str(Path(__file__).parent))

from extract_auction_location_data import (
    fetch_auction_data,
    extract_auction_fields,
    process_single_auction,
    haversine_distance,
    TORONTO_LAT,
    TORONTO_LNG,
    MAX_DISTANCE_METERS
)


def test_fetch_auction(auction_id: int):
    """Test fetching auction data from API"""
    print(f"\n{'='*60}")
    print(f"TEST 1: Fetching auction {auction_id}")
    print('='*60)
    
    data = fetch_auction_data(auction_id)
    
    if data is None:
        print(f"❌ Failed to fetch auction {auction_id}")
        return None
    
    print(f"✓ Successfully fetched auction {auction_id}")
    print(f"\nRaw JSON response:")
    print(json.dumps(data, indent=2))
    return data


def test_extract_fields(data: dict, auction_id: int):
    """Test extracting fields from auction data"""
    print(f"\n{'='*60}")
    print(f"TEST 2: Extracting fields from auction {auction_id}")
    print('='*60)
    
    if data is None:
        print("❌ No data to extract from")
        return None
    
    extracted = extract_auction_fields(data)
    
    if extracted is None:
        print(f"❌ Failed to extract fields from auction {auction_id}")
        print("\nAvailable keys in data:")
        print(list(data.keys()))
        return None
    
    print(f"✓ Successfully extracted fields")
    print(f"\nExtracted data:")
    for key, value in extracted.items():
        print(f"  {key}: {value}")
    
    return extracted


def test_distance_calculation(extracted: dict):
    """Test distance calculation"""
    print(f"\n{'='*60}")
    print(f"TEST 3: Calculating distance")
    print('='*60)
    
    if extracted is None:
        print("❌ No extracted data to calculate distance from")
        return None
    
    try:
        lat = extracted['lat']
        lng = extracted['lng']
        
        distance = haversine_distance(lat, lng, TORONTO_LAT, TORONTO_LNG)
        
        print(f"✓ Distance calculation successful")
        print(f"\nLocation details:")
        print(f"  Auction coordinates: ({lat}, {lng})")
        print(f"  Toronto coordinates: ({TORONTO_LAT}, {TORONTO_LNG})")
        print(f"  Distance: {distance:,.2f} meters ({distance/1000:.2f} km)")
        print(f"  Max allowed distance: {MAX_DISTANCE_METERS:,} meters ({MAX_DISTANCE_METERS/1000:.2f} km)")
        print(f"  Within radius: {'✓ YES' if distance <= MAX_DISTANCE_METERS else '✗ NO'}")
        
        return distance
    except Exception as e:
        print(f"❌ Error calculating distance: {e}")
        return None


def test_process_single_auction(auction_id: int):
    """Test the complete process_single_auction function"""
    print(f"\n{'='*60}")
    print(f"TEST 4: Processing complete auction {auction_id}")
    print('='*60)
    
    result = process_single_auction(auction_id)
    
    if result is None:
        print(f"❌ process_single_auction returned None for auction {auction_id}")
        return None
    
    print(f"✓ Successfully processed auction {auction_id}")
    print(f"\nFinal result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result


def run_all_tests(auction_id: int):
    """Run all tests for a specific auction ID"""
    print(f"\n{'#'*60}")
    print(f"DEBUGGING AUCTION ID: {auction_id}")
    print('#'*60)
    
    # Test 1: Fetch
    data = test_fetch_auction(auction_id)
    
    # Test 2: Extract
    extracted = test_extract_fields(data, auction_id)
    
    # Test 3: Distance
    distance = test_distance_calculation(extracted)
    
    # Test 4: Complete process
    result = test_process_single_auction(auction_id)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Auction ID: {auction_id}")
    print(f"Fetch successful: {'✓' if data is not None else '✗'}")
    print(f"Extract successful: {'✓' if extracted is not None else '✗'}")
    print(f"Distance calculated: {'✓' if distance is not None else '✗'}")
    print(f"Process successful: {'✓' if result is not None else '✗'}")
    
    if result:
        within_radius = result['distance_from_toronto_meters'] <= MAX_DISTANCE_METERS
        print(f"Within radius: {'✓ YES' if within_radius else '✗ NO'}")
        print(f"\nExpected behavior: {'SAVED' if within_radius else 'DISCARDED'}")
    
    print('='*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and debug auction scraper')
    parser.add_argument('--auction-id', type=int, default=1528, 
                        help='Auction ID to test (default: 1528)')
    parser.add_argument('--test-range', nargs=2, type=int, metavar=('START', 'END'),
                        help='Test a range of auction IDs')
    
    args = parser.parse_args()
    
    if args.test_range:
        start_id, end_id = args.test_range
        print(f"\nTesting auction IDs from {start_id} to {end_id}\n")
        
        for aid in range(start_id, end_id + 1):
            run_all_tests(aid)
            if aid < end_id:
                print("\n" + "~"*60 + "\n")
    else:
        run_all_tests(args.auction_id)

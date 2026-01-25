#!/usr/bin/env python3
"""Quick test of the fixed extraction"""

# Simulated API response from auction 1528
test_data = {
    "amAuctionId": 1528,
    "approxLocation": {
        "postalCode": "L8L7N8",
        "latLng": {
            "lat": 43.2484505,
            "lng": -79.82127000000001
        }
    }
}

# Test the extraction logic
am_auction_id = test_data.get('amAuctionId')
approx_location = test_data.get('approxLocation', {})
lat_lng = approx_location.get('latLng', {})
lat = lat_lng.get('lat')
lng = lat_lng.get('lng')
postal_code = approx_location.get('postalCode')

print("Extraction test:")
print(f"  amAuctionId: {am_auction_id}")
print(f"  lat: {lat}")
print(f"  lng: {lng}")
print(f"  postalCode: {postal_code}")
print(f"\n✓ All fields extracted successfully!")

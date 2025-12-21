#!/usr/bin/env python3
"""
Extract item enriched details from MaxSold API and transform for ML model
Extracts data from:
  1. https://api.maxsold.com/listings/am/{item_id}/enriched
  2. https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}&itemid={item_id}
  3. https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id} (auction-level data)
Transforms the data into ML-ready JSON format with auction_id as a feature
"""

import requests
import json
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

ENRICHED_API_URL = "https://api.maxsold.com/listings/am/{}/enriched"
ITEMS_API_URL = "https://maxsold.maxsold.com/msapi/auctions/items"


def fetch_enriched_details(item_id: str, timeout: int = 30) -> Any:
    """Fetch enriched details for a single item from MaxSold API"""
    url = ENRICHED_API_URL.format(item_id)
    
    print(f"Fetching enriched data from: {url}")
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_item_details(auction_id: str, item_id: str, timeout: int = 30) -> Any:
    """Fetch item-level details from MaxSold items API"""
    params = {
        "auctionid": auction_id,
        "itemid": item_id
    }
    
    print(f"Fetching item details from: {ITEMS_API_URL}")
    print(f"  Parameters: auctionid={auction_id}, itemid={item_id}")
    r = requests.get(ITEMS_API_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_auction_details(auction_id: str, timeout: int = 30) -> Any:
    """Fetch auction-level details from MaxSold items API"""
    params = {
        "auctionid": auction_id
    }
    
    print(f"Fetching auction details from: {ITEMS_API_URL}")
    print(f"  Parameters: auctionid={auction_id}")
    r = requests.get(ITEMS_API_URL, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_auction_data(data: Any) -> Optional[Dict[str, Any]]:
    """Extract auction-level data from auction API response"""
    
    # Navigate to auction object
    auction = None
    
    if isinstance(data, dict):
        # Try common paths
        if "auction" in data and isinstance(data["auction"], dict):
            auction = data["auction"]
        elif "auctions" in data:
            auctions = data["auctions"]
            if isinstance(auctions, list) and auctions:
                auction = auctions[0]
            elif isinstance(auctions, dict):
                auction = auctions
        elif "data" in data and isinstance(data["data"], dict):
            auction = data["data"]
        # Fallback: check if data itself has the expected fields
        elif any(k in data for k in ["id", "title", "starts", "ends"]):
            auction = data
    
    if not auction or not isinstance(auction, dict):
        return None
    
    # Extract all relevant auction fields
    auction_data = {
        'auction_api_id': auction.get('id'),
        'auction_title': auction.get('title'),
        'auction_starts': auction.get('starts'),
        'auction_ends': auction.get('ends'),
        'last_item_closes': auction.get('last_item_closes'),
        'removal_info': auction.get('removal_info'),
        'inspection_info': auction.get('inspection_info'),
        'auction_intro': auction.get('intro'),
        'pickup_time': auction.get('pickup_time'),
        'partner_url': auction.get('partner_url'),
        'extended_bidding': auction.get('extended_bidding'),
        'extended_bidding_interval': auction.get('extended_bidding_interval'),
        'extended_bidding_threshold': auction.get('extended_bidding_threshold'),
        'catalog_lots': auction.get('catalog_lots'),
    }
    
    return auction_data


def extract_item_data(data: Any) -> Optional[Dict[str, Any]]:
    """Extract item-level data from items API response"""
    
    # Navigate to item data
    item = None
    
    if isinstance(data, dict):
        # Try common paths
        if "auction" in data and isinstance(data["auction"], dict):
            auction_obj = data["auction"]
            if "items" in auction_obj and isinstance(auction_obj["items"], list):
                items = auction_obj["items"]
                if items:
                    item = items[0]  # Should only be one item
        
        # Try direct items key
        if item is None and "items" in data and isinstance(data["items"], list):
            items = data["items"]
            if items:
                item = items[0]
        
        # Try single item key
        if item is None and "item" in data:
            item = data["item"]
    
    if not item or not isinstance(item, dict):
        return None
    
    # Extract all relevant fields
    item_data = {
        'item_api_id': item.get('id'),
        'item_title': item.get('title'),
        'item_description': item.get('description'),
        'taxable': item.get('taxable'),
        'viewed': item.get('viewed'),
        'minimum_bid': item.get('minimum_bid'),
        'starting_bid': item.get('starting_bid'),
        'current_bid': item.get('current_bid'),
        'proxy_bid': item.get('proxy_bid'),
        'start_time': item.get('start_time'),
        'end_time': item.get('end_time'),
        'lot_number': item.get('lot_number'),
        'bid_count': item.get('bid_count'),
        'bidding_extended': item.get('bidding_extended'),
        'buyer_premium': item.get('buyer_premium'),
        'timezone': item.get('timezone'),
    }
    
    return item_data


def transform_for_ml(
    enriched_data: Any, 
    item_data: Optional[Dict[str, Any]], 
    auction_data: Optional[Dict[str, Any]],
    item_id: str, 
    auction_id: str
) -> Optional[Dict[str, Any]]:
    """
    Transform enriched item data, item-level data, and auction-level data into ML-ready format
    Extracts key features including auction_id for machine learning
    """
    
    if not isinstance(enriched_data, dict):
        print("Warning: Enriched API returned non-dict data")
        return None
    
    # Initialize ML features dictionary
    ml_features = {
        # Identifiers
        'item_id': item_id,
        'lot_id': enriched_data.get('amLotId'),
        'auction_id': auction_id,  # Auction ID as requested feature
        
        # Metadata
        'extracted_at': datetime.now().isoformat(),
    }
    
    # Add auction-level data
    if auction_data:
        ml_features.update({
            'auction_api_id': auction_data.get('auction_api_id'),
            'auction_title': auction_data.get('auction_title'),
            'auction_starts': auction_data.get('auction_starts'),
            'auction_ends': auction_data.get('auction_ends'),
            'last_item_closes': auction_data.get('last_item_closes'),
            'auction_intro': auction_data.get('auction_intro'),
            'auction_intro_length': len(auction_data.get('auction_intro', '')),
            'pickup_time': auction_data.get('pickup_time'),
            'extended_bidding': 1 if auction_data.get('extended_bidding') else 0,
            'extended_bidding_interval': auction_data.get('extended_bidding_interval'),
            'extended_bidding_threshold': auction_data.get('extended_bidding_threshold'),
            'catalog_lots': auction_data.get('catalog_lots'),
            'has_partner_url': 1 if auction_data.get('partner_url') else 0,
            'has_removal_info': 1 if auction_data.get('removal_info') else 0,
            'has_inspection_info': 1 if auction_data.get('inspection_info') else 0,
        })
    
    # Add item-level data from items API
    if item_data:
        ml_features.update({
            'item_api_id': item_data.get('item_api_id'),
            'lot_number': item_data.get('lot_number'),
            'taxable': 1 if item_data.get('taxable') else 0,
            'viewed_count': item_data.get('viewed', 0),
            'minimum_bid': item_data.get('minimum_bid'),
            'starting_bid': item_data.get('starting_bid'),
            'current_bid': item_data.get('current_bid'),
            'proxy_bid': item_data.get('proxy_bid'),
            'start_time': item_data.get('start_time'),
            'end_time': item_data.get('end_time'),
            'bid_count': item_data.get('bid_count', 0),
            'bidding_extended': 1 if item_data.get('bidding_extended') else 0,
            'buyer_premium': item_data.get('buyer_premium'),
            'timezone': item_data.get('timezone'),
            'has_bids': 1 if item_data.get('bid_count', 0) > 0 else 0,
            'has_current_bid': 1 if item_data.get('current_bid') else 0,
        })
        
        # Calculate derived bidding features
        if item_data.get('current_bid') and item_data.get('starting_bid'):
            ml_features['bid_increase_from_start'] = item_data['current_bid'] - item_data['starting_bid']
            ml_features['bid_increase_ratio'] = (
                item_data['current_bid'] / item_data['starting_bid']
                if item_data['starting_bid'] > 0 else 0
            )
        else:
            ml_features['bid_increase_from_start'] = None
            ml_features['bid_increase_ratio'] = None
        
        if item_data.get('bid_count') and item_data.get('viewed'):
            ml_features['bid_to_view_ratio'] = item_data['bid_count'] / item_data['viewed']
        else:
            ml_features['bid_to_view_ratio'] = None
    
    # Extract data from generatedDescription (enriched API)
    gen_desc = enriched_data.get('generatedDescription', {})
    
    if not isinstance(gen_desc, dict):
        print(f"Warning: generatedDescription not found or invalid for item {item_id}")
        return ml_features
    
    # Text features
    ml_features['title'] = gen_desc.get('title')
    ml_features['title_length'] = len(gen_desc.get('title', ''))
    ml_features['description'] = gen_desc.get('description')
    ml_features['description_length'] = len(gen_desc.get('description', ''))
    ml_features['qualitative_description'] = gen_desc.get('qualitativeDescription')
    ml_features['qualitative_description_length'] = len(gen_desc.get('qualitativeDescription', ''))
    
    # Brand features
    ml_features['brand'] = gen_desc.get('brand')
    ml_features['has_brand'] = 1 if gen_desc.get('brand') else 0
    ml_features['series_line'] = gen_desc.get('seriesLine')
    ml_features['has_series_line'] = 1 if gen_desc.get('seriesLine') else 0
    
    # Condition and working status
    ml_features['condition'] = gen_desc.get('condition')
    ml_features['is_new'] = 1 if gen_desc.get('condition') == 'New' else 0
    ml_features['is_used'] = 1 if gen_desc.get('condition') in ['Used', 'Good', 'Fair', 'Poor'] else 0
    ml_features['working_status'] = gen_desc.get('working')
    ml_features['is_working'] = 1 if gen_desc.get('working') == 'Yes' else 0
    
    # Item complexity features
    ml_features['single_key_item'] = gen_desc.get('singleKeyItem')
    ml_features['is_single_item'] = 1 if gen_desc.get('singleKeyItem') else 0
    ml_features['num_items'] = gen_desc.get('numItems', 0)
    ml_features['is_multi_item'] = 1 if gen_desc.get('numItems', 0) > 1 else 0
    
    # Brands list (multiple brands possible)
    brands = gen_desc.get('brands', [])
    if brands and isinstance(brands, list):
        ml_features['brands_list'] = brands
        ml_features['brands_count'] = len(brands)
        ml_features['has_multiple_brands'] = 1 if len(brands) > 1 else 0
        ml_features['primary_brand'] = brands[0] if brands else None
    else:
        ml_features['brands_list'] = []
        ml_features['brands_count'] = 0
        ml_features['has_multiple_brands'] = 0
        ml_features['primary_brand'] = None
    
    # Categories list
    categories = gen_desc.get('categories', [])
    if categories and isinstance(categories, list):
        ml_features['categories_list'] = categories
        ml_features['categories_count'] = len(categories)
        ml_features['has_multiple_categories'] = 1 if len(categories) > 1 else 0
        ml_features['primary_category'] = categories[0] if categories else None
    else:
        ml_features['categories_list'] = []
        ml_features['categories_count'] = 0
        ml_features['has_multiple_categories'] = 0
        ml_features['primary_category'] = None
    
    # Items breakdown (for multi-item lots)
    items = gen_desc.get('items', [])
    if items and isinstance(items, list):
        ml_features['items_breakdown'] = [
            {
                'title': item.get('title'),
                'category': item.get('category')
            }
            for item in items if isinstance(item, dict)
        ]
        ml_features['items_breakdown_count'] = len(items)
        ml_features['has_items_breakdown'] = 1
    else:
        ml_features['items_breakdown'] = []
        ml_features['items_breakdown_count'] = 0
        ml_features['has_items_breakdown'] = 0
    
    # Attributes (specifications)
    attributes = gen_desc.get('attributes', [])
    if attributes and isinstance(attributes, list):
        ml_features['attributes'] = [
            {
                'name': attr.get('name'),
                'value': attr.get('value')
            }
            for attr in attributes if isinstance(attr, dict)
        ]
        ml_features['attributes_count'] = len(attributes)
        ml_features['has_attributes'] = 1
        
        # Extract specific common attributes as features
        attr_dict = {attr.get('name'): attr.get('value') for attr in attributes if isinstance(attr, dict)}
        ml_features['material'] = attr_dict.get('Material')
        ml_features['color'] = attr_dict.get('Color')
        ml_features['size'] = attr_dict.get('Size')
        ml_features['dimensions'] = attr_dict.get('Dimensions')
    else:
        ml_features['attributes'] = []
        ml_features['attributes_count'] = 0
        ml_features['has_attributes'] = 0
        ml_features['material'] = None
        ml_features['color'] = None
        ml_features['size'] = None
        ml_features['dimensions'] = None
    
    # Derived features
    ml_features['total_text_length'] = (
        ml_features['title_length'] + 
        ml_features['description_length'] + 
        ml_features['qualitative_description_length']
    )
    ml_features['has_detailed_description'] = 1 if ml_features['description_length'] > 100 else 0
    ml_features['complexity_score'] = (
        ml_features['brands_count'] + 
        ml_features['categories_count'] + 
        ml_features['attributes_count'] +
        ml_features['items_breakdown_count']
    )
    
    return ml_features


def save_to_json(data: Dict[str, Any], output_path: str):
    """Save transformed data to JSON file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nData saved to: {output_path}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")


def main(item_id: str = "7433915", auction_id: Optional[str] = None, output_path: Optional[str] = None):
    """
    Main function to extract and transform enriched item details
    
    Args:
        item_id: MaxSold item ID to fetch (default: 7433915)
        auction_id: MaxSold auction ID (optional, will be fetched from enriched API if not provided)
        output_path: Path to save JSON file (default: auto-generated)
    """
    
    print("=" * 70)
    print("MaxSold Item Enriched Details Extractor for ML")
    print("=" * 70)
    print(f"Item ID: {item_id}")
    if auction_id:
        print(f"Auction ID: {auction_id}")
    print("=" * 70)
    
    enriched_data = None
    item_data = None
    auction_data = None
    
    try:
        # Fetch enriched data from API
        enriched_data = fetch_enriched_details(item_id)
        print(f"✓ Successfully fetched enriched data from API")
        
        # Get auction_id from enriched data if not provided
        if not auction_id:
            auction_id = enriched_data.get('amAuctionId')
            if auction_id:
                print(f"✓ Extracted auction_id from enriched data: {auction_id}")
            else:
                print("⚠ Warning: Could not determine auction_id from enriched data")
        
        # Fetch auction-level details if we have auction_id
        if auction_id:
            try:
                auction_api_response = fetch_auction_details(auction_id)
                auction_data = extract_auction_data(auction_api_response)
                if auction_data:
                    print(f"✓ Successfully fetched auction-level data from API")
                else:
                    print(f"⚠ Warning: Could not extract auction data from API response")
            except Exception as e:
                print(f"⚠ Warning: Failed to fetch auction-level data: {e}")
                print("  Continuing without auction-level data...")
            
            # Fetch item-level details
            try:
                item_api_response = fetch_item_details(auction_id, item_id)
                item_data = extract_item_data(item_api_response)
                if item_data:
                    print(f"✓ Successfully fetched item-level data from items API")
                else:
                    print(f"⚠ Warning: Could not extract item data from items API response")
            except Exception as e:
                print(f"⚠ Warning: Failed to fetch item-level data: {e}")
                print("  Continuing with enriched data only...")
        
        # Transform for ML
        ml_data = transform_for_ml(enriched_data, item_data, auction_data, item_id, auction_id)
        
        if not ml_data:
            print("✗ Failed to transform data")
            return
        
        print(f"✓ Successfully transformed data")
        print(f"\nExtracted Features:")
        print(f"  - Auction ID: {ml_data.get('auction_id')}")
        print(f"  - Auction Title: {ml_data.get('auction_title', 'N/A')[:50]}...")
        print(f"  - Auction Starts: {ml_data.get('auction_starts')}")
        print(f"  - Catalog Lots: {ml_data.get('catalog_lots')}")
        print(f"  - Lot Number: {ml_data.get('lot_number')}")
        print(f"  - Item Title: {ml_data.get('title', 'N/A')[:50]}...")
        print(f"  - Current Bid: ${ml_data.get('current_bid', 0)}")
        print(f"  - Bid Count: {ml_data.get('bid_count', 0)}")
        print(f"  - Views: {ml_data.get('viewed_count', 0)}")
        print(f"  - Categories: {ml_data.get('categories_count', 0)}")
        print(f"  - Brands: {ml_data.get('brands_count', 0)}")
        print(f"  - Condition: {ml_data.get('condition')}")
        
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"data/ml_ready/item_{item_id}_ml_features_{timestamp}.json"
        
        # Save to JSON
        save_to_json(ml_data, output_path)
        
        print("\n" + "=" * 70)
        print("✓ Extraction and transformation complete!")
        print("=" * 70)
        
        # Print sample of features
        print("\nSample ML Features (first 20):")
        for i, (key, value) in enumerate(list(ml_data.items())[:20]):
            if isinstance(value, str) and len(value) > 60:
                value = value[:60] + "..."
            print(f"  {key}: {value}")
        
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}")
        if e.response.status_code == 404:
            print(f"Item {item_id} not found")
        else:
            print(f"Status code: {e.response.status_code}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract MaxSold item enriched details and transform for ML"
    )
    parser.add_argument(
        "item_id",
        nargs="?",
        default="7433915",
        help="Item ID to fetch (default: 7433915)"
    )
    parser.add_argument(
        "-a", "--auction-id",
        help="Auction ID (optional, will be auto-detected if not provided)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    main(item_id=args.item_id, auction_id=args.auction_id, output_path=args.output)
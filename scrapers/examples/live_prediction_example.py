#!/usr/bin/env python3
"""
Example: Using extractors for live ML prediction
This demonstrates how to fetch and extract data from a URL without saving to disk
Perfect for a live ML model that takes auction/item URLs as input
"""

import sys
from pathlib import Path

# Add scrapers to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors.auction_details import fetch_auction_details, extract_auction_from_json
from extractors.item_details import fetch_auction_items, extract_items_from_json
from extractors.bid_history import fetch_item_bid_history, extract_bids_from_json
from extractors.item_enriched import fetch_enriched_details, extract_enriched_data


def get_auction_data_for_prediction(auction_id: str) -> dict:
    """
    Fetch all data for an auction for live ML prediction
    Returns structured data without saving to disk
    
    Args:
        auction_id: The MaxSold auction ID (e.g., "12345")
    
    Returns:
        Dictionary containing auction details, items, and bid history
    """
    print(f"Fetching data for auction {auction_id}...")
    
    result = {
        'auction_id': auction_id,
        'auction_details': None,
        'items': [],
        'error': None
    }
    
    try:
        # 1. Fetch auction details
        print("  → Fetching auction details...")
        auction_json = fetch_auction_details(auction_id)
        auction_details = extract_auction_from_json(auction_json, auction_id)
        result['auction_details'] = auction_details
        
        # 2. Fetch all items in the auction
        print("  → Fetching items...")
        items_json = fetch_auction_items(auction_id)
        items = extract_items_from_json(items_json, auction_id)
        result['items'] = items
        print(f"  ✓ Found {len(items)} items")
        
        # 3. Optionally fetch bid history for each item
        # Uncomment this if you need bid history for predictions
        # for item in items:
        #     item_id = item.get('id')
        #     if item_id:
        #         bids_json = fetch_item_bid_history(auction_id, item_id)
        #         bids = extract_bids_from_json(bids_json, auction_id, item_id)
        #         item['bid_history'] = bids
        
        print(f"✓ Successfully fetched data for auction {auction_id}")
        
    except Exception as e:
        print(f"✗ Error fetching auction data: {e}")
        result['error'] = str(e)
    
    return result


def get_item_enriched_data(item_id: str) -> dict:
    """
    Fetch enriched AI-generated data for a single item
    
    Args:
        item_id: The MaxSold item ID
    
    Returns:
        Dictionary with enriched item details (categories, brands, attributes)
    """
    print(f"Fetching enriched data for item {item_id}...")
    
    try:
        enriched_json = fetch_enriched_details(item_id)
        enriched_data = extract_enriched_data(enriched_json, item_id)
        
        if enriched_data:
            print(f"✓ Successfully fetched enriched data")
            return enriched_data
        else:
            print(f"✗ No enriched data found")
            return {'item_id': item_id, 'error': 'No data found'}
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return {'item_id': item_id, 'error': str(e)}


def predict_item_value(item_id: str, auction_id: str) -> dict:
    """
    Example ML prediction workflow for a single item
    
    This shows how you'd structure a live prediction that:
    1. Takes an item URL/ID as input
    2. Fetches all relevant data
    3. Runs your ML model
    4. Returns a prediction
    
    Args:
        item_id: The item ID
        auction_id: The auction ID containing the item
    
    Returns:
        Dictionary with prediction results
    """
    print(f"\n{'='*60}")
    print(f"LIVE PREDICTION FOR ITEM {item_id}")
    print(f"{'='*60}\n")
    
    # Fetch all necessary data
    auction_data = get_auction_data_for_prediction(auction_id)
    
    if auction_data['error']:
        return {'error': auction_data['error']}
    
    # Find the specific item
    target_item = None
    for item in auction_data['items']:
        if str(item.get('id')) == str(item_id):
            target_item = item
            break
    
    if not target_item:
        return {'error': f'Item {item_id} not found in auction {auction_id}'}
    
    # Fetch enriched data
    enriched_data = get_item_enriched_data(item_id)
    
    # Combine all features
    features = {
        'auction': auction_data['auction_details'],
        'item': target_item,
        'enriched': enriched_data,
    }
    
    print("\nExtracted features:")
    print(f"  - Auction title: {features['auction'].get('title', 'N/A')}")
    print(f"  - Item title: {features['item'].get('title', 'N/A')}")
    print(f"  - Current bid: ${features['item'].get('current_bid', 0)}")
    print(f"  - Bid count: {features['item'].get('bid_count', 0)}")
    print(f"  - AI brand: {features['enriched'].get('brand', 'N/A')}")
    print(f"  - AI categories: {features['enriched'].get('categories_count', 0)} found")
    
    # THIS IS WHERE YOU'D CALL YOUR ML MODEL
    # For example:
    # prediction = your_ml_model.predict(features)
    
    # Placeholder prediction
    prediction = {
        'item_id': item_id,
        'auction_id': auction_id,
        'predicted_final_price': 125.50,  # Your model output
        'confidence': 0.85,
        'features_used': len(features),
        'model_version': '1.0.0',
    }
    
    print(f"\n🎯 PREDICTION: ${prediction['predicted_final_price']:.2f}")
    print(f"   Confidence: {prediction['confidence']*100:.1f}%")
    
    return prediction


def main():
    """
    Main demo function
    """
    print("="*60)
    print("LIVE ML PREDICTION EXAMPLE")
    print("Using refactored extractors for real-time predictions")
    print("="*60)
    print()
    
    # Example: Predict value for a specific item
    # Replace these with real IDs from MaxSold
    example_auction_id = "12345"  # Replace with real auction ID
    example_item_id = "67890"      # Replace with real item ID
    
    print("NOTE: This example uses placeholder IDs.")
    print("Replace with real MaxSold auction/item IDs to test.\n")
    
    # Uncomment to run with real IDs:
    # result = predict_item_value(example_item_id, example_auction_id)
    # print("\nFull result:", result)
    
    # Show just the extractor usage
    print("Example extractor usage:\n")
    print("1. For auction details:")
    print("   from extractors.auction_details import fetch_auction_details, extract_auction_from_json")
    print("   data = fetch_auction_details('12345')")
    print("   parsed = extract_auction_from_json(data, '12345')")
    print()
    print("2. For item enriched data:")
    print("   from extractors.item_enriched import fetch_enriched_details, extract_enriched_data")
    print("   data = fetch_enriched_details('67890')")
    print("   parsed = extract_enriched_data(data, '67890')")
    print()
    print("The extractors return Python dictionaries - no file I/O!")
    print("Perfect for live predictions where you don't want to save data.")


if __name__ == "__main__":
    main()

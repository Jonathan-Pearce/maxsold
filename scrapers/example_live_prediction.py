"""
Example: Using extractors for live ML prediction without file I/O
This demonstrates how to use the refactored extraction code for real-time predictions
"""

from extractors.auction_search import fetch_sales_search, extract_sales_from_json
from extractors.auction_details import fetch_auction_details, extract_auction_from_json
from extractors.item_details import fetch_auction_items, extract_items_from_json
from extractors.bid_history import fetch_item_bid_history, extract_bids_from_json
from extractors.item_enriched import fetch_enriched_details, extract_enriched_data


def get_live_auction_data(auction_id: str):
    """
    Fetch all data for an auction in real-time without saving to disk
    
    Args:
        auction_id: The auction ID to fetch
        
    Returns:
        Dictionary with auction details, items, and bid histories
    """
    print(f"\n{'='*60}")
    print(f"Fetching live data for auction {auction_id}")
    print(f"{'='*60}\n")
    
    # 1. Get auction details
    print("Step 1: Fetching auction details...")
    auction_data = fetch_auction_details(auction_id)
    auction = extract_auction_from_json(auction_data, auction_id)
    
    if not auction:
        print("ERROR: Could not extract auction details")
        return None
    
    print(f"✓ Auction: {auction['title']}")
    print(f"  Starts: {auction['starts']}")
    print(f"  Ends: {auction['ends']}")
    
    # 2. Get all items for this auction
    print("\nStep 2: Fetching item details...")
    items_data = fetch_auction_items(auction_id)
    items = extract_items_from_json(items_data, auction_id)
    
    print(f"✓ Found {len(items)} items")
    
    # 3. Get bid history for first few items (as example)
    print("\nStep 3: Fetching bid history for first 3 items...")
    items_with_bids = []
    
    for i, item in enumerate(items[:3]):  # Just first 3 as example
        item_id = item['id']
        print(f"  Item {i+1}/{min(3, len(items))}: {item['title'][:50]}...")
        
        try:
            bid_data = fetch_item_bid_history(auction_id, item_id)
            bids = extract_bids_from_json(bid_data, auction_id, item_id)
            
            items_with_bids.append({
                'item': item,
                'bids': bids
            })
            
            print(f"    ✓ {len(bids)} bids")
        except Exception as e:
            print(f"    ✗ Error fetching bids: {e}")
            items_with_bids.append({
                'item': item,
                'bids': []
            })
    
    # 4. Get enriched details for first item (as example)
    print("\nStep 4: Fetching enriched details for first item...")
    enriched_items = []
    
    if items:
        first_item_id = items[0]['id']
        try:
            enriched_data = fetch_enriched_details(first_item_id)
            enriched = extract_enriched_data(enriched_data, first_item_id)
            
            if enriched:
                enriched_items.append(enriched)
                print(f"  ✓ Brand: {enriched.get('brand', 'N/A')}")
                print(f"  ✓ Condition: {enriched.get('condition', 'N/A')}")
                print(f"  ✓ Categories: {enriched.get('categories_count', 0)}")
        except Exception as e:
            print(f"  ✗ Error fetching enriched details: {e}")
    
    # Return structured data
    result = {
        'auction': auction,
        'items': items,
        'items_with_bids': items_with_bids,
        'enriched_items': enriched_items
    }
    
    print(f"\n{'='*60}")
    print("Data fetching complete!")
    print(f"{'='*60}\n")
    
    return result


def predict_item_final_price(auction_id: str, item_id: str):
    """
    Example: Predict final price for an item using live data
    This is a placeholder - replace with your actual ML model
    
    Args:
        auction_id: The auction ID
        item_id: The item ID
        
    Returns:
        Predicted price (placeholder logic)
    """
    print(f"\nPredicting price for item {item_id} in auction {auction_id}...")
    
    # Fetch item details
    items_data = fetch_auction_items(auction_id)
    items = extract_items_from_json(items_data, auction_id)
    item = next((i for i in items if str(i['id']) == str(item_id)), None)
    
    if not item:
        print(f"ERROR: Item {item_id} not found")
        return None
    
    print(f"Item: {item['title']}")
    print(f"Current bid: ${item.get('current_bid', 0)}")
    print(f"Bid count: {item.get('bid_count', 0)}")
    
    # Fetch bid history
    bid_data = fetch_item_bid_history(auction_id, item_id)
    bids = extract_bids_from_json(bid_data, auction_id, item_id)
    
    print(f"Total bids: {len(bids)}")
    
    # Fetch enriched details
    try:
        enriched_data = fetch_enriched_details(item_id)
        enriched = extract_enriched_data(enriched_data, item_id)
        
        if enriched:
            print(f"Brand: {enriched.get('brand', 'N/A')}")
            print(f"Condition: {enriched.get('condition', 'N/A')}")
    except:
        enriched = None
        print("No enriched data available")
    
    # Placeholder prediction logic
    # TODO: Replace with actual ML model
    current_bid = item.get('current_bid', 0) or 0
    bid_count = item.get('bid_count', 0) or 0
    
    # Simple heuristic: estimate 10-20% increase based on bid activity
    if bid_count > 10:
        predicted_price = current_bid * 1.20
    elif bid_count > 5:
        predicted_price = current_bid * 1.15
    elif bid_count > 0:
        predicted_price = current_bid * 1.10
    else:
        predicted_price = current_bid
    
    print(f"\n💰 Predicted final price: ${predicted_price:.2f}")
    print(f"   (based on current bid: ${current_bid}, {bid_count} bids)")
    
    return predicted_price


def search_nearby_auctions(lat: float, lng: float, radius_km: float = 50):
    """
    Example: Search for auctions near a location
    
    Args:
        lat: Latitude
        lng: Longitude
        radius_km: Search radius in kilometers
        
    Returns:
        List of auction data
    """
    print(f"\nSearching for auctions near ({lat}, {lng}) within {radius_km}km...")
    
    # Fetch sales search results
    data = fetch_sales_search(
        lat=lat,
        lng=lng,
        radius_metres=int(radius_km * 1000),
        days=30,
        limit=10  # Just first 10 for example
    )
    
    sales = extract_sales_from_json(data)
    
    print(f"\nFound {len(sales)} auctions:")
    for i, sale in enumerate(sales, 1):
        distance_km = (sale.get('distanceMeters') or 0) / 1000
        print(f"{i}. {sale['title']}")
        print(f"   Distance: {distance_km:.1f}km")
        print(f"   Lots: {sale.get('numberLots', 'N/A')}, Bids: {sale.get('totalBids', 'N/A')}")
        print()
    
    return sales


if __name__ == "__main__":
    # Example 1: Get all data for a specific auction
    print("\n" + "="*60)
    print("EXAMPLE 1: Fetch complete auction data")
    print("="*60)
    
    # Replace with a real auction ID
    auction_id = "12345"  # TODO: Replace with real auction ID
    # data = get_live_auction_data(auction_id)
    
    # Example 2: Predict price for specific item
    print("\n" + "="*60)
    print("EXAMPLE 2: Predict item price")
    print("="*60)
    
    # Replace with real IDs
    # auction_id = "12345"
    # item_id = "67890"
    # predicted_price = predict_item_final_price(auction_id, item_id)
    
    # Example 3: Search nearby auctions
    print("\n" + "="*60)
    print("EXAMPLE 3: Search nearby auctions")
    print("="*60)
    
    # Toronto coordinates
    toronto_lat = 43.653226
    toronto_lng = -79.3831843
    
    nearby_auctions = search_nearby_auctions(toronto_lat, toronto_lng, radius_km=50)
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nNOTE: These examples use placeholder logic.")
    print("Replace with your actual ML model for real predictions.")

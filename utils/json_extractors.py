
import json
from typing import Any, Dict, List, Optional


# Run from 01_extract_auction_search.py 
def extract_sales_from_json(data: Any) -> List[Dict[str, Any]]:
    """Extract sale items from JSON response"""
    # API typically returns: { "sales": [...], "total": N }
    sales = []
    
    if isinstance(data, dict):
        sales_list = data.get("sales") or data.get("results") or data.get("data")
        if isinstance(sales_list, list):
            sales = sales_list
        elif isinstance(data, dict) and any(k in data for k in ["amAuctionId", "title"]):
            # Single sale object
            sales = [data]
    elif isinstance(data, list):
        sales = data
    
    extracted = []
    for sale in sales:
        if not isinstance(sale, dict):
            continue
        
        # Extract address fields
        address = sale.get("address", {}) or {}
        city = address.get("city", "")
        region = address.get("region", "")
        country = address.get("country", "")
        
        # Extract latLng fields
        latlng = address.get("latLng", {}) or {}
        lat = latlng.get("lat", None)
        lng = latlng.get("lng", None)
        
        row = {
            "amAuctionId": sale.get("amAuctionId", ""),
            "title": sale.get("title", ""),
            "displayRegion": sale.get("displayRegion", ""),
            "distanceMeters": sale.get("distanceMeters", None),
            "saleType": sale.get("saleType", ""),
            "saleCategory": sale.get("saleCategory", ""),
            "openTime": sale.get("openTime", ""),
            "closeTime": sale.get("closeTime", ""),
            "totalBids": sale.get("totalBids", None),
            "numberLots": sale.get("numberLots", None),
            "hasShipping": sale.get("hasShipping", None),
            "city": city,
            "region": region,
            "country": country,
            "lat": lat,
            "lng": lng,
        }
        extracted.append(row)
    
    return extracted


# Run from 02_extract_auction_details.py
def extract_auction_from_json(data: Any, auction_id: str) -> Optional[Dict[str, Any]]:
    """Extract auction details from JSON response"""
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
        print(f"Could not locate auction object in response for {auction_id}", file=sys.stderr)
        return None
    
    # Extract fields
    row = {
        "auction_id": auction_id,
        "id": auction.get("id", ""),
        "title": auction.get("title", ""),
        "starts": auction.get("starts", ""),
        "ends": auction.get("ends", ""),
        "last_item_closes": auction.get("last_item_closes", ""),
        "removal_info": auction.get("removal_info", ""),
        "inspection_info": auction.get("inspection_info", ""),
        "intro": auction.get("intro", ""),
        "pickup_time": auction.get("pickup_time", ""),
        "partner_url": auction.get("partner_url", ""),
        "extended_bidding": auction.get("extended_bidding", None),
        "extended_bidding_interval": auction.get("extended_bidding_interval", None),
        "extended_bidding_threshold": auction.get("extended_bidding_threshold", None),
        "catalog_lots": auction.get("catalog_lots", None),
    }
    
    return row


#Run 03_extract_items_details.py
def extract_items_from_json(data: Any, auction_id: str, item_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Extract item details from JSON response
    
    Args:
        data: JSON response data
        auction_id: Auction ID
        item_ids: Optional list of specific item IDs to extract. If None, extracts all items.
    
    Returns:
        List of extracted item dictionaries
    """
    # Navigate to items array
    items = None
    
    if isinstance(data, dict):
        # Try common paths
        if "auction" in data and isinstance(data["auction"], dict):
            auction_obj = data["auction"]
            if "items" in auction_obj and isinstance(auction_obj["items"], list):
                items = auction_obj["items"]
        
        # Try direct items key
        if items is None and "items" in data and isinstance(data["items"], list):
            items = data["items"]
        
        # Try other common paths
        if items is None:
            for key in ["results", "data", "lots", "auctionItems"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
    
    if not items:
        print(f"Could not locate items array in response for auction {auction_id}", file=sys.stderr)
        return []
    
    extracted_items = []
    
    # Convert item_ids to strings for comparison if provided
    filter_ids = [str(id) for id in item_ids] if item_ids else None
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        # Filter by item_ids if provided
        item_id = str(item.get("id", ""))
        if filter_ids and item_id not in filter_ids:
            continue
        
        # Extract fields
        row = {
            "id": item.get("id", ""),
            "auction_id": auction_id,
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "taxable": item.get("taxable", None),
            "viewed": item.get("viewed", None),
            "minimum_bid": item.get("minimum_bid", None),
            "starting_bid": item.get("starting_bid", None),
            "current_bid": item.get("current_bid", None),
            "proxy_bid": item.get("proxy_bid", None),
            "start_time": item.get("start_time", ""),
            "end_time": item.get("end_time", ""),
            "lot_number": item.get("lot_number", ""),
            "bid_count": item.get("bid_count", None),
            "bidding_extended": item.get("bidding_extended", None),
            "buyer_premium": item.get("buyer_premium", None),
            "timezone": item.get("timezone", ""),
        }
        
        extracted_items.append(row)
    
    return extracted_items


#Run from 04_extract_bid_history.py
def extract_bids_from_json(data: Any, auction_id: str, item_id: str) -> List[Dict[str, Any]]:
    """Extract bid history from JSON response"""
    # Navigate to bid history array
    bids = None
    
    # Try to find the item and its bid history
    if isinstance(data, dict):
        # Try common paths for auction/items structure
        auction_obj = None
        if "auction" in data and isinstance(data["auction"], dict):
            auction_obj = data["auction"]
        
        # Find items array
        items_list = None
        if auction_obj and "items" in auction_obj and isinstance(auction_obj["items"], list):
            items_list = auction_obj["items"]
        elif "items" in data and isinstance(data["items"], list):
            items_list = data["items"]
        
        # Find the specific item
        if items_list:
            for item in items_list:
                if not isinstance(item, dict):
                    continue
                item_id_str = str(item.get("id", ""))
                if item_id_str == str(item_id):
                    # Found the item, look for bid history
                    for key in ["bid_history", "bidHistory", "bids", "bid_history_list"]:
                        if key in item and isinstance(item[key], list):
                            bids = item[key]
                            break
                    # Sometimes bid_history is nested: [[{...}, ...]]
                    if bids and isinstance(bids[0], list):
                        bids = bids[0]
                    break
    
    if not bids:
        return []
    
    extracted_bids = []
    
    for i, bid in enumerate(bids, start=1):
        if not isinstance(bid, dict):
            continue
        
        # Extract fields with various possible key names
        time_of_bid = (bid.get("time_of_bid") or bid.get("timeOfBid") or 
                       bid.get("time") or bid.get("bidTime") or 
                       bid.get("createdAt") or "")
        
        amount = (bid.get("amount") or bid.get("bidAmount") or 
                  bid.get("value") or bid.get("currentBid") or None)
        
        isproxy = (bid.get("isproxy") or bid.get("isProxy") or 
                   bid.get("proxy") or bid.get("is_proxy") or 
                   bid.get("isProxyBid") or False)
        
        row = {
            "auction_id": auction_id,
            "item_id": item_id,
            "bid_number": i,
            "time_of_bid": str(time_of_bid),
            "amount": amount,
            "isproxy": bool(isproxy),
        }
        
        extracted_bids.append(row)
    
    return extracted_bids


#Run from 05_extract_enriched_item_data.py
def extract_enriched_data(data: Any, item_id: str) -> Optional[Dict[str, Any]]:
    """Parse the enriched item data and extract required fields"""
    
    if not isinstance(data, dict):
        return None
    
    result = {
        'item_id': item_id,
        'amLotId': data.get('amLotId'),
        'amAuctionId': data.get('amAuctionId'),
    }
    
    # Extract data from generatedDescription
    gen_desc = data.get('generatedDescription', {})
    
    if not isinstance(gen_desc, dict):
        return result
    
    # Simple fields
    result['title'] = gen_desc.get('title')
    result['description'] = gen_desc.get('description')
    result['qualitativeDescription'] = gen_desc.get('qualitativeDescription')
    result['brand'] = gen_desc.get('brand')
    result['seriesLine'] = gen_desc.get('seriesLine')
    result['condition'] = gen_desc.get('condition')
    result['working'] = gen_desc.get('working')
    result['singleKeyItem'] = gen_desc.get('singleKeyItem')
    result['numItems'] = gen_desc.get('numItems')
    
    # Extract brands (first 10)
    brands = gen_desc.get('brands', [])
    if brands and isinstance(brands, list):
        brands_subset = brands[:10] if len(brands) > 10 else brands
        result['brands'] = json.dumps(brands_subset)
        result['brands_count'] = len(brands)
    else:
        result['brands'] = None
        result['brands_count'] = 0
    
    # Extract categories (first 10)
    categories = gen_desc.get('categories', [])
    if categories and isinstance(categories, list):
        categories_subset = categories[:10] if len(categories) > 10 else categories
        result['categories'] = json.dumps(categories_subset)
        result['categories_count'] = len(categories)
    else:
        result['categories'] = None
        result['categories_count'] = 0
    
    # Extract items (first 10) - save title and category
    items = gen_desc.get('items', [])
    if items and isinstance(items, list):
        items_subset = items[:10] if len(items) > 10 else items
        items_parsed = [
            {
                'title': item.get('title'),
                'category': item.get('category')
            }
            for item in items_subset if isinstance(item, dict)
        ]
        result['items'] = json.dumps(items_parsed)
        result['items_count'] = len(items)
    else:
        result['items'] = None
        result['items_count'] = 0
    
    # Extract attributes (first 10) - save name and value
    attributes = gen_desc.get('attributes', [])
    if attributes and isinstance(attributes, list):
        attributes_subset = attributes[:10] if len(attributes) > 10 else attributes
        attributes_parsed = [
            {
                'name': attr.get('name'),
                'value': attr.get('value')
            }
            for attr in attributes_subset if isinstance(attr, dict)
        ]
        result['attributes'] = json.dumps(attributes_parsed)
        result['attributes_count'] = len(attributes)
    else:
        result['attributes'] = None
        result['attributes_count'] = 0
    
    return result
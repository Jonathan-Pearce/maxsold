"""
Item details data extraction
Fetches and extracts item/lot details from MaxSold API
Can be used standalone for live predictions or as part of the scraping pipeline
"""

import requests
from typing import Any, Dict, List
import sys
from utils.config import HEADERS, API_URLS


def fetch_auction_items(auction_id: str, timeout: int = 30) -> Any:
    """
    Fetch auction items from MaxSold API
    
    Args:
        auction_id: The auction ID to fetch items for
        timeout: Request timeout in seconds
        
    Returns:
        JSON response from API
    """
    params = {"auctionid": auction_id}
    
    print(f"Fetching items for auction {auction_id}...")
    r = requests.get(API_URLS["auction_items"], params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_items_from_json(data: Any, auction_id: str) -> List[Dict[str, Any]]:
    """
    Extract item details from JSON response
    
    Args:
        data: JSON response from API
        auction_id: The auction ID these items belong to
        
    Returns:
        List of dictionaries with extracted item data
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
    
    for item in items:
        if not isinstance(item, dict):
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

"""
Bid history data extraction
Fetches and extracts bid history for items from MaxSold API
Can be used standalone for live predictions or as part of the scraping pipeline
"""

import requests
from typing import Any, Dict, List
import sys
from utils.config import HEADERS, API_URLS


def fetch_item_bid_history(auction_id: str, item_id: str, timeout: int = 30) -> Any:
    """
    Fetch item bid history from MaxSold API
    
    Args:
        auction_id: The auction ID
        item_id: The item ID to fetch bid history for
        timeout: Request timeout in seconds
        
    Returns:
        JSON response from API
    """
    params = {
        "auctionid": auction_id,
        "itemid": item_id
    }
    
    r = requests.get(API_URLS["auction_items"], params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_bids_from_json(data: Any, auction_id: str, item_id: str) -> List[Dict[str, Any]]:
    """
    Extract bid history from JSON response
    
    Args:
        data: JSON response from API
        auction_id: The auction ID
        item_id: The item ID
        
    Returns:
        List of dictionaries with extracted bid data
    """
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
            "bid_sequence": i,
            "time_of_bid": time_of_bid,
            "amount": amount,
            "isproxy": isproxy,
        }
        
        extracted_bids.append(row)
    
    return extracted_bids

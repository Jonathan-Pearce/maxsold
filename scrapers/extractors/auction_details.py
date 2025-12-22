"""
Auction details data extraction
Fetches and extracts detailed auction information from MaxSold API
Can be used standalone for live predictions or as part of the scraping pipeline
"""

import requests
from typing import Any, Dict, Optional
import sys
from utils.config import HEADERS, API_URLS


def fetch_auction_details(auction_id: str, timeout: int = 30) -> Any:
    """
    Fetch auction details from MaxSold API
    
    Args:
        auction_id: The auction ID to fetch details for
        timeout: Request timeout in seconds
        
    Returns:
        JSON response from API
    """
    params = {"auctionid": auction_id}
    
    print(f"Fetching auction {auction_id}...")
    r = requests.get(API_URLS["auction_items"], params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_auction_from_json(data: Any, auction_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract auction details from JSON response
    
    Args:
        data: JSON response from API
        auction_id: The auction ID being extracted
        
    Returns:
        Dictionary with extracted auction data, or None if extraction failed
    """
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

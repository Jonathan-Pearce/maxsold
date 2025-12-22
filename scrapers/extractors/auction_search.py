"""
Auction search data extraction
Fetches and extracts sales/auction search data from MaxSold API
Can be used standalone for live predictions or as part of the scraping pipeline
"""

import requests
from typing import Any, Dict, List
import sys
from utils.config import HEADERS, API_URLS


def fetch_sales_search(
    lat: float = 43.653226,
    lng: float = -79.3831843,
    radius_metres: int = 201168,
    country: str = "canada",
    page_number: int = 0,
    limit: int = 100,
    sale_state: str = "closed",
    days: int = 120,
    total: bool = True,
    timeout: int = 30
) -> Any:
    """
    Fetch sales search data from MaxSold API
    
    Args:
        lat: Latitude for search center
        lng: Longitude for search center
        radius_metres: Search radius in metres
        country: Country filter
        page_number: Page number for pagination
        limit: Number of results per page
        sale_state: State of sales to fetch (e.g., 'closed', 'open')
        days: Number of days back to search
        total: Whether to include total count in response
        timeout: Request timeout in seconds
        
    Returns:
        JSON response from API
    """
    params = {
        "lat": lat,
        "lng": lng,
        "radiusMetres": radius_metres,
        "country": country,
        "pageNumber": page_number,
        "limit": limit,
        "saleState": sale_state,
        "days": days,
        "total": str(total).lower()
    }
    
    print(f"Fetching sales data: page={page_number}, limit={limit}...")
    r = requests.get(API_URLS["auction_search"], params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_sales_from_json(data: Any) -> List[Dict[str, Any]]:
    """
    Extract sale items from JSON response
    
    Args:
        data: JSON response from API
        
    Returns:
        List of dictionaries with extracted sale data
    """
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

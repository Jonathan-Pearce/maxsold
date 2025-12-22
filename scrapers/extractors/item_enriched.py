"""
Item enriched details data extraction
Fetches and extracts AI-generated enriched item details from MaxSold API
Can be used standalone for live predictions or as part of the scraping pipeline
"""

import requests
import json
from typing import Any, Dict, Optional
import sys
from utils.config import HEADERS, API_URLS


def fetch_enriched_details(item_id: str, timeout: int = 30) -> Any:
    """
    Fetch enriched details for a single item from MaxSold API
    
    Args:
        item_id: The item ID to fetch enriched details for
        timeout: Request timeout in seconds
        
    Returns:
        JSON response from API
    """
    url = API_URLS["item_enriched"].format(item_id)
    
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_enriched_data(data: Any, item_id: str) -> Optional[Dict[str, Any]]:
    """
    Parse the enriched item data and extract required fields
    
    Args:
        data: JSON response from API
        item_id: The item ID being extracted
        
    Returns:
        Dictionary with extracted enriched data, or None if extraction failed
    """
    
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

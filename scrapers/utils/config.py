"""Shared configuration and constants"""

# HTTP Headers for API requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# API Endpoints
API_URLS = {
    "auction_search": "https://api.maxsold.com/sales/search",
    "auction_items": "https://maxsold.maxsold.com/msapi/auctions/items",
    "item_enriched": "https://api.maxsold.com/listings/am/{}/enriched",
}

# Default output directories
DEFAULT_OUTPUT_DIRS = {
    "auction_search": "data/raw_data/auction_search",
    "auction_details": "data/raw_data/auction_details",
    "item_details": "data/raw_data/items_details",
    "bid_history": "data/raw_data/bid_history",
    "item_enriched": "data/raw_data/item_enriched_details",
}

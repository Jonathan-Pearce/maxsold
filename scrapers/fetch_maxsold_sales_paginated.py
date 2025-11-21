import os
import json
import time
from datetime import date
from typing import Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

BASE_URL = (
    "https://api.maxsold.com/sales/search?"
    "lat=43.653226&lng=-79.3831843&radiusMetres=201168&country=canada"
    "&pageNumber=0&limit=24&saleState=closed&days=90&total=true"
)


def set_page_number(url: str, page: int) -> str:
    """Update pageNumber query parameter in URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params["pageNumber"] = [str(page)]
    new_query = urlencode(params, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def is_empty_response(data: Any) -> bool:
    """Check if API response is empty."""
    if data is None:
        return True
    if isinstance(data, list):
        return len(data) == 0
    if isinstance(data, dict):
        # Check for common response patterns
        if "sales" in data:
            return len(data.get("sales", [])) == 0
        if "results" in data:
            return len(data.get("results", [])) == 0
        if "data" in data:
            return len(data.get("data", [])) == 0
        # If no known key, check if any list value is non-empty
        for v in data.values():
            if isinstance(v, list) and len(v) > 0:
                return False
        # Empty dict or all lists are empty
        return True
    return False


def fetch_json(url: str, timeout: int = 15) -> Any:
    """Fetch JSON from API."""
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def scrape_sales_api(base_url: str = BASE_URL, data_root: str = "data", delay: float = 1.0) -> int:
    """
    Scrape MaxSold sales API with pagination.
    
    Args:
        base_url: API endpoint with initial parameters
        data_root: Root directory for saving data
        delay: Delay between requests in seconds
        
    Returns:
        Total number of pages saved
    """
    today_str = date.today().isoformat()
    output_dir = os.path.join(data_root, today_str)
    os.makedirs(output_dir, exist_ok=True)
    
    page = 0
    total_saved = 0
    
    while True:
        url = set_page_number(base_url, page)
        print(f"Fetching page {page}...")
        
        try:
            data = fetch_json(url)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
        
        # Check if response is empty
        if is_empty_response(data):
            print(f"Empty response at page {page}. Stopping.")
            break
        
        # Save JSON to file
        filename = f"sales_page_{page:04d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        total_saved += 1
        print(f"Saved page {page} → {filepath}")
        
        page += 1
        
        # Polite delay between requests
        if delay > 0:
            time.sleep(delay)
    
    print(f"\nCompleted! Saved {total_saved} pages to {output_dir}/")
    return total_saved


if __name__ == "__main__":
    import sys
    
    # Allow custom URL as first argument
    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    scrape_sales_api(url, data_dir)
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

def scrape_maxsold_auction(auction_id, offset=0):
    """
    Scrape auction URLs from a MaxSold auction page
    
    Args:
        auction_id: The auction ID (e.g., 102916)
        offset: Pagination offset (default: 0)
    
    Returns:
        List of item URLs
    """
    base_url = "https://maxsold.com"
    url = f"{base_url}/auction/{auction_id}/bidgallery?offset={offset}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all item links in the bid gallery
        item_urls = []
        
        # Look for item cards/links (adjust selectors based on actual HTML structure)
        items = soup.find_all('a', class_='item-link')  # Common pattern
        
        # Alternative selectors if the above doesn't work:
        if not items:
            items = soup.select('a[href*="/item/"]')
        
        for item in items:
            item_url = item.get('href')
            if item_url:
                full_url = urljoin(base_url, item_url)
                item_urls.append(full_url)
        
        return item_urls
        
    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        return []

def scrape_all_pages(auction_id, max_pages=None):
    """
    Scrape all pages of an auction
    
    Args:
        auction_id: The auction ID
        max_pages: Maximum number of pages to scrape (None for all)
    
    Returns:
        List of all item URLs
    """
    all_urls = []
    offset = 0
    page = 1
    
    while True:
        if max_pages and page > max_pages:
            break
            
        print(f"Scraping page {page} (offset: {offset})...")
        urls = scrape_maxsold_auction(auction_id, offset)
        
        if not urls:
            print(f"No more items found at offset {offset}")
            break
        
        all_urls.extend(urls)
        print(f"Found {len(urls)} items on page {page}")
        
        # Increment offset (typically by 100 or similar)
        offset += 72
        page += 1
        
        # Be respectful - add a delay between requests
        time.sleep(1)
    
    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(all_urls))
    return unique_urls

# Example usage
if __name__ == "__main__":
    auction_id = 102916
    
    # Scrape first page only
    print(f"Scraping auction {auction_id}...\n")
    urls = scrape_maxsold_auction(auction_id, offset=0)
    
    print(f"\nFound {len(urls)} items:")
    for url in urls:
        print(url)
    
    # To scrape all pages:
    # all_urls = scrape_all_pages(auction_id)
    # print(f"\nTotal unique items: {len(all_urls)}")
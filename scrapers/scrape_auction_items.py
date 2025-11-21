import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Set

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def scrape_auction_page(auction_id: int, offset: int = 0, timeout: int = 12) -> List[str]:
    """
    Scrape item URLs from a single MaxSold auction bidgallery page.
    
    Args:
        auction_id: The auction ID
        offset: Pagination offset
        timeout: Request timeout in seconds
        
    Returns:
        List of item URLs found on the page
    """
    url = f"https://maxsold.com/auction/{auction_id}/bidgallery?offset={offset}"
    print(url)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all links containing '/item/' in their href
        item_links = soup.find_all("a", href=lambda h: h and "/item/" in h)
        
        item_urls = []
        for link in item_links:
            href = link.get("href")
            if href:
                full_url = urljoin("https://maxsold.com", href)
                print(full_url)
                item_urls.append(full_url)
        
        return item_urls
        
    except requests.RequestException as e:
        print(f"Error fetching offset {offset}: {e}")
        return []


def scrape_all_auction_items(auction_id: int, max_pages: int = None, delay: float = 1.0) -> List[str]:
    """
    Scrape all item URLs from an auction across all pages.
    
    Args:
        auction_id: The auction ID
        max_pages: Maximum number of pages to scrape (None for all)
        delay: Delay between requests in seconds
        
    Returns:
        List of unique item URLs
    """
    all_urls: Set[str] = set()
    offset = 0
    page = 0
    
    while True:
        if max_pages is not None and page > max_pages:
            break
        
        print(f"Scraping page {page} (offset={offset})...")
        urls = scrape_auction_page(auction_id, offset)
        
        if not urls:
            print(f"No items found at offset {offset}. Stopping.")
            break
        
        new_urls = set(urls) - all_urls
        all_urls.update(new_urls)
        
        print(f"  Found {len(urls)} URLs ({len(new_urls)} new, {len(all_urls)} total)")
        
        # If no new URLs were found, we've likely reached the end
        if not new_urls:
            print("No new URLs found. Stopping pagination.")
            break
        
        offset += 72  # Typical pagination increment
        page += 1
        
        if delay > 0:
            time.sleep(delay)
    
    return sorted(all_urls)


def save_urls_to_csv(urls: List[str], output_csv: str, auction_id: int) -> None:
    """
    Save URLs to CSV with auction_id and url columns.
    
    Args:
        urls: List of URLs to save
        output_csv: Path to output CSV file
        auction_id: The auction ID
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["auction_id", "url"])
        
        for url in urls:
            writer.writerow([auction_id, url])
    
    print(f"\nSaved {len(urls)} unique URLs to {output_csv}")


def main():
    import sys
    
    # Get auction ID from command line or use default
    auction_id = int(sys.argv[1]) if len(sys.argv) > 1 else 102916
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_csv = sys.argv[3] if len(sys.argv) > 3 else f"data/auction/items_{auction_id}.csv"
    
    print(f"Scraping auction {auction_id}...\n")
    
    # Scrape all items
    urls = scrape_all_auction_items(auction_id, max_pages, delay=1.0)
    
    # Save to CSV
    if urls:
        save_urls_to_csv(urls, output_csv, auction_id)
    else:
        print("No URLs found.")


if __name__ == "__main__":
    main()
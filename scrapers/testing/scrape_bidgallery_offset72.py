import os
import csv
import sys
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def scrape_urls(page_url: str, timeout: int = 12):
    resp = requests.get(page_url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    out_dir = "/workspaces/maxsold/data"
    os.makedirs(out_dir, exist_ok=True)
    fname = re.sub(r'[^0-9A-Za-z]+', '_', page_url).strip('_') + ".txt"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print(f"Wrote prettified HTML to {out_path}")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/item/" in href:
            full = urljoin(page_url, href)
            links.append(full)

    # preserve order, remove duplicates
    unique = list(dict.fromkeys(links))
    return unique


def save_urls_csv(urls, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url"])
        for u in urls:
            w.writerow([u])


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://maxsold.com/auction/102916/bidgallery?offset=72"
    out = sys.argv[2] if len(sys.argv) > 2 else "/workspaces/maxsold/data/auction/items_102916_offset72.csv"

    urls = scrape_urls(url)
    save_urls_csv(urls, out)
    print(f"Saved {len(urls)} unique URLs to {out}")
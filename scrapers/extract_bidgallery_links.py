import re
import json
import os
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def fetch_bidgallery_links(url: str, timeout: int = 12) -> List[str]:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    base_netloc = urlparse(url).netloc
    hrefs = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # normalize
        full = urljoin(url, href)
        p = urlparse(full)
        # keep same-site links only
        if p.netloc and p.netloc != base_netloc:
            continue
        # accept listing/lot links (common MaxSold patterns)
        if re.search(r"/(?:listing|lot|item)/\d+", p.path, re.I) or re.search(r"/listing/|/lot/", p.path, re.I):
            hrefs.add(full)
    # return stable list
    return sorted(hrefs)

def save_links(links: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    data = {"count": len(links), "links": links}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://maxsold.com/auction/103482/bidgallery"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "auction_103482_links.json"

    try:
        links = fetch_bidgallery_links(url)
    except Exception as e:
        print("Fetch error:", e)
        sys.exit(2)

    save_links(links, out_file)
    print(f"Saved {len(links)} links -> {out_file}")
    if len(links) != 72:
        print(f"Warning: expected 72 links but found {len(links)}")
import re
import json
import os
import csv
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


def extract_listing_ids(links: List[str]) -> List[str]:
    """Return listing IDs extracted from URLs.

    IDs are numeric and are expected to appear in the URL as
    `/item/<id>` (optionally followed by query params like `?offset=`).
    """
    ids = []
    for u in links:
        m = re.search(r"/item/(\d+)", u)
        if m:
            ids.append(m.group(1))
    return ids


def save_ids(links: List[str], out_csv: str) -> None:
    """Write listing IDs and their source URL to a CSV file."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["listing_id", "url"])
        for u in links:
            m = re.search(r"/item/(\d+)", u)
            if m:
                writer.writerow([m.group(1), u])

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
    # also write listing IDs to CSV next to the JSON file
    csv_out = os.path.splitext(out_file)[0] + ".csv"
    save_ids(links, csv_out)
    print(f"Saved listing IDs -> {csv_out}")
    if len(links) != 72:
        print(f"Warning: expected 72 links but found {len(links)}")
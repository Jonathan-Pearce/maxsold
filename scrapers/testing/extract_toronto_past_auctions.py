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

TORONTO_PAST_URL = "https://maxsold.com/canada/ontario/toronto/past"

def fetch(url: str, timeout: int = 12) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def normalize_auction_bidgallery_url(url: str) -> str:
    # extract auction id and produce canonical bidgallery URL
    m = re.search(r"/auction/(\d+)", url)
    if not m:
        return ""
    auction_id = m.group(1)
    return f"https://maxsold.com/auction/{auction_id}/bidgallery"

def extract_auction_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        normalized = normalize_auction_bidgallery_url(full)
        if not normalized:
            continue
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out

def extract_auction_ids(links: List[str]) -> List[str]:
    ids = []
    seen = set()
    for l in links:
        m = re.search(r"/auction/(\d+)/bidgallery", l)
        if not m:
            continue
        aid = m.group(1)
        if aid not in seen:
            seen.add(aid)
            ids.append(aid)
    return sorted(ids, key=lambda x: int(x))

def save_csv_auction_ids(auction_ids: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AUCTION_ID"])
        for aid in auction_ids:
            writer.writerow([aid])

def save_json(links: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(links), "links": links}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else TORONTO_PAST_URL
    out_file = sys.argv[2] if len(sys.argv) > 2 else "toronto_past_auctions.csv"
    try:
        html = fetch(url)
        links = extract_auction_links(html, url)
        auction_ids = extract_auction_ids(links)
        save_csv_auction_ids(auction_ids, out_file)
        print(f"Saved {len(auction_ids)} auction ids -> {out_file}")
    except Exception as e:
        print("Error:", e)
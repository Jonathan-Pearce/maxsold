import os
import csv
import json
import time
from typing import Set, Any, Dict
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

DEFAULT_RATE_LIMIT_SLEEP = 0.25  # seconds between requests


def _merge_query(url: str, params: Dict[str, Any]) -> str:
    p = urlparse(url)
    q = parse_qs(p.query)
    for k, v in params.items():
        q[k] = [str(v)]
    new_q = urlencode({k: v[0] for k, v in q.items()})
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_q, p.fragment))


def _walk_for_amAuctionId(obj, out: Set[str]) -> None:
    if isinstance(obj, dict):
        if "amAuctionId" in obj:
            out.add(str(obj["amAuctionId"]))
        for v in obj.values():
            _walk_for_amAuctionId(v, out)
    elif isinstance(obj, list):
        for i in obj:
            _walk_for_amAuctionId(i, out)


def fetch_amAuctionIds_from_api(api_url: str, start_page: int = 1, limit: int = 24,
                                max_pages: int = 0, sleep: float = DEFAULT_RATE_LIMIT_SLEEP) -> Set[str]:
    """
    Paginate the provided MaxSold API search URL and extract all amAuctionId values.
    - api_url: full API endpoint (may already contain query params)
    - start_page: 1-based page to start from
    - limit: page size (will be set in query)
    - max_pages: if >0 stop after this many pages, otherwise continue until an empty page
    Returns a set of auction id strings.
    """
    ids = set()
    page = start_page
    pages_fetched = 0

    while True:
        params = {"pageNumber": page, "limit": limit}
        url = _merge_query(api_url, params)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # collect amAuctionId anywhere in the JSON
        _walk_for_amAuctionId(data, ids)

        pages_fetched += 1
        # determine whether there are more results:
        # - common API pattern: a list under 'sales' or 'items'
        count_on_page = 0
        if isinstance(data, dict):
            for cand in ("sales", "items", "results"):
                if cand in data and isinstance(data[cand], list):
                    count_on_page = len(data[cand])
                    break
        elif isinstance(data, list):
            count_on_page = len(data)

        # stop conditions
        if max_pages and pages_fetched >= max_pages:
            break
        if count_on_page == 0:
            break

        page += 1
        time.sleep(sleep)

    return ids


def save_ids_csv(ids: Set[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AUCTION_ID"])
        for aid in sorted(ids, key=lambda x: int(x)):
            writer.writerow([aid])


def save_ids_json(ids: Set[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(ids), "auction_ids": sorted(list(ids), key=lambda x: int(x))}, f, indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 maxsold_api_auction_ids.py <api-url> [out_prefix] [start_page] [limit] [max_pages]")
        sys.exit(1)

    api_url = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "maxsold_api_auctions"
    start_page = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    max_pages = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    ids = fetch_amAuctionIds_from_api(api_url, start_page=start_page, limit=limit, max_pages=max_pages)
    csv_path = f"{out_prefix}.csv"
    json_path = f"{out_prefix}.json"
    save_ids_csv(ids, csv_path)
    save_ids_json(ids, json_path)
    print(f"Saved {len(ids)} ids -> {csv_path}, {json_path}")
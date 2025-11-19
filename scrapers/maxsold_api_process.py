import os
import re
import csv
import json
from typing import Any, Dict, List, Optional
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

ID_KEYS = re.compile(r"amAuctionId", re.I)
TOTAL_BIDS_KEYS = re.compile(r"totalBids", re.I)
NUMBER_LOTS_KEYS = re.compile(r"numberLots", re.I)

# new keys
TITLE_KEYS = re.compile(r"title", re.I)
SALETYPE_KEYS = re.compile(r"saleType", re.I)
SALECATEGORY_KEYS = re.compile(r"saleCategory", re.I)
OPEN_TIME_KEYS = re.compile(r"openTime", re.I)
CLOSE_TIME_KEYS = re.compile(r"closeTime", re.I)


def fetch_json(api_url: str, timeout: int = 15) -> Any:
    r = requests.get(api_url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def find_value(obj: Any, key_re: re.Pattern) -> Optional[Any]:
    """
    Walk a dict/list and return the first value whose key matches key_re.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if key_re.search(str(k)):
                return v
        for v in obj.values():
            res = find_value(v, key_re)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_value(item, key_re)
            if res is not None:
                return res
    return None


def extract_from_sale(sale: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # amAuctionId is required; if missing skip
    am_id = find_value(sale, ID_KEYS)
    total_bids = find_value(sale, TOTAL_BIDS_KEYS)
    number_lots = find_value(sale, NUMBER_LOTS_KEYS)
    title = find_value(sale, TITLE_KEYS)
    sale_type = find_value(sale, SALETYPE_KEYS)
    sale_category = find_value(sale, SALECATEGORY_KEYS)
    open_time = find_value(sale, OPEN_TIME_KEYS)
    close_time = find_value(sale, CLOSE_TIME_KEYS)

    # normalize numeric-like values
    def to_int_or_none(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    def to_str_or_empty(v):
        return "" if v is None else str(v)

    return {
        "amAuctionId": str(am_id),
        "title": to_str_or_empty(title),
        "saleType": to_str_or_empty(sale_type),
        "saleCategory": to_str_or_empty(sale_category),
        "totalBids": to_int_or_none(total_bids) or 0,
        "numberLots": to_int_or_none(number_lots) or 0,
        "openTime": to_str_or_empty(open_time),
        "closeTime": to_str_or_empty(close_time),
    }


def process_api_to_csv(api_url: str, out_csv: str) -> int:
    data = fetch_json(api_url)
    # locate list of sales
    sales = None
    if isinstance(data, dict):
        for cand in ("sales", "items", "results", "data"):
            if cand in data and isinstance(data[cand], list):
                sales = data[cand]
                break
    if sales is None:
        if isinstance(data, list):
            sales = data
        else:
            # try to find any list of objects in the response
            def find_first_list(obj):
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        res = find_first_list(v)
                        if res is not None:
                            return res
                return None
            sales = find_first_list(data) or []

    rows = []
    for s in sales:
        if not isinstance(s, dict):
            continue
        rec = extract_from_sale(s)
        if rec:
            rows.append(rec)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["amAuctionId", "title", "saleType", "saleCategory", "totalBids", "numberLots", "openTime", "closeTime"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return len(rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 maxsold_api_process.py <api-url> [out.csv]")
        sys.exit(1)
    api_url = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "maxsold_api_auctions.csv"
    count = process_api_to_csv(api_url, out_csv)
    print(f"Wrote {count} rows -> {out_csv}")
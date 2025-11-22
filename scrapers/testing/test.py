import csv
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
import requests
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

API_TEMPLATE = "https://maxsold.maxsold.com/msapi/auctions/items?{query}"

OUT_DIR_DEFAULT = "/workspaces/maxsold/data/auction"
OUT_CSV_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "items_103293.csv")


def fetch_json_for_auction(auctionid: str, timeout: int = 15) -> Any:
    url = API_TEMPLATE.format(query=urlencode({"auctionid": auctionid}))
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def first_present(obj: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for k in candidates:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return ""


def to_float(v):
    try:
        return float(v)
    except Exception:
        return ""


def to_str(v):
    if v is None:
        return ""
    return str(v)


# ...existing code...
def extract_items_from_json(data: Any) -> List[Dict[str, Any]]:
    """
    Locate the auction object in the JSON and extract its items array,
    then normalize each item into the expected row shape.
    """
    auction = None
    # Find auction object in common locations
    if isinstance(data, dict):
        if isinstance(data.get("auction"), dict):
            auction = data["auction"]
        else:
            # common alternatives: 'auctions' (list), 'results', or nested 'data'
            for k in ("auctions", "results", "data"):
                v = data.get(k)
                if isinstance(v, dict):
                    auction = v
                    break
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    # take first auction if list provided
                    auction = v[0]
                    break

    # Items are expected inside the auction object
    items = None
    if isinstance(auction, dict):
        for key in ("items", "Items", "auctionItems", "itemsList", "results"):
            if key in auction and isinstance(auction[key], list):
                items = auction[key]
                break

    # Fallback: look for top-level items array
    if items is None and isinstance(data, dict):
        for key in ("items", "Items", "data", "results"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break

    if items is None:
        return []

    extracted = []
    for it in items:
        if not isinstance(it, dict):
            continue
        row = {
            "id": first_present(it, ["id", "itemId", "item_id", "ItemId", "Item_ID"]) or "",
            "auction_id": first_present(it, ["auctionId", "auction_id", "auctionid"]) or "",
            "title": to_str(first_present(it, ["title", "name"])),
            "viewed": to_int(first_present(it, ["viewed", "viewCount", "views"])),
            "minimum_bid": to_float(first_present(it, ["minimumBid", "minimum_bid", "minBid"])),
            "starting_bid": to_float(first_present(it, ["startingBid", "starting_bid", "startBid", "start_price"])),
            "current_bid": to_float(first_present(it, ["currentBid", "current_bid", "currentBidAmount", "bid"])),
            "proxy_bid": to_float(first_present(it, ["proxyBid", "proxy_bid"])),
            "start_time": to_str(first_present(it, ["start_time", "startTime", "starts"])),
            "end_time": to_str(first_present(it, ["end_time", "endTime", "ends"])),
            "lot_number": to_str(first_present(it, ["lotNumber", "lot_number", "lot"])),
            "bid_count": to_int(first_present(it, ["bidCount", "bids", "bid_count"])),
            "bidding_extended": first_present(it, ["biddingExtended", "bidding_extended", "extended"]) or False,
            "description": " ".join(to_str(first_present(it, ["description", "desc", "Details"])).split()),
        }
        extracted.append(row)
    return extracted


def write_items_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "id",
        "auction_id",
        "title",
        "viewed",
        "minimum_bid",
        "starting_bid",
        "current_bid",
        "proxy_bid",
        "start_time",
        "end_time",
        "lot_number",
        "bid_count",
        "bidding_extended",
        "description",

    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            # coerce values to serializable types/strings
            out = {k: (r[k] if r[k] is not None else "") for k in header}
            w.writerow(out)


def main(auctionid: str = "103293", out_csv: Optional[str] = None):
    out_csv = out_csv or OUT_CSV_DEFAULT
    try:
        data = fetch_json_for_auction(auctionid)
    except Exception as e:
        print(f"error fetching auction {auctionid}: {e}", file=sys.stderr)
        raise SystemExit(1)

    items = extract_items_from_json(data)
    if not items:
        print("no items found in API response", file=sys.stderr)
    write_items_csv(out_csv, items)
    print(f"wrote {len(items)} items to {out_csv}")


if __name__ == "__main__":
    import sys
    aid = sys.argv[1] if len(sys.argv) > 1 else "103293"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    main(aid, out)
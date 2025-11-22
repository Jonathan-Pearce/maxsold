import os
import sys
import csv
import requests
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

API_TEMPLATE = "https://maxsold.maxsold.com/msapi/auctions/items?{query}"


def fetch_json(auctionid: str, itemid: Optional[str] = None, timeout: int = 15) -> Any:
    q = {"auctionid": auctionid}
    if itemid:
        q["itemid"] = itemid
    url = API_TEMPLATE.format(query=urlencode(q))
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def locate_auction(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("auction"), dict):
            return data["auction"]
        for k in ("auctions", "results", "data"):
            v = data.get(k)
            if isinstance(v, dict):
                return v
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[0]
        # fallback: data itself may be the auction
        if any(key in data for key in ("items", "id", "auctionId", "title")):
            return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def extract_bid_entries(item_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract bids from item_obj -> bid_history -> [0] (if present).
    Normalizes keys to time_of_bid, amount, isproxy.
    """
    bids_container = None
    # find bid_history any key name variants
    for k in ("bid_history", "bidHistory", "bids", "bid_history_list", "bidHistoryList"):
        if k in item_obj and item_obj[k] is not None:
            bids_container = item_obj[k]
            break
    if bids_container is None:
        return []

    # many API responses nest bid rounds: bid_history might be [[{...}, ...], ...]
    # we target the first nested list (index 0) if that's the structure
    if isinstance(bids_container, list) and bids_container:
        first_lvl = bids_container[0]
        # if first_lvl is list, use it; else if it's dict list, treat bids_container as list of dicts
        if isinstance(first_lvl, list):
            bid_list = first_lvl
        elif isinstance(first_lvl, dict):
            # bids_container itself is list of dicts
            bid_list = bids_container
        else:
            # unknown shape
            bid_list = []
    else:
        bid_list = []

    out = []
    for b in bid_list:
        if not isinstance(b, dict):
            continue
        time_of_bid = first_present(b, ["time_of_bid", "timeOfBid", "time", "bidTime", "createdAt"]) or ""
        amount = first_present(b, ["amount", "bidAmount", "value", "currentBid"]) or ""
        isproxy = first_present(b, ["isproxy", "isProxy", "proxy", "is_proxy", "isProxyBid"]) or False
        out.append({
            "time_of_bid": str(time_of_bid),
            "amount": str(amount),
            "isproxy": bool(isproxy),
        })
    return out


def write_csv(path: str, rows: List[Dict[str, Any]], header: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main(auctionid: str, itemid: str, out_csv: Optional[str] = None):
    out_csv = out_csv or f"/workspaces/maxsold/data/auction/bid_history_{auctionid}_{itemid}.csv"
    data = fetch_json(auctionid, itemid)
    auction = locate_auction(data)
    if auction is None:
        print("No auction object found in response", file=sys.stderr)
        return

    # find items array inside auction
    items = None
    for k in ("items", "Items", "auctionItems", "itemsList", "results"):
        if k in auction and isinstance(auction[k], list):
            items = auction[k]
            break
    if items is None:
        # try top-level
        for k in ("items", "Items", "data"):
            if k in data and isinstance(data[k], list):
                items = data[k]
                break

    if not items:
        print("No items array found in auction", file=sys.stderr)
        return

    # target first item (index 0)
    item0 = items[0] if len(items) > 0 else None
    if item0 is None:
        print("Item[0] not present", file=sys.stderr)
        return

    bids = extract_bid_entries(item0)
    if not bids:
        print("No bids found in item[0].bid_history[0]", file=sys.stderr)
        return

    # attach auction_id and item_id to each row
    rows = []
    for b in bids:
        rows.append({
            "auction_id": auctionid,
            "item_id": itemid,
            "time_of_bid": b["time_of_bid"],
            "amount": b["amount"],
            "isproxy": b["isproxy"],
        })

    header = ["auction_id", "item_id", "time_of_bid", "amount", "isproxy"]
    write_csv(out_csv, rows, header)
    print(f"Wrote {len(rows)} bid rows to {out_csv}")


if __name__ == "__main__":
    aid = sys.argv[1] if len(sys.argv) > 1 else "103293"
    iid = sys.argv[2] if len(sys.argv) > 2 else "7433850"
    out = sys.argv[3] if len(sys.argv) > 3 else None
    main(aid, iid, out)
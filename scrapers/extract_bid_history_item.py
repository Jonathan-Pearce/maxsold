import os
import sys
import csv
import time
import requests
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from pathlib import Path

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
        if any(key in data for key in ("items", "id", "auctionId", "title")):
            return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def find_item_by_id(items: List[Dict[str, Any]], itemid: str) -> Optional[Dict[str, Any]]:
    sid = str(itemid)
    for it in items:
        if not isinstance(it, dict):
            continue
        cand = first_present(it, ["id", "itemId", "item_id", "ItemId", "Item_ID"])
        if cand is None:
            continue
        if str(cand) == sid:
            return it
    return None


def extract_bid_entries(item_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract bids from item_obj -> bid_history (or variants).
    Normalizes keys to time_of_bid, amount, isproxy.
    """
    bids_container = None
    for k in ("bid_history", "bidHistory", "bids", "bid_history_list", "bidHistoryList"):
        if k in item_obj and item_obj[k] is not None:
            bids_container = item_obj[k]
            break
    if bids_container is None:
        return []

    # bids_container may be nested lists: [[{...}, ...], ...]
    if isinstance(bids_container, list) and bids_container:
        first_lvl = bids_container[0]
        if isinstance(first_lvl, list):
            bid_list = first_lvl
        elif isinstance(first_lvl, dict):
            # list of dicts
            bid_list = bids_container
        else:
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


def read_items_csv(path: str) -> List[Dict[str, str]]:
    """
    Read CSV that contains at least item id and auction id.
    Tries common header names, falls back to positional columns.
    Returns list of dicts with keys 'auction_id' and 'item_id'.
    """
    out = []
    if not os.path.exists(path):
        print(f"file not found: {path}", file=sys.stderr)
        return out
    with open(path, newline="", encoding="utf-8") as f:
        # try DictReader first
        rdr = csv.DictReader(f)
        headers = [h.lower() for h in (rdr.fieldnames or [])]
        if headers:
            # common header names
            aid_keys = [k for k in headers if k in ("auction_id", "auctionid", "auctionid".lower(), "auction")]
            iid_keys = [k for k in headers if k in ("id", "item_id", "itemid", "item")]
            aid_key = aid_keys[0] if aid_keys else None
            iid_key = iid_keys[0] if iid_keys else None
            if aid_key and iid_key:
                for r in rdr:
                    aid = r.get(aid_key, "").strip()
                    iid = r.get(iid_key, "").strip()
                    if aid and iid:
                        out.append({"auction_id": aid, "item_id": iid})
                return out
        # fallback: read as simple CSV (no reliable headers)
        f.seek(0)
        rdr2 = csv.reader(f)
        for row in rdr2:
            if not row:
                continue
            # assume first column is item id and second is auction id OR vice-versa
            if len(row) == 1:
                # single column: could be item id only; skip
                continue
            # Heuristic: if first looks like auction id present in sales CSV (numeric) and second numeric, we try both orders.
            a, b = row[0].strip(), row[1].strip()
            # prefer treating first as auction id if it matches common lengths (>5 digits)
            if len(a) >= 5 and len(b) <= 7:
                out.append({"auction_id": a, "item_id": b})
            else:
                out.append({"auction_id": b, "item_id": a})
    return out


def main(items_csv: Optional[str] = None, out_csv: Optional[str] = None, delay: float = 0.3):
    items_csv = items_csv or "/workspaces/maxsold/data/auction/items_all_2025-11-21.csv"
    out_csv = out_csv or "/workspaces/maxsold/data/auction/bid_history_all_2025-11-21.csv"

    rows = read_items_csv(items_csv)
    if not rows:
        print("no items to process", file=sys.stderr)
        return

    # include bid_number column
    header = ["auction_id", "item_id", "bid_number", "time_of_bid", "amount", "isproxy"]
    total_written = 0
    for i, rec in enumerate(rows, 1):
        auction_id = rec.get("auction_id")
        item_id = rec.get("item_id")
        if not auction_id or not item_id:
            continue
        print(f"[{i}/{len(rows)}] fetching bids for auction={auction_id} item={item_id}...")
        try:
            data = fetch_json(auction_id, item_id)
        except Exception as e:
            print(f"  fetch error for {auction_id}/{item_id}: {e}", file=sys.stderr)
            time.sleep(delay)
            continue

        auction = locate_auction(data)
        # find items array
        items = None
        if isinstance(auction, dict):
            for k in ("items", "Items", "auctionItems", "itemsList", "results"):
                if k in auction and isinstance(auction[k], list):
                    items = auction[k]
                    break
        if items is None and isinstance(data, dict):
            for k in ("items", "Items", "data", "results"):
                if k in data and isinstance(data[k], list):
                    items = data[k]
                    break

        target_item = None
        if items:
            target_item = find_item_by_id(items, item_id)
        else:
            # sometimes API returns single item dict instead of list
            if isinstance(auction, dict):
                # try treating auction as item wrapper
                if str(first_present(auction, ["id", "itemId", "item_id"]) or "") == str(item_id):
                    target_item = auction

        if not target_item:
            print(f"  item {item_id} not found in response", file=sys.stderr)
            time.sleep(delay)
            continue

        bids = extract_bid_entries(target_item)
        if not bids:
            print(f"  no bids for {auction_id}/{item_id}")
            time.sleep(delay)
            continue

        out_rows = []
        # add bid_number sequentially per item (1 = first in bid_list)
        for bn, b in enumerate(bids, start=1):
            out_rows.append({
                "auction_id": auction_id,
                "item_id": item_id,
                "bid_number": bn,
                "time_of_bid": b["time_of_bid"],
                "amount": b["amount"],
                "isproxy": b["isproxy"],
            })

        write_csv(out_csv, out_rows, header)
        total_written += len(out_rows)
        print(f"  wrote {len(out_rows)} bids for {auction_id}/{item_id}")
        time.sleep(delay)

    print(f"done. total bid rows written: {total_written} -> {out_csv}")


if __name__ == "__main__":
    items_csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_csv_arg = sys.argv[2] if len(sys.argv) > 2 else None
    delay_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    main(items_csv_arg, out_csv_arg, delay_arg)
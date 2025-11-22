import requests
import csv
import sys
import time
import os
from pathlib import Path

URL = "https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auctionid}"
OUT_CSV = "/workspaces/maxsold/data/auction/clean/auctions_items_summary_2025-11-21.csv"

def get_auction_from_response(data):
    # Try common locations for the auction object
    if isinstance(data, dict):
        if 'auction' in data and isinstance(data['auction'], dict):
            return data['auction']
        # sometimes the payload might put data at top-level
        possible_keys = ['auctions', 'results', 'data']
        for k in possible_keys:
            if k in data and isinstance(data[k], list) and data[k]:
                return data[k][0]
        return data
    return None

def extract_item_id(item):
    # try a few common key names and coerce to int if possible
    if not isinstance(item, dict):
        return None
    for key in ('id', 'itemId', 'ItemId', 'item_id', 'Item_ID'):
        if key in item and item[key] is not None:
            try:
                return int(item[key])
            except (ValueError, TypeError):
                try:
                    return int(float(item[key]))
                except Exception:
                    return None
    return None

def fetch_json(url, timeout=15):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def process_auction(auctionid):
    api_url = URL.format(auctionid=auctionid)
    try:
        data = fetch_json(api_url)
    except Exception as e:
        print(f"error fetching {auctionid}: {e}", file=sys.stderr)
        return None

    auction = get_auction_from_response(data)
    if auction is None:
        print(f"no auction object for {auctionid}", file=sys.stderr)
        return None

    auction_id = auction.get('id') or auction.get('auctionid') or auction.get('auctionId') or auctionid
    title = auction.get('title')
    starts = auction.get('starts')
    ends = auction.get('ends')
    last_item_closes = auction.get('last_item_closes') or auction.get('lastItemCloses')
    pickup_time = auction.get('pickup_time') or auction.get('pickupTime')

    # locate items list
    items = None
    for key in ('items', 'Items', 'auctionItems', 'itemsList'):
        if key in auction and isinstance(auction[key], list):
            items = auction[key]
            break
    if items is None and isinstance(data, dict):
        for key in ('items', 'Items'):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break

    min_id = None
    max_id = None
    if items:
        ids = [extract_item_id(it) for it in items]
        ids = [i for i in ids if i is not None]
        if ids:
            min_id = min(ids)
            max_id = max(ids)

    return {
        "auction_id": auction_id,
        "title": title,
        "starts": starts,
        "ends": ends,
        "last_item_closes": last_item_closes,
        "pickup_time": pickup_time,
        "min_item_id": min_id,
        "max_item_id": max_id,
        "source_url": api_url
    }

def read_auction_ids_from_sales(csv_path):
    ids = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                val = r.get("amAuctionId") or r.get("amAuctionID") or r.get("auction_id") or r.get("amAuctionId".lower())
                if not val:
                    # maybe CSV has no header; take first column
                    keys = list(r.keys())
                    if keys:
                        val = r[keys[0]]
                if val:
                    try:
                        ids.append(str(int(float(val))))
                    except Exception:
                        ids.append(val)
    except FileNotFoundError:
        print(f"sales CSV not found: {csv_path}", file=sys.stderr)
    return sorted(set(ids), key=lambda x: int(x) if x.isdigit() else x)

def write_rows(output_path, rows, header):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_header = not Path(output_path).exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

def main(sales_csv=None, out_csv=None, delay=0.5):
    sales_csv = sales_csv or "/workspaces/maxsold/data/auction/clean/sales_2025-11-21.csv"
    out_csv = out_csv or OUT_CSV

    auction_ids = read_auction_ids_from_sales(sales_csv)
    if not auction_ids:
        print("no auction ids found; exiting", file=sys.stderr)
        return

    header = ['auction_id','title','starts','ends','last_item_closes','pickup_time','min_item_id','max_item_id','source_url']
    rows_to_write = []

    for i, aid in enumerate(auction_ids, 1):
        print(f"[{i}/{len(auction_ids)}] processing auction {aid}...")
        info = process_auction(aid)
        if info:
            # ensure CSV-friendly values (None -> empty string)
            for k in header:
                if k not in info:
                    info[k] = ""
                elif info[k] is None:
                    info[k] = ""
            rows_to_write.append(info)
            # write incrementally to avoid data loss on long runs
            write_rows(out_csv, [info], header)
            print(f"  saved auction {aid}")
        else:
            print(f"  skipped auction {aid} (error or no data)", file=sys.stderr)
        time.sleep(delay)

    print(f"done. results appended to {out_csv}")

if __name__ == "__main__":
    # usage:
    # python3 scrapers/scrape_auction.py [sales_csv] [out_csv]
    sales_csv = sys.argv[1] if len(sys.argv) > 1 else None
    out_csv = sys.argv[2] if len(sys.argv) > 2 else None
    main(sales_csv, out_csv)
import requests
import csv
import sys

URL = "https://maxsold.maxsold.com/msapi/auctions/items?auctionid=103293"
OUT_CSV = "auction_summary.csv"

def get_auction_from_response(data):
    # Try common locations for the auction object
    if isinstance(data, dict):
        if 'auction' in data and isinstance(data['auction'], dict):
            return data['auction']
        # sometimes the payload might put data at top-level
        possible_keys = ['auctions', 'results']
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
                # maybe it's numeric string with decimals etc.
                try:
                    return int(float(item[key]))
                except Exception:
                    return None
    return None

def main(url=URL, out_csv=OUT_CSV):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching or parsing JSON from {url}: {e}", file=sys.stderr)
        sys.exit(1)

    auction = get_auction_from_response(data)
    if auction is None:
        print("Couldn't find auction object in JSON response.", file=sys.stderr)
        sys.exit(1)

    # Extract requested auction fields (use .get so missing keys become None)
    auction_id = auction.get('id') or auction.get('auctionid') or auction.get('auctionId')
    title = auction.get('title')
    starts = auction.get('starts')
    ends = auction.get('ends')
    last_item_closes = auction.get('last_item_closes')
    pickup_time = auction.get('pickup_time')

    # Find items list in several possible locations
    items = None
    for key in ('items', 'Items', 'auctionItems', 'itemsList'):
        if key in auction and isinstance(auction[key], list):
            items = auction[key]
            break
    # fallback to top-level items if not inside auction
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

    # Write CSV (single-row)
    header = ['auction_id', 'title', 'starts', 'ends', 'last_item_closes', 'pickup_time', 'min_item_id', 'max_item_id', 'source_url']
    row = [auction_id, title, starts, ends, last_item_closes, pickup_time, min_id, max_id, url]

    try:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
        print(f"Wrote results to {out_csv}")
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
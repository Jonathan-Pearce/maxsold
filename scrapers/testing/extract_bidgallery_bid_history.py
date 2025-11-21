import re
import csv
import os
import time
import random
from typing import List, Tuple, Optional
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def fetch_html(url: str, timeout: int = 12) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def find_bid_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    # look for an explicit "Bid History" heading then the following table
    heading = soup.find(lambda t: t.name in ("h1","h2","h3","h4","h5","strong","p","span") and re.search(r"bid history", t.get_text("", strip=True), re.I))
    if heading:
        tbl = heading.find_next("table")
        if tbl:
            return tbl
    # fallback: find first table whose headers mention date and amount/bid/price
    for tbl in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
        if headers:
            if any("date" in h for h in headers) and any(("amount" in h or "bid" in h or "price" in h) for h in headers):
                return tbl
        # sometimes the table has no th but contains date-like text and $ amounts
        text = tbl.get_text(" ", strip=True)
        if re.search(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", text) and re.search(r"\$\s*\d", text):
            return tbl
    return None

def parse_table_rows(tbl: BeautifulSoup) -> List[Tuple[str, str]]:
    rows = []
    # build header mapping if present
    ths = [th.get_text(" ", strip=True).lower() for th in tbl.find_all("th")]
    has_header = bool(ths)
    idx_date = idx_amount = None
    if has_header:
        for i, h in enumerate(ths):
            if "date" in h:
                idx_date = i
            if ("amount" in h) or ("bid" in h) or ("price" in h) or ("amount" in h):
                idx_amount = i
    # iterate tr elements
    for tr in tbl.find_all("tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
        if not tds:
            continue
        # skip header row if it matches header
        if has_header and [x.lower() for x in tds] == ths:
            continue
        if idx_date is not None and idx_amount is not None:
            if idx_date < len(tds) and idx_amount < len(tds):
                date = tds[idx_date]
                amount = tds[idx_amount]
                rows.append((date, normalize_amount(amount)))
            continue
        # fallback heuristics:
        #  - if two columns, assume first is date, second is amount
        #  - if >=2 columns, try first column date-like and any column containing $ as amount
        if len(tds) == 2:
            rows.append((tds[0], normalize_amount(tds[1])))
            continue
        # find a date-like and an amount-like cell
        date_cell = None
        amount_cell = None
        for cell in tds:
            if not date_cell and re.search(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}", cell):
                date_cell = cell
        for cell in tds:
            if not amount_cell and re.search(r"\$\s*[\d,]+(?:\.\d{1,2})?", cell):
                amount_cell = cell
        if date_cell and amount_cell:
            rows.append((date_cell, normalize_amount(amount_cell)))
            continue
        # last resort: take first and last columns
        if len(tds) >= 2:
            rows.append((tds[0], normalize_amount(tds[-1])))
    return rows

def normalize_amount(s: str) -> str:
    # return amount as a cleaned string like "123.45" prefixed by $ if present
    if s is None:
        return ""
    s = s.strip()
    m = re.search(r"(-?\$?\s*[\d,]+(?:\.\d{1,2})?)", s)
    if not m:
        # try to extract numbers
        m2 = re.search(r"(-?\d[\d,]*(?:\.\d{1,2})?)", s)
        val = m2.group(1) if m2 else s
    else:
        val = m.group(1)
    val = val.replace(" ", "").replace(",", "")
    # ensure leading $
    if val and not val.startswith("$") and re.match(r"-?\d", val):
        val = "$" + val
    return val

def scrape_bid_history_from_url(url: str) -> List[Tuple[str, str]]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    tbl = find_bid_table(soup)
    if not tbl:
        return []
    rows = parse_table_rows(tbl)
    return rows

def read_links_csv(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            lid = r.get("listing_id") or r.get("id") or ""
            url = r.get("url") or ""
            if url:
                out.append((lid, url))
    return out

def append_bid_rows(out_csv: str, rows: List[Tuple[str, str, str]]) -> None:
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["listing_id", "Bid Date", "Bid Amount"])
        for r in rows:
            w.writerow(r)

def main(links_csv: str = "auction_103482_links.csv", out_csv: str = "auction_103482_bid_history.csv"):
    links = read_links_csv(links_csv)
    for listing_id, url in links:
        try:
            bid_rows = scrape_bid_history_from_url(url)
        except Exception as e:
            print(f"error fetching {url}: {e}")
            bid_rows = []
        formatted = [(listing_id, date, amount) for date, amount in bid_rows]
        if formatted:
            append_bid_rows(out_csv, formatted)
            print(f"saved {len(formatted)} bids for {listing_id}")
        else:
            # write no rows (optional) -- skipping to keep CSV compact
            print(f"no bid history found for {listing_id}")
        # polite delay
        time.sleep(random.uniform(0.8, 1.6))

if __name__ == "__main__":
    import sys
    links_csv = sys.argv[1] if len(sys.argv) > 1 else "auction_103482_links.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "auction_103482_bid_history.csv"
    main(links_csv, out_csv)
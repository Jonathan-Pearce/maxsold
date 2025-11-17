import re
import json
from typing import Optional, Dict, List

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def text_after_label(soup: BeautifulSoup, label_regex: str) -> Optional[str]:
    label = soup.find(lambda t: t.string and re.search(label_regex, t.get_text(strip=True), re.I))
    if not label:
        return None
    # look for sibling or next element with text
    for attr in ("next_sibling", "find_next"):
        try:
            if attr == "next_sibling":
                nxt = label.next_sibling
            else:
                nxt = label.find_next(text=True)
            if nxt and isinstance(nxt, str):
                txt = nxt.strip()
                if txt:
                    return txt
            # try element node
            nxt_el = label.find_next()
            if nxt_el:
                txt = nxt_el.get_text(" ", strip=True)
                if txt:
                    return txt
        except Exception:
            continue
    return None

def extract_location(soup: BeautifulSoup) -> Optional[str]:
    # common patterns
    candidates = []
    # look for explicit labels
    for lab in (r"\blocation\b", r"\baddress\b", r"\bvenue\b"):
        v = text_after_label(soup, lab)
        if v:
            candidates.append(v)
    # search for element with class/id containing 'location' or 'address'
    el = soup.find(attrs={"class": re.compile(r"(location|address|venue)", re.I)})
    if el:
        candidates.append(el.get_text(" ", strip=True))
    el = soup.find(id=re.compile(r"(location|address|venue)", re.I))
    if el:
        candidates.append(el.get_text(" ", strip=True))
    # return first non-empty candidate
    for c in candidates:
        if c and not re.match(r"^\s*$", c):
            return c
    return None

def extract_date_times(soup: BeautifulSoup) -> List[Dict[str,str]]:
    results: List[Dict[str,str]] = []
    # common structure: rows with label and value
    rows = soup.find_all(lambda t: t.name in ("div","li","tr") and t.get_text(" ", strip=True) and re.search(r"(date|time|starts|ends|bidding)", t.get_text(" ", strip=True), re.I))
    seen = set()
    for r in rows:
        txt = r.get_text(" ", strip=True)
        # simple split on colon or hyphen
        parts = [p.strip() for p in re.split(r"\s*:\s*|\s{2,}|\s*-\s*", txt) if p.strip()]
        if len(parts) >= 2:
            label = parts[0]
            value = " : ".join(parts[1:])
            key = f"{label}|{value}"
            if key not in seen:
                seen.add(key)
                results.append({"label": label, "value": value})
    # fallback: look for headings that contain date/time text and grab following text
    if not results:
        for h in soup.find_all(re.compile("^h[1-6]$")):
            if re.search(r"(date|time|starts|ends|bidding)", h.get_text(" ", strip=True), re.I):
                nxt = h.find_next(string=True)
                if nxt:
                    v = nxt.strip()
                    if v:
                        results.append({"label": h.get_text(" ", strip=True), "value": v})
    return results

def extract_lot_count_from_bidgallery(soup: BeautifulSoup) -> Optional[int]:
    # Try explicit text like "X lots" or "X items"
    txt = soup.get_text(" ", strip=True)
    m = re.search(r"(\d{1,6})\s+(?:lots|items|listings)\b", txt, re.I)
    if m:
        return int(m.group(1))
    # Count likely lot cards: links to /lot/ or elements with 'lot' class
    links = soup.find_all("a", href=re.compile(r"/lot/|/listing/"), limit=10000)
    if links:
        # de-duplicate by href
        hrefs = {a.get("href") for a in links if a.get("href")}
        return len(hrefs)
    cards = soup.find_all(attrs={"class": re.compile(r"(lot|listing|gallery-item|card)", re.I)})
    if cards:
        return len(cards)
    return None

def fetch_url(url: str, timeout: int = 12) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def fetch_auction_metadata(auction_id: str) -> Dict:
    base = f"https://maxsold.com/auction/{auction_id}"
    out: Dict = {"auction_id": auction_id, "source": base}
    # date-times page
    try:
        dt_soup = fetch_url(f"{base}/date-times")
        out["location"] = extract_location(dt_soup)
        out["date_times"] = extract_date_times(dt_soup)
    except Exception as e:
        out["date_times_error"] = str(e)
    # bidgallery page -> lots count
    try:
        bg_soup = fetch_url(f"{base}/bidgallery")
        out["lots_count"] = extract_lot_count_from_bidgallery(bg_soup)
    except Exception as e:
        out["lots_error"] = str(e)
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 maxsold_auction_metadata.py <auction-id-or-url>")
        sys.exit(1)
    arg = sys.argv[1].strip()
    # accept either numeric id or full url
    m = re.search(r"/auction/(\d+)", arg)
    auction_id = m.group(1) if m else (arg if arg.isdigit() else arg)
    data = fetch_auction_metadata(auction_id)
    print(json.dumps(data, indent=2))
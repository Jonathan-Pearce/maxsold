import re
import json
import os
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
    nxt_el = label.find_next()
    if nxt_el:
        txt = nxt_el.get_text(" ", strip=True)
        if txt:
            return txt
    return None

def extract_location(soup: BeautifulSoup) -> Optional[str]:
    for lab in (r"\blocation\b", r"\baddress\b", r"\bvenue\b"):
        v = text_after_label(soup, lab)
        if v:
            return v
    el = soup.find(attrs={"class": re.compile(r"(location|address|venue)", re.I)})
    if el:
        return el.get_text(" ", strip=True)
    el = soup.find(id=re.compile(r"(location|address|venue)", re.I))
    if el:
        return el.get_text(" ", strip=True)
    return None

def extract_date_times(soup: BeautifulSoup) -> List[Dict[str,str]]:
    results: List[Dict[str,str]] = []
    rows = soup.find_all(lambda t: t.name in ("div","li","tr") and t.get_text(" ", strip=True) and re.search(r"(date|time|starts|ends|bidding)", t.get_text(" ", strip=True), re.I))
    seen = set()
    for r in rows:
        txt = r.get_text(" ", strip=True)
        parts = [p.strip() for p in re.split(r"\s*:\s*|\s{2,}|\s*-\s*", txt) if p.strip()]
        if len(parts) >= 2:
            label = parts[0]
            value = " : ".join(parts[1:])
            key = f"{label}|{value}"
            if key not in seen:
                seen.add(key)
                results.append({"label": label, "value": value})
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
    txt = soup.get_text(" ", strip=True)
    m = re.search(r"(\d{1,6})\s+(?:lots|items|listings)\b", txt, re.I)
    if m:
        return int(m.group(1))
    links = soup.find_all("a", href=re.compile(r"/lot/|/listing/"), limit=10000)
    if links:
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
    try:
        dt_soup = fetch_url(f"{base}/date-times")
        out["location"] = extract_location(dt_soup)
        out["date_times"] = extract_date_times(dt_soup)
    except Exception as e:
        out["date_times_error"] = str(e)
    try:
        bg_soup = fetch_url(f"{base}/bidgallery")
        out["lots_count"] = extract_lot_count_from_bidgallery(bg_soup)
    except Exception as e:
        out["lots_error"] = str(e)
    return out

def save_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 maxsold_auction_metadata.py <auction-id-or-url> [out.json]")
        sys.exit(1)
    arg = sys.argv[1].strip()
    m = re.search(r"/auction/(\d+)", arg)
    auction_id = m.group(1) if m else (arg if arg.isdigit() else arg)
    out_path = sys.argv[2].strip() if len(sys.argv) > 2 else f"auction_{auction_id}.json"
    data = fetch_auction_metadata(auction_id)
    try:
        save_json(data, out_path)
        print(out_path)
    except Exception as e:
        print("Save error:", e)
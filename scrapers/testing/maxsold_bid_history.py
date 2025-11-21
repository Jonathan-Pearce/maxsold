import re
import json
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

# Optional selenium imports (only used if render_js=True)
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def parse_bid_rows_from_table(table) -> List[Dict]:
    rows = []
    thead = table.find("thead")
    headers = []
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    for tr in table.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if not cols or all(not c for c in cols):
            continue
        if headers and len(cols) == len(headers):
            rows.append(dict(zip(headers, cols)))
        else:
            # generic columns as column0, column1...
            row = {f"col{i}": v for i, v in enumerate(cols)}
            rows.append(row)
    return rows

def extract_bid_history_from_soup(soup: BeautifulSoup) -> Optional[List[Dict]]:
    # 1) Find heading that contains "Bid History"
    heading = soup.find(lambda t: t.name in ("h1","h2","h3","h4","h5","h6") and "bid history" in t.get_text(strip=True).lower())
    if heading:
        tbl = heading.find_next("table")
        if tbl:
            return parse_bid_rows_from_table(tbl)
        # sometimes rendered as div rows after heading
        container = heading.find_next(lambda t: t.name in ("div","section") and t.find_all(["div","li"]))
        if container:
            # try to extract structured rows
            rows = []
            for row in container.find_all(recursive=False):
                text = row.get_text(" ", strip=True)
                if text:
                    rows.append({"text": text})
            if rows:
                return rows

    # 2) Look for a table with a likely class or id
    for key in ("bid-history", "bid_history", "bidhistory"):
        tbl = soup.find("table", attrs={"class": re.compile(key, re.I)}) or soup.find("table", id=re.compile(key, re.I))
        if tbl:
            return parse_bid_rows_from_table(tbl)

    # 3) Look for script tags that include JSON payload with "bid" or "bidHistory"
    for s in soup.find_all("script"):
        if not s.string:
            continue
        txt = s.string
        if "bidHistory" in txt or "\"bids\"" in txt or "bidhistory" in txt.lower():
            # attempt to find JSON substring
            try:
                json_texts = re.findall(r"(\{.*\"bid.*\}|\[.*\"bid.*\])", txt, flags=re.S)
                for jt in json_texts:
                    try:
                        data = json.loads(jt)
                        # search for lists of bids
                        def walk(o):
                            if isinstance(o, dict):
                                for k, v in o.items():
                                    if k and "bid" in k.lower() and isinstance(v, (list, dict)):
                                        return v
                                    res = walk(v)
                                    if res:
                                        return res
                            elif isinstance(o, list):
                                for x in o:
                                    res = walk(x)
                                    if res:
                                        return res
                            return None
                        bids = walk(data)
                        if bids:
                            return bids if isinstance(bids, list) else [bids]
                    except Exception:
                        continue
            except Exception:
                continue

    # 4) fallback: look for any table that has "Bid" in header text
    for table in soup.find_all("table"):
        if any("bid" in (th.get_text(strip=True).lower()) for th in table.find_all("th")):
            return parse_bid_rows_from_table(table)

    return None

def fetch_static(url: str, timeout: int = 15) -> Optional[List[Dict]]:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return extract_bid_history_from_soup(soup)

# Example Selenium fallback (uncomment imports above and pip-install selenium + chromedriver)
# def fetch_with_selenium(url: str, timeout: int = 20) -> Optional[List[Dict]]:
#     chrome_options = Options()
#     chrome_options.add_argument("--headless=new")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     driver = webdriver.Chrome(options=chrome_options)
#     try:
#         driver.get(url)
#         WebDriverWait(driver, timeout).until(
#             lambda d: "Bid" in d.title or d.find_elements(By.TAG_NAME, "table")
#         )
#         html = driver.page_source
#         soup = BeautifulSoup(html, "html.parser")
#         return extract_bid_history_from_soup(soup)
#     finally:
#         driver.quit()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python maxsold_bid_history.py <listing-url>")
        sys.exit(1)
    url = sys.argv[1]
    try:
        bids = fetch_static(url)
    except Exception as e:
        print("Static fetch failed:", e)
        bids = None
    if not bids:
        print("Attempting JS-rendered fetch with Selenium (if enabled)...")
        # bids = fetch_with_selenium(url)   # uncomment to use selenium fallback
    if not bids:
        print("No bid history found.")
    else:
        print(json.dumps(bids, indent=2))
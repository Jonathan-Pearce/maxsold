# Basic usage (default parameters)
python3 scrapers/01_extract_auction_search.py

# Custom output path
python3 scrapers/01_extract_auction_search.py -o /workspaces/maxsold/data/auction/sales_closed_90days.parquet

# Different location (Vancouver)
python3 scrapers/01_extract_auction_search.py --lat 49.2827 --lng -123.1207 --radius 100000

# Limit to first 5 pages for testing
python3 scrapers/01_extract_auction_search.py --max-pages 5

# Different time range
python3 scrapers/01_extract_auction_search.py --days 30
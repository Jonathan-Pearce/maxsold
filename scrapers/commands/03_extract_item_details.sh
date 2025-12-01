# Single auction
python3 scrapers/03_extract_items_details.py 103293

# Multiple auctions
python3 scrapers/03_extract_items_details.py 103293 103294 103295

# Default: Load from auction search parquet file
python3 scrapers/03_extract_items_details.py

# Specify custom parquet file
python3 scrapers/03_extract_items_details.py -p /workspaces/maxsold/data/auction_search/auction_search_20251201.parquet

# With custom output path
python3 scrapers/03_extract_items_details.py -p /workspaces/maxsold/data/auction_search/auction_search_20251201.parquet -o /workspaces/maxsold/data/auction/items_batch.parquet

# Read auction IDs from text file
echo -e "103293\n103294" > /tmp/auction_ids.txt
python3 scrapers/03_extract_items_details.py -i /tmp/auction_ids.txt
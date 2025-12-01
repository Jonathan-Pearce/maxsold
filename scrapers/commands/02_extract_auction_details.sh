# Default: Load from /workspaces/maxsold/data/auction_search/auction_search_20251201.parquet
python3 scrapers/02_extract_auction_details.py

# Specify custom parquet file
python3 scrapers/02_extract_auction_details.py -p /workspaces/maxsold/data/auction_search/auction_search_20251201.parquet

# With custom output path
python3 scrapers/02_extract_auction_details.py -p /workspaces/maxsold/data/auction_search/auction_search_20251201.parquet -o /workspaces/maxsold/data/auction/auctions_batch.parquet

# Still supports manual auction IDs
python3 scrapers/02_extract_auction_details.py 103293 103294

# Still supports text file input
python3 scrapers/02_extract_auction_details.py -i /tmp/auction_ids.txt
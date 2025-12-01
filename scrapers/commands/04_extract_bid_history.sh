# Default: 10 parallel workers
python3 scrapers/04_extract_bid_history.py

# Use 20 workers for faster processing
python3 scrapers/04_extract_bid_history.py -w 20

# Test with first 100 items only
python3 scrapers/04_extract_bid_history.py -l 100

# Sequential mode for debugging
python3 scrapers/04_extract_bid_history.py --sequential -l 10

# Custom input and output
python3 scrapers/04_extract_bid_history.py -p /workspaces/maxsold/data/item_details/items_details_20251201.parquet -o /tmp/bids.parquet -w 15

# Process all items with 25 workers
python3 scrapers/04_extract_bid_history.py -w 25
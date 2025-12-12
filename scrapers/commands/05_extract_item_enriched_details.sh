# From parquet file (default)
python3 /workspaces/maxsold/scrapers/05_extract_item_enriched_details.py

# With specific item IDs
python3 /workspaces/maxsold/scrapers/05_extract_item_enriched_details.py 7433915 7433916

# With options
python3 /workspaces/maxsold/scrapers/05_extract_item_enriched_details.py -p data/item_details/items_details_20251201.parquet -w 20 -l 100
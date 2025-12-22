# MaxSold Scrapers - Refactored Structure

This directory contains refactored scraping code for the MaxSold auction platform. The code has been reorganized to separate data extraction logic from file I/O operations, making it reusable for both batch scraping pipelines and live ML model predictions.

## Directory Structure

```
scrapers/
├── extractors/              # Core extraction logic (reusable)
│   ├── __init__.py
│   ├── auction_search.py   # Fetch & extract auction search results
│   ├── auction_details.py  # Fetch & extract auction details
│   ├── item_details.py     # Fetch & extract item/lot details
│   ├── bid_history.py      # Fetch & extract bid history
│   └── item_enriched.py    # Fetch & extract AI-enriched item data
│
├── pipelines/               # Batch scraping pipelines with file I/O
│   ├── __init__.py
│   ├── auction_search_pipeline.py
│   ├── auction_details_pipeline.py
│   ├── item_details_pipeline.py
│   ├── bid_history_pipeline.py
│   └── item_enriched_pipeline.py
│
├── utils/                   # Shared utilities
│   ├── __init__.py
│   ├── config.py           # Shared constants (HEADERS, API_URLS, etc.)
│   └── file_io.py          # Save/load parquet, JSON files
│
├── 01_extract_auction_search.py      # CLI wrapper (updated)
├── 02_extract_auction_details.py    # CLI wrapper (updated)
├── 03_extract_items_details.py      # CLI wrapper (updated)
├── 04_extract_bid_history.py        # Original (kept for reference)
├── 04_extract_bid_history_new.py    # New CLI wrapper
├── 05_extract_item_enriched_details.py  # Original (kept for reference)
├── 05_extract_item_enriched_details_new.py  # New CLI wrapper
└── REFACTORING_GUIDE.md    # This file
```

## Key Features

### 1. **Extractors** - Pure Data Extraction
The `extractors/` directory contains functions that:
- Fetch data from MaxSold APIs
- Parse JSON responses
- Extract and structure data
- **Do NOT** save to files
- Can be imported and used anywhere

**Example usage:**
```python
from scrapers.extractors.auction_search import fetch_sales_search, extract_sales_from_json

# For live ML prediction - no file I/O!
data = fetch_sales_search(lat=43.65, lng=-79.38, days=30)
sales = extract_sales_from_json(data)
# Use sales directly for prediction
```

### 2. **Pipelines** - Batch Processing with File I/O
The `pipelines/` directory contains functions that:
- Call extractor functions
- Handle batch processing
- Save results to parquet files
- Print progress and summaries

**Example usage:**
```python
from scrapers.pipelines.auction_search_pipeline import run_auction_search_pipeline

# For batch scraping - saves to file
run_auction_search_pipeline(
    output_path="data/auctions.parquet",
    days=60
)
```

### 3. **Utils** - Shared Configuration & I/O
- `config.py`: Centralized API URLs, headers, default directories
- `file_io.py`: Reusable functions for saving/loading parquet and JSON

## Usage Examples

### Example 1: Live ML Model Prediction (No File I/O)

```python
# Import ONLY the extractors
from scrapers.extractors.item_details import fetch_auction_items, extract_items_from_json
from scrapers.extractors.bid_history import fetch_item_bid_history, extract_bids_from_json
from scrapers.extractors.item_enriched import fetch_enriched_details, extract_enriched_data

def predict_item_price(auction_id: str, item_id: str):
    """Predict price for a live item without saving data"""
    
    # Fetch item details
    item_data = fetch_auction_items(auction_id)
    items = extract_items_from_json(item_data, auction_id)
    item = next((i for i in items if i['id'] == item_id), None)
    
    # Fetch bid history
    bid_data = fetch_item_bid_history(auction_id, item_id)
    bids = extract_bids_from_json(bid_data, auction_id, item_id)
    
    # Fetch enriched details
    enriched_data = fetch_enriched_details(item_id)
    enriched = extract_enriched_data(enriched_data, item_id)
    
    # Combine features and make prediction
    features = prepare_features(item, bids, enriched)
    prediction = model.predict(features)
    
    return prediction
```

### Example 2: Batch Scraping Pipeline

```python
from scrapers.pipelines import (
    run_auction_search_pipeline,
    run_item_details_pipeline,
    run_bid_history_pipeline
)

# Step 1: Search for auctions
run_auction_search_pipeline(
    output_path="data/auctions.parquet",
    days=90
)

# Step 2: Get item details for those auctions
run_item_details_pipeline(
    input_parquet="data/auctions.parquet",
    output_path="data/items.parquet"
)

# Step 3: Get bid history for items
run_bid_history_pipeline(
    input_parquet="data/items.parquet",
    output_path="data/bids.parquet",
    max_workers=20  # Parallel processing
)
```

### Example 3: Mixing Extractors with Custom Logic

```python
from scrapers.extractors.auction_details import fetch_auction_details, extract_auction_from_json
import my_custom_ml_model

def analyze_auction(auction_id: str):
    """Fetch auction, analyze with ML, return insights"""
    
    # Use extractor to get data
    data = fetch_auction_details(auction_id)
    auction = extract_auction_from_json(data, auction_id)
    
    # Your custom analysis
    insights = my_custom_ml_model.analyze(auction)
    
    # Return results without saving
    return {
        'auction_id': auction_id,
        'title': auction['title'],
        'predicted_revenue': insights['revenue'],
        'risk_score': insights['risk']
    }
```

## CLI Usage (Backwards Compatible)

The original CLI scripts still work:

```bash
# Search for auctions
python 01_extract_auction_search.py --days 60 -o data/auctions.parquet

# Get auction details
python 02_extract_auction_details.py -p data/auctions.parquet

# Get item details
python 03_extract_items_details.py -p data/auctions.parquet

# Get bid history (parallel)
python 04_extract_bid_history_new.py -p data/items.parquet -w 20

# Get enriched details
python 05_extract_item_enriched_details_new.py -p data/items.parquet --limit 100
```

## Migration from Old Code

If you have existing code using the old scrapers:

**Old way:**
```python
from scrapers.01_extract_auction_search import fetch_sales_search
# This mixed extraction with file I/O
```

**New way - For live predictions:**
```python
from scrapers.extractors.auction_search import fetch_sales_search, extract_sales_from_json
# Pure extraction, no file I/O
```

**New way - For batch scraping:**
```python
from scrapers.pipelines.auction_search_pipeline import run_auction_search_pipeline
# Handles everything including file I/O
```

## Benefits of This Structure

1. ✅ **Reusability**: Extractors can be used by both scrapers AND live ML models
2. ✅ **No Code Duplication**: Single source of truth for extraction logic
3. ✅ **Clean Separation**: Data extraction vs. file I/O are separate concerns
4. ✅ **Testability**: Easy to unit test extraction logic independently
5. ✅ **Flexibility**: Mix and match extractors with custom logic
6. ✅ **Backwards Compatible**: Old CLI scripts still work

## API Reference

See individual module docstrings for detailed API documentation:
- `scrapers/extractors/` - All extraction functions
- `scrapers/pipelines/` - All pipeline functions
- `scrapers/utils/` - Configuration and utilities

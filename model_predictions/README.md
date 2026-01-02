# Model Predictions Pipeline

This pipeline fetches and processes data from MaxSold for ML model inference. It takes a MaxSold item URL and extracts all necessary data for making predictions.

## Overview

The prediction pipeline performs the following steps:

1. **Extract Item ID** from the MaxSold URL
2. **Fetch Enriched Data** from the enriched item API endpoint
3. **Extract Auction ID** from the enriched data
4. **Fetch Auction Data** using the auction ID
5. **Extract Item Details** (filtered to the specific item)
6. **Fetch Bid History** for the item
7. **Save All Data** to the `data/model_predictions` folder

## Directory Structure

```
model_predictions/
├── __init__.py
├── data_fetcher.py          # API fetching utilities
├── prediction_pipeline.py   # Main pipeline script
└── README.md               # This file

data/model_predictions/      # Output directory for all fetched data
├── enriched_item_*.json    # Raw enriched item data
├── auction_*.json          # Raw auction data
├── bid_history_*.json      # Raw bid history data
└── processed_item_*.json   # Combined processed results
```

## Usage

### Basic Usage

Run the pipeline with a MaxSold item URL:

```bash
python model_predictions/prediction_pipeline.py "https://maxsold.com/listing/7561262/tascam-da-60-stereo-dat-recorder"
```

### Custom Output Directory

Specify a different output directory:

```bash
python model_predictions/prediction_pipeline.py \
    "https://maxsold.com/listing/7561262/tascam-da-60-stereo-dat-recorder" \
    --output-dir "custom/output/path"
```

### Using as a Python Module

```python
from model_predictions.prediction_pipeline import PredictionPipeline

# Initialize pipeline
pipeline = PredictionPipeline(output_dir="data/model_predictions")

# Process an item URL
results = pipeline.process_item_url(
    "https://maxsold.com/listing/7561262/tascam-da-60-stereo-dat-recorder"
)

# Check results
if results['success']:
    print(f"Item ID: {results['item_id']}")
    print(f"Auction ID: {results['auction_id']}")
    print(f"Enriched Data: {results['enriched_data']}")
    print(f"Auction Data: {results['auction_data']}")
    print(f"Item Details: {results['item_details']}")
    print(f"Bid History: {results['bid_history']}")
else:
    print(f"Errors: {results['errors']}")
```

## API Endpoints Used

1. **Enriched Item Data**: `https://api.maxsold.com/listings/am/{item_id}/enriched`
2. **Auction Data**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}`
3. **Bid History**: `https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}&itemid={item_id}`

## Output Files

### Raw JSON Files
- `enriched_item_{item_id}_{timestamp}.json` - Raw enriched item data
- `auction_{auction_id}_{timestamp}.json` - Raw auction data  
- `bid_history_auction_{auction_id}_item_{item_id}_{timestamp}.json` - Raw bid history

### Processed Results
- `processed_item_{item_id}_{timestamp}.json` - Combined processed data with structure:

```json
{
  "item_url": "https://maxsold.com/listing/...",
  "timestamp": "2026-01-02T...",
  "success": true,
  "errors": [],
  "item_id": "7561262",
  "auction_id": "104375",
  "enriched_data": { ... },
  "auction_data": { ... },
  "item_details": { ... },
  "bid_history": [ ... ]
}
```

## Data Extractors

The pipeline uses extraction functions from `utils/json_extractors.py`:

- `extract_enriched_data()` - Extracts and flattens enriched item data
- `extract_auction_from_json()` - Extracts auction details
- `extract_items_from_json()` - Extracts item details (with optional item ID filtering)
- `extract_bids_from_json()` - Extracts bid history

### Updated extract_items_from_json

The `extract_items_from_json` function now supports optional item ID filtering:

```python
# Extract all items
all_items = extract_items_from_json(data, auction_id)

# Extract specific items only
specific_items = extract_items_from_json(data, auction_id, item_ids=["7561262", "7561263"])
```

## Error Handling

The pipeline is designed to be resilient:
- Continues processing even if some API calls fail
- Records all errors in the results
- Saves partial results
- Returns success=True if at least enriched data was fetched

## Next Steps: Model Integration

To integrate with your trained ML model:

1. **Load the processed data**:
   ```python
   import json
   with open('data/model_predictions/processed_item_*.json') as f:
       data = json.load(f)
   ```

2. **Feature engineering**: Use the same feature engineering pipeline from `feature_engineering/` to transform the raw data into model features

3. **Model prediction**: Load your trained model and make predictions

4. **Example integration**:
   ```python
   # Run pipeline
   pipeline = PredictionPipeline()
   results = pipeline.process_item_url(item_url)
   
   # Feature engineering
   from feature_engineering.final_dataset_builder import build_features
   features = build_features(results)
   
   # Load model and predict
   import joblib
   model = joblib.load('data/models/trained_model.pkl')
   prediction = model.predict(features)
   ```

## Dependencies

- requests
- Python 3.7+
- Standard library: json, re, pathlib, datetime, argparse

Install dependencies:
```bash
pip install requests
```

## Examples

### Example 1: Process a single item
```bash
python model_predictions/prediction_pipeline.py \
    "https://maxsold.com/listing/7561262/tascam-da-60-stereo-dat-recorder"
```

### Example 2: Batch processing multiple items
```python
from model_predictions.prediction_pipeline import PredictionPipeline

urls = [
    "https://maxsold.com/listing/7561262/item-1",
    "https://maxsold.com/listing/7561263/item-2",
    "https://maxsold.com/listing/7561264/item-3",
]

pipeline = PredictionPipeline()

for url in urls:
    try:
        results = pipeline.process_item_url(url)
        print(f"Processed {results['item_id']}: Success={results['success']}")
    except Exception as e:
        print(f"Error processing {url}: {e}")
```

## Troubleshooting

### Item ID extraction fails
- Ensure the URL format is correct: `https://maxsold.com/listing/{item_id}/...`
- Check that the item ID is present in the URL

### API request fails
- Check your internet connection
- Verify the item/auction exists and is publicly accessible
- Check if the API endpoints have changed

### No bid history found
- This is normal for items with no bids yet
- The pipeline will continue and mark this section as empty

## Contact & Support

For issues or questions, please check the main project documentation in `/docs/`.

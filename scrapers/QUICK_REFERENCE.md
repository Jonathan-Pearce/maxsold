# Quick Reference: Using Extractors for Live ML Predictions

## 🚀 Quick Start

### Step 1: Import Extractors

```python
import sys
sys.path.insert(0, '/workspaces/maxsold/scrapers')

from extractors import (
    fetch_auction_details, extract_auction_from_json,
    fetch_auction_items, extract_items_from_json,
    fetch_enriched_details, extract_enriched_data,
    fetch_item_bid_history, extract_bids_from_json
)
```

### Step 2: Fetch Data

```python
# Get auction details
auction_json = fetch_auction_details(auction_id="12345")
auction_data = extract_auction_from_json(auction_json, "12345")

# Get all items in auction
items_json = fetch_auction_items(auction_id="12345")
items = extract_items_from_json(items_json, "12345")

# Get enriched AI data for specific item
enriched_json = fetch_enriched_details(item_id="67890")
enriched_data = extract_enriched_data(enriched_json, "67890")

# Get bid history for specific item
bids_json = fetch_item_bid_history(auction_id="12345", item_id="67890")
bids = extract_bids_from_json(bids_json, "12345", "67890")
```

### Step 3: Use in Your ML Pipeline

```python
# All functions return Python dicts - ready for ML!
features = {
    'auction_title': auction_data['title'],
    'item_title': items[0]['title'],
    'current_bid': items[0]['current_bid'],
    'bid_count': items[0]['bid_count'],
    'ai_brand': enriched_data['brand'],
    'ai_categories': enriched_data['categories_count'],
    'total_bids': len(bids)
}

# Feed to your model
prediction = your_model.predict(features)
```

---

## 📦 Available Extractors

| Module | What It Fetches | Key Data Returned |
|--------|----------------|-------------------|
| **auction_search** | Search results | List of auctions by location/date |
| **auction_details** | Auction metadata | Title, dates, pickup info, lots count |
| **item_details** | Items/lots | Title, description, bids, prices |
| **bid_history** | Bid timeline | Bid amounts, timestamps, proxy flags |
| **item_enriched** | AI analysis | Brands, categories, attributes, condition |

---

## 🎯 Common Use Cases

### Use Case 1: Predict Final Price for Single Item

```python
def predict_item_price(auction_id: str, item_id: str):
    # Fetch all necessary data
    auction_json = fetch_auction_details(auction_id)
    auction = extract_auction_from_json(auction_json, auction_id)
    
    items_json = fetch_auction_items(auction_id)
    items = extract_items_from_json(items_json, auction_id)
    
    enriched_json = fetch_enriched_details(item_id)
    enriched = extract_enriched_data(enriched_json, item_id)
    
    # Find the target item
    item = next((i for i in items if str(i['id']) == str(item_id)), None)
    
    # Make prediction
    return your_model.predict(auction, item, enriched)
```

### Use Case 2: Analyze Entire Auction

```python
def analyze_auction(auction_id: str):
    # Fetch auction and all items
    auction_json = fetch_auction_details(auction_id)
    auction = extract_auction_from_json(auction_json, auction_id)
    
    items_json = fetch_auction_items(auction_id)
    items = extract_items_from_json(items_json, auction_id)
    
    # Predict for each item
    predictions = []
    for item in items:
        pred = your_model.predict_item(auction, item)
        predictions.append({
            'item_id': item['id'],
            'title': item['title'],
            'current_bid': item['current_bid'],
            'predicted_final': pred
        })
    
    return predictions
```

### Use Case 3: Real-Time API Endpoint

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """
    POST /api/predict
    Body: {"auction_id": "12345", "item_id": "67890"}
    """
    data = request.json
    
    try:
        # Fetch data using extractors
        auction_json = fetch_auction_details(data['auction_id'])
        auction = extract_auction_from_json(auction_json, data['auction_id'])
        
        enriched_json = fetch_enriched_details(data['item_id'])
        enriched = extract_enriched_data(enriched_json, data['item_id'])
        
        # Make prediction
        prediction = your_model.predict(auction, enriched)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
```

---

## 🔑 Key Features of Each Extractor

### auction_details
```python
auction = extract_auction_from_json(json, auction_id)
# Returns:
{
    'id': '12345',
    'title': 'Estate Sale in Toronto',
    'starts': '2025-01-01T10:00:00',
    'ends': '2025-01-03T20:00:00',
    'catalog_lots': 150,
    'intro': 'Description...',
    # ... more fields
}
```

### item_details
```python
items = extract_items_from_json(json, auction_id)
# Returns list of:
{
    'id': '67890',
    'title': 'Antique Chair',
    'description': 'Wooden chair...',
    'current_bid': 25.00,
    'bid_count': 5,
    'lot_number': '001',
    # ... more fields
}
```

### item_enriched
```python
enriched = extract_enriched_data(json, item_id)
# Returns:
{
    'brand': 'Herman Miller',
    'categories': '["Furniture", "Chairs"]',
    'categories_count': 2,
    'condition': 'good',
    'attributes': '[{"name":"Material","value":"Wood"}]',
    # ... more fields
}
```

### bid_history
```python
bids = extract_bids_from_json(json, auction_id, item_id)
# Returns list of:
{
    'time_of_bid': '2025-01-02T14:30:00',
    'amount': 25.00,
    'isproxy': False,
    'bid_number': 1
}
```

---

## ⚡ Performance Tips

1. **Cache auction data** if predicting multiple items from same auction:
   ```python
   auction_cache = {}
   
   def get_auction_cached(auction_id):
       if auction_id not in auction_cache:
           json = fetch_auction_details(auction_id)
           auction_cache[auction_id] = extract_auction_from_json(json, auction_id)
       return auction_cache[auction_id]
   ```

2. **Skip bid history** if not needed (saves API calls):
   ```python
   # Only fetch if your model uses bid history features
   if model_needs_bids:
       bids = fetch_item_bid_history(...)
   ```

3. **Batch item predictions** by fetching auction once:
   ```python
   # Fetch all items at once
   items = extract_items_from_json(fetch_auction_items(auction_id), auction_id)
   
   # Predict all items
   for item in items:
       pred = model.predict(item)
   ```

---

## 🐛 Error Handling

```python
def safe_fetch_item_data(auction_id: str, item_id: str) -> dict:
    """Fetch item data with error handling"""
    result = {
        'status': 'success',
        'data': {},
        'errors': []
    }
    
    try:
        auction_json = fetch_auction_details(auction_id)
        result['data']['auction'] = extract_auction_from_json(auction_json, auction_id)
    except Exception as e:
        result['errors'].append(f'Auction fetch failed: {e}')
        result['status'] = 'partial'
    
    try:
        enriched_json = fetch_enriched_details(item_id)
        result['data']['enriched'] = extract_enriched_data(enriched_json, item_id)
    except Exception as e:
        result['errors'].append(f'Enriched fetch failed: {e}')
        result['status'] = 'partial'
    
    if not result['data']:
        result['status'] = 'error'
    
    return result
```

---

## 📌 Important Notes

1. **No files are saved** - All extractors return Python dicts in memory
2. **API rate limits** - Be mindful when making many requests
3. **Error handling** - Always wrap API calls in try/except
4. **Data freshness** - Data is fetched in real-time (no caching by default)
5. **Network dependency** - Requires internet connection to fetch data

---

## 🔗 See Also

- `REFACTORING_SUMMARY.md` - Complete overview of refactoring
- `examples/live_prediction_example.py` - Full working example
- `examples/ml_integration_example.py` - ML model integration patterns
- `extractors/` - Source code for all extractors

---

*Need help? Check the examples/ directory for complete working code!*

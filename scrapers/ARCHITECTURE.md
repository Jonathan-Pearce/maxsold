# Architecture Overview

## Before Refactoring

```
scrapers/
├── 01_extract_auction_search.py
│   ├── fetch_sales_search()
│   ├── extract_sales_from_json()
│   ├── fetch_all_pages()
│   ├── save_to_parquet()
│   └── main()
│
├── 02_extract_auction_details.py
│   ├── fetch_auction_details()
│   ├── extract_auction_from_json()
│   ├── fetch_multiple_auctions()
│   ├── save_to_parquet()
│   └── main()
│
└── ... (similar for other scrapers)

❌ Problems:
- Extraction logic mixed with I/O
- Can't reuse fetch/extract for live predictions
- Code duplication across files
- Hard to test individual components
```

## After Refactoring

```
┌─────────────────────────────────────────────────────────────┐
│                      TWO USE CASES                           │
├─────────────────────────────┬───────────────────────────────┤
│   DATA SCRAPING PIPELINE    │   LIVE ML PREDICTIONS         │
│   (Save to files)           │   (In-memory only)            │
└─────────────────────────────┴───────────────────────────────┘
             │                            │
             │                            │
             ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    SHARED EXTRACTORS                         │
│  Pure data fetching & parsing - No side effects             │
├─────────────────────────────────────────────────────────────┤
│  extractors/                                                 │
│  ├── auction_search.py                                      │
│  │   ├── fetch_sales_search()    ← API call                │
│  │   └── extract_sales_from_json() ← Parse JSON            │
│  │                                                           │
│  ├── auction_details.py                                     │
│  │   ├── fetch_auction_details()                           │
│  │   └── extract_auction_from_json()                       │
│  │                                                           │
│  ├── item_details.py                                        │
│  ├── bid_history.py                                         │
│  └── item_enriched.py                                       │
└─────────────────────────────────────────────────────────────┘
             ▲                            ▲
             │                            │
    ┌────────┴────────┐        ┌─────────┴──────────┐
    │                 │        │                    │
    │  PIPELINES      │        │  YOUR ML MODEL     │
    │  (Batch I/O)    │        │  (Live predict)    │
    │                 │        │                    │
    │  pipelines/     │        │  model_pipeline/   │
    │  ├── auction_   │        │  ├── predict.py    │
    │  │   search_    │        │  └── api.py        │
    │  │   pipeline   │        │                    │
    │  ├── auction_   │        │  Uses extractors   │
    │  │   details_   │        │  to fetch data     │
    │  │   pipeline   │        │  on-demand!        │
    │  └── ...        │        │                    │
    │                 │        │  NO FILES SAVED    │
    │  Adds:          │        └────────────────────┘
    │  - Batch loops  │
    │  - Progress     │
    │  - save_to_     │
    │    parquet()    │
    │  - Error retry  │
    └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  OUTPUT FILES   │
    │  data/*.parquet │
    └─────────────────┘

✅ Benefits:
- Clear separation of concerns
- Extractors reusable in both contexts
- No code duplication
- Easy to test
- Live predictions without file I/O
```

## Data Flow Comparison

### OLD: Scraping Pipeline (monolithic)

```
User runs script
      │
      ▼
┌──────────────────────┐
│  01_extract_...py    │
│                      │
│  ┌────────────────┐  │
│  │ fetch API      │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ parse JSON     │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ batch loop     │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ save parquet   │  │
│  └────────────────┘  │
└──────────────────────┘
      │
      ▼
  data/*.parquet

❌ Can't reuse fetch/parse for live predictions
```

### NEW: Two Separate Flows

#### Flow 1: Scraping Pipeline

```
User runs script
      │
      ▼
┌──────────────────────┐
│  01_extract_...py    │
│    (CLI wrapper)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Pipeline Module     │
│                      │
│  ┌────────────────┐  │
│  │ batch loop     │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ progress bar   │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ save parquet   │  │
│  └────────────────┘  │
└──────────┬───────────┘
           │ calls
           ▼
┌──────────────────────┐
│  Extractor Module    │
│  (pure functions)    │
│                      │
│  ┌────────────────┐  │
│  │ fetch_*()      │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ extract_*()    │  │
│  └────────────────┘  │
└──────────────────────┘
```

#### Flow 2: Live Predictions

```
API request
 (item_id)
      │
      ▼
┌──────────────────────┐
│  Your ML Model       │
│  predict.py          │
│                      │
│  ┌────────────────┐  │
│  │ preprocess     │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ model.predict()│  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ return result  │  │
│  └────────────────┘  │
└──────────┬───────────┘
           │ calls
           ▼
┌──────────────────────┐
│  Extractor Module    │
│  (same as above!)    │
│                      │
│  ┌────────────────┐  │
│  │ fetch_*()      │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │ extract_*()    │  │
│  └────────────────┘  │
└──────────────────────┘
           │
           ▼
    Returns dict
    (no file I/O!)
```

## Module Dependencies

```
┌─────────────────────────────────────────────┐
│  CLI Scripts (01_*.py, 02_*.py, ...)       │
│  - Backwards compatible                     │
│  - Parse command line args                  │
│  - Call pipeline functions                  │
└─────────────────┬───────────────────────────┘
                  │ imports
                  ▼
┌─────────────────────────────────────────────┐
│  Pipelines (pipelines/*.py)                 │
│  - Batch processing logic                   │
│  - Progress tracking                        │
│  - File I/O operations                      │
│  - Error handling & retries                 │
└─────────────────┬───────────────────────────┘
                  │ imports
                  ▼
┌─────────────────────────────────────────────┐
│  Extractors (extractors/*.py)               │
│  - Pure functions (no side effects)         │
│  - fetch_*(): Make API calls                │
│  - extract_*(): Parse JSON to dict          │
│  - Return Python dicts                      │
└─────────────────┬───────────────────────────┘
                  │ imports
                  ▼
┌─────────────────────────────────────────────┐
│  Utils (utils/*.py)                         │
│  - config.py: API URLs, headers             │
│  - file_io.py: Save/load functions          │
└─────────────────────────────────────────────┘

✅ Clean dependency tree: CLI → Pipelines → Extractors → Utils
✅ No circular dependencies
✅ ML models can import Extractors directly (bypass Pipelines)
```

## Example: How Both Use Cases Share Code

```python
# ═══════════════════════════════════════════════════════
# EXTRACTOR (Shared by both use cases)
# File: extractors/auction_details.py
# ═══════════════════════════════════════════════════════

def fetch_auction_details(auction_id: str) -> dict:
    """Fetch auction data from API"""
    response = requests.get(f"{API_URL}?auctionid={auction_id}")
    return response.json()

def extract_auction_from_json(data: dict, auction_id: str) -> dict:
    """Parse JSON and return structured dict"""
    auction = data.get('auction', {})
    return {
        'id': auction.get('id'),
        'title': auction.get('title'),
        'starts': auction.get('starts'),
        # ... more fields
    }

# ═══════════════════════════════════════════════════════
# USE CASE 1: Data Scraping Pipeline
# File: pipelines/auction_details_pipeline.py
# ═══════════════════════════════════════════════════════

from extractors.auction_details import fetch_auction_details, extract_auction_from_json
from utils.file_io import save_to_parquet

def run_auction_details_pipeline(auction_ids: list, output_path: str):
    """Batch fetch and save to file"""
    all_auctions = []
    
    for auction_id in auction_ids:
        # Use shared extractor
        json_data = fetch_auction_details(auction_id)
        parsed = extract_auction_from_json(json_data, auction_id)
        all_auctions.append(parsed)
    
    # Save to file (pipeline-specific)
    save_to_parquet(all_auctions, output_path)
    print(f"Saved {len(all_auctions)} auctions")

# ═══════════════════════════════════════════════════════
# USE CASE 2: Live ML Prediction
# File: model_pipeline/predict.py
# ═══════════════════════════════════════════════════════

from scrapers.extractors.auction_details import fetch_auction_details, extract_auction_from_json

def predict_item_value(auction_id: str, item_id: str) -> float:
    """Make live prediction without saving files"""
    
    # Use same shared extractor
    json_data = fetch_auction_details(auction_id)
    auction = extract_auction_from_json(json_data, auction_id)
    
    # Extract features
    features = extract_ml_features(auction)
    
    # Predict
    prediction = ml_model.predict(features)
    
    # Return result (NO FILE I/O)
    return prediction

# ═══════════════════════════════════════════════════════
# KEY INSIGHT: Same fetch/extract code, different contexts!
# ═══════════════════════════════════════════════════════
```

## Testing Strategy

```
┌─────────────────────────────────────────────┐
│  Unit Tests                                  │
├─────────────────────────────────────────────┤
│  ✓ Test extractors with mock API responses  │
│  ✓ Test utils functions independently       │
│  ✓ Test pipelines with mock extractors      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Integration Tests                           │
├─────────────────────────────────────────────┤
│  ✓ Test full pipeline end-to-end            │
│  ✓ Test ML prediction workflow              │
│  ✓ Test with real API (carefully!)          │
└─────────────────────────────────────────────┘

Example test structure:

tests/
├── test_extractors.py
│   ├── test_fetch_auction_details()
│   └── test_extract_auction_from_json()
│
├── test_pipelines.py
│   └── test_auction_details_pipeline()
│
└── test_integration.py
    └── test_live_prediction_flow()
```

---

*This architecture enables code reuse while maintaining clean separation of concerns.*

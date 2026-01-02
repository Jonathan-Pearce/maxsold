# MaxSold Feature Engineering Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MAXSOLD FEATURE ENGINEERING                      │
│                              PIPELINE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                           KAGGLE (DATA SOURCE)                             │
├───────────────────────────────────────────────────────────────────────────┤
│  • pearcej/raw-maxsold-auction                                            │
│  • pearcej/raw-maxsold-item                                               │
│  • pearcej/raw-maxsold-item-enriched                                      │
└───────────────────┬───────────────────────────────────────────────────────┘
                    │
                    ▼ (KaggleDataPipeline.download_dataset)
┌───────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA (LOCAL STORAGE)                          │
├───────────────────────────────────────────────────────────────────────────┤
│  ./data/raw/                                                              │
│    ├── auction/*.parquet                                                  │
│    ├── item/*.parquet                                                     │
│    └── item_enriched/*.parquet                                            │
└───────┬─────────────┬─────────────┬─────────────────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌───────────────┐ ┌─────────────────┐ ┌──────────────────────────────┐
│   AUCTION     │ │      ITEM       │ │     ITEM ENRICHED            │
│   FEATURE     │ │    FEATURE      │ │      FEATURE                 │
│   ENGINEER    │ │    ENGINEER     │ │      ENGINEER                │
├───────────────┤ ├─────────────────┤ ├──────────────────────────────┤
│ .fit()        │ │ .fit()          │ │ .fit()                       │
│ .transform()  │ │ .transform()    │ │ .transform()                 │
│               │ │ .save_models()  │ │                              │
├───────────────┤ ├─────────────────┤ ├──────────────────────────────┤
│ Features:     │ │ Features:       │ │ Features:                    │
│ • Duration    │ │ • Embeddings    │ │ • Brands                     │
│ • Postal code │ │   (64-dim)      │ │ • Categories                 │
│ • Pickup time │ │ • Bid features  │ │ • Attributes                 │
│ • Type flags  │ │                 │ │ • Text quality               │
└───────┬───────┘ └────────┬────────┘ └──────────────┬───────────────┘
        │                  │                          │
        ▼                  ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    ENGINEERED DATA (LOCAL STORAGE)                        │
├───────────────────────────────────────────────────────────────────────────┤
│  ./data/engineered/                                                       │
│    ├── auction/auction_engineered.parquet                                 │
│    ├── item/item_engineered.parquet                                       │
│    └── item_enriched/item_enriched_engineered.parquet                     │
│                                                                           │
│  ./data/models/                                                           │
│    └── item_features/                                                     │
│        ├── combined_tfidf_vectorizer.pkl    (for deployment)             │
│        ├── combined_svd_model.pkl           (for deployment)             │
│        └── embeddings_metadata.pkl                                        │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │
                            ▼ (KaggleDataPipeline.upload_dataset)
┌───────────────────────────────────────────────────────────────────────────┐
│                      KAGGLE (ENGINEERED DATASETS)                         │
├───────────────────────────────────────────────────────────────────────────┤
│  • pearcej/engineered-maxsold-auction                                     │
│  • pearcej/engineered-maxsold-item                                        │
│  • pearcej/engineered-maxsold-item-enriched                               │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │
                            ▼ (All 3 datasets loaded)
                    ┌───────────────┐
                    │    DATASET    │
                    │     MERGER    │
                    ├───────────────┤
                    │ .merge()      │
                    │               │
                    │ Handles:      │
                    │ • ID mapping  │
                    │ • Joins       │
                    │ • Overlaps    │
                    └───────┬───────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      FINAL MERGED DATASET                                 │
├───────────────────────────────────────────────────────────────────────────┤
│  ./data/final/maxsold_final_dataset.parquet                              │
│                                                                           │
│  Contains:                                                                │
│  • All auction features                                                   │
│  • All item features (including embeddings)                               │
│  • All enriched item features                                             │
│  • Joined on auction_id and item_id                                       │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │
                            ▼ (KaggleDataPipeline.upload_dataset)
┌───────────────────────────────────────────────────────────────────────────┐
│                    KAGGLE (FINAL DATASET)                                 │
├───────────────────────────────────────────────────────────────────────────┤
│  • pearcej/maxsold-final-dataset                                          │
│    Ready for ML model training!                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE                                 │
└────────────────────────────────────────────────────────────────────────┘

run_pipeline.py
     │
     ├──► KaggleDataPipeline.download_dataset()
     │         │
     │         └──► Downloads 3 raw datasets
     │
     ├──► AuctionFeatureEngineer()
     │         │
     │         ├──► .fit(training_data)      # Learn categories
     │         └──► .transform(data)         # Apply transformations
     │
     ├──► ItemFeatureEngineer()
     │         │
     │         ├──► .fit(training_data)      # Learn TF-IDF vocab + SVD
     │         ├──► .transform(data)         # Generate embeddings
     │         └──► .save_models(path)       # ★ Save for deployment
     │
     ├──► ItemEnrichedFeatureEngineer()
     │         │
     │         ├──► .fit(training_data)      # Learn top brands/categories
     │         └──► .transform(data)         # Apply transformations
     │
     ├──► KaggleDataPipeline.upload_dataset()  (×3)
     │         │
     │         └──► Upload engineered datasets
     │
     ├──► DatasetMerger.merge()
     │         │
     │         └──► Combine all datasets
     │
     └──► KaggleDataPipeline.upload_dataset()
               │
               └──► Upload final dataset


┌────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PHASE                                 │
│                      (Model Deployment / Live ML)                       │
└────────────────────────────────────────────────────────────────────────┘

your_api.py / scoring_service.py
     │
     ├──► Load trained ML model
     │         model = joblib.load('model.pkl')
     │
     ├──► Load feature engineering transformers
     │         auction_eng = AuctionFeatureEngineer()
     │         auction_eng.fit(reference_data)  # Or load saved state
     │         
     │         item_eng = ItemFeatureEngineer()
     │         item_eng.load_models('path/to/models')  # ★ Load saved models
     │         
     │         enriched_eng = ItemEnrichedFeatureEngineer()
     │         enriched_eng.fit(reference_data)  # Or load saved state
     │
     ├──► Receive new data
     │         new_item = get_new_item()
     │
     ├──► Transform features
     │         auction_features = auction_eng.transform(new_auction)
     │         item_features = item_eng.transform(new_item)
     │         enriched_features = enriched_eng.transform(new_enriched)
     │
     ├──► Merge features
     │         merger = DatasetMerger()
     │         final_features = merger.merge(
     │             auction_features,
     │             item_features,
     │             enriched_features
     │         )
     │
     ├──► Make prediction
     │         prediction = model.predict(final_features[model_columns])
     │
     └──► Return result
               return {"prediction": prediction}
```

## Class Hierarchy

```
feature_engineering/
│
├── AuctionFeatureEngineer
│   ├── __init__()
│   ├── fit(df) → self
│   ├── transform(df) → df_transformed
│   ├── fit_transform(df) → df_transformed
│   └── get_model_columns() → list
│
├── ItemFeatureEngineer
│   ├── __init__(n_components, max_features)
│   ├── fit(df) → self
│   ├── transform(df) → df_transformed
│   ├── fit_transform(df) → df_transformed
│   ├── get_model_columns() → list
│   ├── save_models(path)        # ★ For deployment
│   └── load_models(path)        # ★ For inference
│
├── ItemEnrichedFeatureEngineer
│   ├── __init__(top_brands, top_categories, top_attributes)
│   ├── fit(df) → self
│   ├── transform(df) → df_transformed
│   ├── fit_transform(df) → df_transformed
│   └── get_model_columns() → list
│
└── DatasetMerger
    ├── __init__()
    └── merge(df_auction, df_items, df_enriched=None) → df_merged

utils/
│
└── KaggleDataPipeline
    ├── __init__(kaggle_json_path)
    ├── download_dataset(dataset_name, download_path)
    ├── load_dataset(file_path) → DataFrame
    ├── save_dataset(df, file_path, file_format)
    ├── upload_dataset(dataset_dir, dataset_slug, ...)
    └── dataset_exists(dataset_slug) → bool
```

## Data Transformation Flow

```
AUCTION DATA TRANSFORMATION:

Raw Auction Data
    ├── starts, ends → auction_length_hours
    ├── removal_info → postal_code, postal_code_pd_*
    ├── intro → intro_cleaned, intro_length
    ├── pickup_time → pickup_day_*, pickup_is_weekend, pickup_time_hour
    ├── partner_url → has_partner_url
    ├── removal_info → pickup_during_work_hours
    └── title → is_seller_managed, is_condo_auction, is_storage_unit_auction


ITEM DATA TRANSFORMATION:

Raw Item Data
    ├── title, description → TF-IDF → SVD → combined_emb_0..63
    ├── current_bid → current_bid_le_10_binary
    └── current_bid → log_current_bid


ITEM ENRICHED DATA TRANSFORMATION:

Raw Enriched Data
    ├── title, description, qualitativeDescription → length features
    ├── brand → has_brand, brand_*
    ├── brands → has_multiple_brands
    ├── categories (JSON) → cat_*
    ├── condition → condition_*
    ├── working → is_working
    ├── singleKeyItem, numItems → item complexity features
    ├── attributes (JSON) → attr_*, has_attributes
    ├── seriesLine → has_series_line
    ├── description → desc_has_luxury, desc_has_vintage, etc.
    └── multiple fields → data_completeness_score


FINAL MERGE:

  Items (left)
      │
      ├── LEFT JOIN Auction ON auction_id
      │       (adds auction features to each item)
      │
      └── LEFT JOIN Enriched ON item_id
              (adds enriched features to each item)
              
  Result: One row per item with all features
```

## File Organization

```
maxsold/
│
├── 📁 feature_engineering/          # Core transformation logic
│   ├── __init__.py                  # Package exports
│   ├── auction_features.py          # Auction transformations
│   ├── item_features.py             # Item + text embedding transformations
│   ├── item_enriched_features.py    # Enriched item transformations
│   └── dataset_merger.py            # Dataset merging logic
│
├── 📁 utils/                        # Supporting utilities
│   ├── __init__.py                  # Package exports
│   └── kaggle_pipeline.py           # Kaggle API wrapper
│
├── 📁 data/                         # Data storage (gitignored)
│   ├── raw/                         # Downloaded from Kaggle
│   ├── engineered/                  # Transformed datasets
│   ├── final/                       # Merged dataset
│   └── models/                      # Saved models (TF-IDF, SVD)
│
├── 📄 run_pipeline.py               # Main orchestration
├── 📄 test_modules.py               # Testing suite
├── 📄 quickstart.py                 # Interactive menu
│
└── 📚 Documentation
    ├── README_PIPELINE.md           # Technical documentation
    ├── REFACTORING_SUMMARY.md       # What was changed
    ├── QUICKSTART.md                # Quick start guide
    └── ARCHITECTURE.md              # This file
```

## Design Patterns Used

### 1. Fit/Transform Pattern (Scikit-learn style)
```python
# Training
engineer.fit(training_data)           # Learn parameters
df_train = engineer.transform(training_data)

# Inference
df_test = engineer.transform(test_data)  # Use learned parameters
```

### 2. Pipeline Pattern
```python
# Chain transformations
df = load_data()
df = auction_engineer.transform(df)
df = item_engineer.transform(df)
df = enriched_engineer.transform(df)
df = merger.merge(...)
```

### 3. Strategy Pattern
```python
# Different pipelines for different needs
if training:
    engineer.fit_transform(data)
    engineer.save_models()
else:
    engineer.load_models()
    engineer.transform(data)
```

### 4. Facade Pattern
```python
# KaggleDataPipeline wraps complex Kaggle API
kaggle = KaggleDataPipeline()
kaggle.download_dataset(...)  # Simple interface
kaggle.upload_dataset(...)    # Hides complexity
```

## Key Design Decisions

### ✅ Why Fit/Transform?
- Ensures consistency between training and inference
- Prevents data leakage
- Follows ML best practices

### ✅ Why Save Models?
- Text embeddings require fitted TF-IDF vectorizer and SVD
- Loading saves time vs. re-fitting
- Ensures exact same transformations

### ✅ Why Separate Classes?
- Single responsibility principle
- Easy to test individually
- Reusable in different contexts
- Can extend/modify independently

### ✅ Why Kaggle Integration?
- Version control for datasets
- Easy collaboration
- Reproducible experiments
- Backup and sharing

## Performance Characteristics

### Training Phase
- **Auction Features**: Fast (~1-2 sec for 10K rows)
- **Item Features**: Moderate (~30-60 sec for 100K rows)
  - TF-IDF fitting: ~20 sec
  - SVD fitting: ~10 sec
- **Enriched Features**: Fast (~2-5 sec for 100K rows)
- **Merging**: Fast (~1-3 sec)

### Inference Phase
- **Single Item**: < 10 ms
- **Batch (1000 items)**: < 1 sec
- **Large Batch (100K items)**: ~20-30 sec

## Scalability Considerations

### Current Implementation
- In-memory processing with pandas
- Suitable for datasets up to ~1M rows

### For Larger Scale
- Use Dask for distributed processing
- Process in chunks
- Use sparse matrices for embeddings
- Consider database storage

## Extension Points

### Adding New Features
1. Add to appropriate class's `transform()` method
2. Update `get_model_columns()` if needed

### Adding New Dataset
1. Create new FeatureEngineer class
2. Implement fit/transform
3. Add to merger

### Custom Transformations
```python
class CustomFeatureEngineer:
    def fit(self, df):
        # Learn parameters
        return self
    
    def transform(self, df):
        # Apply transformations
        return df_transformed
```

## Best Practices

✅ Always fit on training data only  
✅ Save fitted transformers for production  
✅ Use same Python environment  
✅ Version control your data (Kaggle)  
✅ Test with single records before batch  
✅ Monitor feature distributions in production  
✅ Document feature engineering decisions  

---

This architecture provides a solid foundation for both batch processing and real-time ML applications!

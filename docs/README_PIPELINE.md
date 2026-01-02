# MaxSold Feature Engineering Pipeline

A modular, reusable feature engineering pipeline for MaxSold auction data that can be used for both batch processing and real-time inference.

## Overview

This project provides a complete pipeline for:
1. Downloading raw datasets from Kaggle
2. Applying feature engineering transformations
3. Saving engineered datasets locally and to Kaggle
4. Merging datasets into a final combined dataset

The feature engineering code is designed to be **reusable** for model deployment and live ML applications.

## Project Structure

```
maxsold/
├── feature_engineering/          # Modular feature engineering classes
│   ├── __init__.py
│   ├── auction_features.py       # AuctionFeatureEngineer class
│   ├── item_features.py          # ItemFeatureEngineer class (with text embeddings)
│   ├── item_enriched_features.py # ItemEnrichedFeatureEngineer class
│   └── dataset_merger.py         # DatasetMerger class
├── utils/
│   ├── __init__.py
│   └── kaggle_pipeline.py        # KaggleDataPipeline class
├── data/                         # Data storage (created at runtime)
│   ├── raw/                      # Raw datasets from Kaggle
│   ├── engineered/               # Engineered datasets
│   ├── final/                    # Final merged dataset
│   └── models/                   # Saved models (TF-IDF, SVD)
├── run_pipeline.py               # Main orchestration script
├── requirements.txt
└── README_PIPELINE.md           # This file
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Kaggle API credentials are set up
# Place kaggle.json at: /workspaces/maxsold/.kaggle/kaggle.json
# Or set KAGGLE_CONFIG_DIR environment variable
```

## Usage

### Running the Full Pipeline

```bash
# Run complete pipeline (download, transform, upload)
python run_pipeline.py

# Skip downloads (use existing data)
python run_pipeline.py --skip-download

# Skip uploads to Kaggle
python run_pipeline.py --skip-upload

# Custom data directory
python run_pipeline.py --data-dir ./my_data

# Custom Kaggle credentials
python run_pipeline.py --kaggle-json /path/to/kaggle.json
```

### Using Feature Engineering Classes (for Deployment)

The feature engineering classes are designed to be used independently:

#### 1. Auction Feature Engineering

```python
from feature_engineering import AuctionFeatureEngineer
import pandas as pd

# Load your data
df_auction = pd.read_parquet('auction_data.parquet')

# Create and fit engineer
engineer = AuctionFeatureEngineer()
engineer.fit(df_auction)  # Learn categories (postal districts, pickup days)

# Transform data
df_transformed = engineer.transform(df_auction)

# Or fit and transform in one step
df_transformed = engineer.fit_transform(df_auction)

# Get model-ready columns
model_cols = engineer.get_model_columns()
df_model = df_transformed[model_cols]
```

#### 2. Item Feature Engineering

```python
from feature_engineering import ItemFeatureEngineer

# Load your data
df_items = pd.read_parquet('items_data.parquet')

# Create engineer with custom parameters
engineer = ItemFeatureEngineer(n_components=64, max_features=5000)

# Fit (learns TF-IDF vocabulary and SVD components)
engineer.fit(df_items)

# Save models for later use (e.g., in deployment)
engineer.save_models('./models/item_features')

# Transform new data
df_transformed = engineer.transform(df_items)

# Get model-ready columns
model_cols = engineer.get_model_columns()
df_model = df_transformed[model_cols]
```

**For Live Inference:**

```python
# Load pre-trained models
engineer = ItemFeatureEngineer()
engineer.load_models('./models/item_features')

# Transform new items in real-time
new_item = pd.DataFrame({
    'title': ['Vintage Chair'],
    'description': ['Beautiful antique chair...'],
    'current_bid': [25.0]
})
features = engineer.transform(new_item)
```

#### 3. Item Enriched Feature Engineering

```python
from feature_engineering import ItemEnrichedFeatureEngineer

# Load your data
df_enriched = pd.read_parquet('item_enriched_data.parquet')

# Create engineer
engineer = ItemEnrichedFeatureEngineer(
    top_brands=20,
    top_categories=25,
    top_attributes=15
)

# Fit and transform
df_transformed = engineer.fit_transform(df_enriched)

# Get columns to exclude (raw text/JSON fields)
exclude_cols = engineer.get_model_columns()
model_cols = [c for c in df_transformed.columns if c not in exclude_cols]
df_model = df_transformed[model_cols]
```

#### 4. Dataset Merger

```python
from feature_engineering import DatasetMerger

# Load engineered datasets
df_auction = pd.read_parquet('auction_engineered.parquet')
df_items = pd.read_parquet('item_engineered.parquet')
df_enriched = pd.read_parquet('item_enriched_engineered.parquet')

# Merge datasets
merger = DatasetMerger()
df_final = merger.merge(
    df_auction=df_auction,
    df_items=df_items,
    df_enriched=df_enriched
)
```

### Using Kaggle Pipeline

```python
from utils.kaggle_pipeline import KaggleDataPipeline

# Initialize with credentials
kaggle = KaggleDataPipeline(kaggle_json_path='/path/to/kaggle.json')

# Download a dataset
kaggle.download_dataset(
    dataset_name='pearcej/raw-maxsold-auction',
    download_path='./data/raw/auction'
)

# Load dataset
df = kaggle.load_dataset('./data/raw/auction/auction_details.parquet')

# Save dataset
kaggle.save_dataset(df, './data/processed/auction.parquet', file_format='parquet')

# Upload to Kaggle
kaggle.upload_dataset(
    dataset_dir='./data/processed',
    dataset_slug='username/my-dataset',
    title='My Processed Dataset',
    description='Feature-engineered dataset',
    version_notes='Initial version'
)
```

## Features Created

### Auction Features
- `auction_length_hours`: Duration of auction
- `postal_code_pd`: Postal district (one-hot encoded)
- `intro_length`: Length of cleaned intro text
- `pickup_day_*`: One-hot encoded pickup days
- `pickup_is_weekend`: Weekend pickup indicator
- `pickup_time_hour`: Extracted pickup hour
- `pickup_during_work_hours`: Work hours pickup indicator
- `has_partner_url`: Partner URL presence
- `is_seller_managed`: Seller-managed indicator
- `is_condo_auction`: Condo auction indicator
- `is_storage_unit_auction`: Storage unit indicator

### Item Features
- `combined_emb_0` to `combined_emb_63`: 64-dimensional text embeddings (TF-IDF + LSA)
- `current_bid_le_10_binary`: Binary feature for bids ≤ $10
- `log_current_bid`: Log-transformed current bid

### Item Enriched Features
- Text length features (title, description, qualitative)
- Brand features (20 top brands one-hot encoded)
- Category features (25 top categories one-hot encoded)
- Condition features (one-hot encoded)
- Working status features
- Item complexity features (single/multiple items)
- Attributes features (15 top attributes one-hot encoded)
- Text quality features (luxury, vintage, new, damaged keywords)
- Data completeness score

## Kaggle Datasets

### Raw Datasets (Input)
- `pearcej/raw-maxsold-auction`
- `pearcej/raw-maxsold-item`
- `pearcej/raw-maxsold-item-enriched`

### Engineered Datasets (Output)
- `pearcej/engineered-maxsold-auction`
- `pearcej/engineered-maxsold-item`
- `pearcej/engineered-maxsold-item-enriched`
- `pearcej/maxsold-final-dataset` (merged)

## Design Principles

1. **Modularity**: Each feature engineering class is independent and reusable
2. **Fit/Transform Pattern**: Follows scikit-learn's fit/transform pattern for consistency
3. **Separation of Concerns**: Data loading, transformation, and storage are separate
4. **Deployment-Ready**: Models can be saved and loaded for real-time inference
5. **Type Safety**: Clear interfaces with type hints
6. **Error Handling**: Comprehensive error checking and informative messages

## For Model Deployment

When deploying models that use these features:

1. **Save fitted transformers** after training:
   ```python
   item_engineer.save_models('./models/item_features')
   ```

2. **Load transformers** in deployment:
   ```python
   item_engineer = ItemFeatureEngineer()
   item_engineer.load_models('./models/item_features')
   ```

3. **Transform new data**:
   ```python
   features = item_engineer.transform(new_data)
   predictions = model.predict(features)
   ```

## Testing

To test individual components:

```python
# Test auction feature engineering
from feature_engineering import AuctionFeatureEngineer
import pandas as pd

df = pd.read_parquet('./data/raw/auction/auction_details.parquet')
engineer = AuctionFeatureEngineer()
df_transformed = engineer.fit_transform(df)
print(f"Shape: {df_transformed.shape}")
print(f"Columns: {df_transformed.columns.tolist()}")
```

## Troubleshooting

### Kaggle Authentication Issues
- Ensure `kaggle.json` is at the correct location
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Verify API token is not expired

### Memory Issues
- Process datasets in chunks if needed
- Use `gc.collect()` between operations
- Consider using Dask for very large datasets

### Missing Columns
- Check that raw datasets have expected columns
- Review column name mapping in `DatasetMerger._standardize_columns()`

## Contributing

When adding new features:
1. Add them to the appropriate feature engineering class
2. Update `get_model_columns()` method
3. Test with both batch and single-record inference
4. Update this README

## License

[Your License Here]

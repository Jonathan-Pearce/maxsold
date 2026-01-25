# Source Code

This directory contains all the source code for the MaxSold data science project, organized following the Cookiecutter Data Science structure.

## Directory Structure

```
src/
├── data/           # Scripts to download or generate data
├── features/       # Scripts to turn raw data into features for modeling
├── models/         # Scripts to train models and make predictions
└── visualization/  # Scripts to create exploratory visualizations
```

## data/
Scripts for data collection and scraping:
- `01_extract_auction_search.py` - Extract auction search results
- `02_extract_auction_details.py` - Extract auction details
- `03_extract_items_details.py` - Extract item details
- `04_extract_bid_history.py` - Extract bidding history
- `05_extract_item_enriched_details.py` - Extract enriched item data
- `06_save_auction_images.py` - Download and save auction images
- `monthly_scraping_pipeline.py` - Automated monthly scraping pipeline
- `extract_auction_location_data.py` - Backfilling auction location data
- `test_scraper.py` - Scraper tests

## features/
Scripts for feature engineering:
- `auction_features.py` - Auction-level features
- `auction_details_features.py` - Detailed auction features
- `item_details_text_features_small.py` - Text-based item features
- `image_features.py` - Image-based features
- `dataset_merger.py` - Merge different feature sets
- `final_dataset_builder.py` - Build final modeling dataset

## models/
Scripts for model training and prediction:
- `bid_sequence_model/` - Bid sequence modeling code
- `scripts/` - Training and prediction scripts
- `utils/` - Model utilities

## visualization/
Scripts for creating visualizations and exploratory analysis.

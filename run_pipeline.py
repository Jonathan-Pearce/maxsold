"""
Main Pipeline Script

This script orchestrates the entire feature engineering pipeline:
1. Download raw datasets from Kaggle
2. Apply feature engineering transformations
3. Save engineered datasets locally and to Kaggle
4. Merge all datasets into final combined dataset
5. Save and upload final dataset
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from feature_engineering import (
    AuctionFeatureEngineer,
    ItemFeatureEngineer,
    ItemEnrichedFeatureEngineer,
    DatasetMerger
)
from utils.kaggle_pipeline import KaggleDataPipeline


def main():
    parser = argparse.ArgumentParser(description='MaxSold Feature Engineering Pipeline')
    parser.add_argument('--kaggle-json', type=str, default=None,
                        help='Path to kaggle.json credentials file (optional, uses environment variables by default)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store data')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading from Kaggle (use existing data)')
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip uploading to Kaggle')
    parser.add_argument('--kaggle-username', type=str, default='pearcej',
                        help='Kaggle username for uploading')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    raw_data_dir = data_dir / 'raw'
    engineered_data_dir = data_dir / 'engineered'
    final_data_dir = data_dir / 'final'
    models_dir = data_dir / 'models'
    
    # Create directories
    for dir_path in [raw_data_dir, engineered_data_dir, final_data_dir, models_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Kaggle pipeline
    print("\n" + "="*80)
    print("MAXSOLD FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Use environment variables by default, only pass path if explicitly provided
    kaggle = KaggleDataPipeline(kaggle_json_path=args.kaggle_json)
    
    # ========== STEP 1: DOWNLOAD RAW DATASETS ==========
    if not args.skip_download:
        print("\n" + "="*80)
        print("STEP 1: DOWNLOADING RAW DATASETS FROM KAGGLE")
        print("="*80)
        
        datasets = {
            'auction': 'pearcej/raw-maxsold-auction',
            'item': 'pearcej/raw-maxsold-item',
            'item_enriched': 'pearcej/raw-maxsold-item-enriched'
        }
        
        raw_files = {}
        for name, dataset_slug in datasets.items():
            print(f"\n--- Downloading {name} dataset ---")
            download_dir = raw_data_dir / name
            kaggle.download_dataset(dataset_slug, download_dir)
            
            # Find the parquet file
            parquet_files = list(download_dir.glob('*.parquet'))
            if parquet_files:
                raw_files[name] = parquet_files[0]
                print(f"✓ Found: {raw_files[name].name}")
            else:
                print(f"✗ No parquet file found in {download_dir}")
                sys.exit(1)
    else:
        print("\n--- Skipping download, using existing data ---")
        raw_files = {
            'auction': next(raw_data_dir.glob('auction/*.parquet'), None),
            'item': next(raw_data_dir.glob('item/*.parquet'), None),
            'item_enriched': next(raw_data_dir.glob('item_enriched/*.parquet'), None)
        }
        
        for name, file_path in raw_files.items():
            if not file_path or not file_path.exists():
                print(f"✗ {name} dataset not found")
                sys.exit(1)
    
    # ========== STEP 2: FEATURE ENGINEERING ==========
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    # 2a. Auction Features
    print("\n--- Processing Auction Dataset ---")
    df_auction_raw = kaggle.load_dataset(raw_files['auction'])
    
    auction_engineer = AuctionFeatureEngineer()
    df_auction_engineered = auction_engineer.fit_transform(df_auction_raw)
    
    # Keep only model columns
    model_cols_auction = auction_engineer.get_model_columns()
    df_auction_final = df_auction_engineered[model_cols_auction]
    
    # Save
    auction_output_dir = engineered_data_dir / 'auction'
    auction_output_dir.mkdir(exist_ok=True)
    auction_output_file = auction_output_dir / 'auction_engineered.parquet'
    kaggle.save_dataset(df_auction_final, auction_output_file)
    
    # 2b. Item Features
    print("\n--- Processing Item Dataset ---")
    df_item_raw = kaggle.load_dataset(raw_files['item'])
    
    item_engineer = ItemFeatureEngineer(n_components=64, max_features=5000)
    df_item_engineered = item_engineer.fit_transform(df_item_raw)
    
    # Save models for later use (e.g., deployment)
    item_models_dir = models_dir / 'item_features'
    item_engineer.save_models(item_models_dir)
    
    # Keep only model columns
    model_cols_item = item_engineer.get_model_columns()
    df_item_final = df_item_engineered[model_cols_item]
    
    # Save
    item_output_dir = engineered_data_dir / 'item'
    item_output_dir.mkdir(exist_ok=True)
    item_output_file = item_output_dir / 'item_engineered.parquet'
    kaggle.save_dataset(df_item_final, item_output_file)
    
    # 2c. Item Enriched Features
    print("\n--- Processing Item Enriched Dataset ---")
    df_enriched_raw = kaggle.load_dataset(raw_files['item_enriched'])
    
    enriched_engineer = ItemEnrichedFeatureEngineer(
        top_brands=20,
        top_categories=25,
        top_attributes=15
    )
    df_enriched_engineered = enriched_engineer.fit_transform(df_enriched_raw)
    
    # Keep only model columns (exclude raw text/JSON)
    cols_to_exclude = enriched_engineer.get_model_columns()
    model_cols_enriched = [c for c in df_enriched_engineered.columns if c not in cols_to_exclude]
    df_enriched_final = df_enriched_engineered[model_cols_enriched]
    
    # Save
    enriched_output_dir = engineered_data_dir / 'item_enriched'
    enriched_output_dir.mkdir(exist_ok=True)
    enriched_output_file = enriched_output_dir / 'item_enriched_engineered.parquet'
    kaggle.save_dataset(df_enriched_final, enriched_output_file)
    
    # ========== STEP 3: UPLOAD ENGINEERED DATASETS TO KAGGLE ==========
    if not args.skip_upload:
        print("\n" + "="*80)
        print("STEP 3: UPLOADING ENGINEERED DATASETS TO KAGGLE")
        print("="*80)
        
        engineered_datasets = [
            {
                'dir': auction_output_dir,
                'slug': f'{args.kaggle_username}/engineered-maxsold-auction',
                'title': 'Engineered MaxSold Auction Dataset',
                'description': 'Feature-engineered auction data with temporal, location, and text features'
            },
            {
                'dir': item_output_dir,
                'slug': f'{args.kaggle_username}/engineered-maxsold-item',
                'title': 'Engineered MaxSold Item Dataset',
                'description': 'Feature-engineered item data with text embeddings and bid features'
            },
            {
                'dir': enriched_output_dir,
                'slug': f'{args.kaggle_username}/engineered-maxsold-item-enriched',
                'title': 'Engineered MaxSold Item Enriched Dataset',
                'description': 'Feature-engineered enriched item data with brand, category, and attribute features'
            }
        ]
        
        for dataset_info in engineered_datasets:
            print(f"\n--- Uploading {dataset_info['title']} ---")
            try:
                kaggle.upload_dataset(
                    dataset_dir=dataset_info['dir'],
                    dataset_slug=dataset_info['slug'],
                    title=dataset_info['title'],
                    description=dataset_info['description'],
                    version_notes='Automated feature engineering pipeline update'
                )
            except Exception as e:
                print(f"⚠ Upload warning: {e}")
                print("Continuing with next dataset...")
    
    # ========== STEP 4: MERGE DATASETS ==========
    print("\n" + "="*80)
    print("STEP 4: MERGING ALL DATASETS")
    print("="*80)
    
    merger = DatasetMerger()
    df_final_merged = merger.merge(
        df_auction=df_auction_final,
        df_items=df_item_final,
        df_enriched=df_enriched_final
    )
    
    # Save final merged dataset
    final_output_file = final_data_dir / 'maxsold_final_dataset.parquet'
    kaggle.save_dataset(df_final_merged, final_output_file)
    
    print(f"\n✓ Final merged dataset shape: {df_final_merged.shape}")
    print(f"✓ Saved to: {final_output_file}")
    
    # ========== STEP 5: UPLOAD FINAL DATASET TO KAGGLE ==========
    if not args.skip_upload:
        print("\n" + "="*80)
        print("STEP 5: UPLOADING FINAL MERGED DATASET TO KAGGLE")
        print("="*80)
        
        try:
            kaggle.upload_dataset(
                dataset_dir=final_data_dir,
                dataset_slug=f'{args.kaggle_username}/maxsold-final-dataset',
                title='MaxSold Final Merged Dataset',
                subtitle='Complete feature-engineered dataset for ML modeling',
                description='Final merged dataset combining auction, item, and enriched item features. '
                           'Ready for machine learning model training and deployment.',
                version_notes='Automated pipeline update with full feature engineering'
            )
        except Exception as e:
            print(f"⚠ Upload warning: {e}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✓ Processed 3 raw datasets")
    print(f"✓ Created 3 engineered datasets")
    print(f"✓ Created 1 final merged dataset")
    print(f"\nData locations:")
    print(f"  Raw data:       {raw_data_dir}")
    print(f"  Engineered:     {engineered_data_dir}")
    print(f"  Final dataset:  {final_output_file}")
    print(f"  Models:         {models_dir}")
    print(f"\nTo use these transformations in deployment:")
    print(f"  1. Load saved models from: {item_models_dir}")
    print(f"  2. Import feature engineering classes:")
    print(f"     from feature_engineering import AuctionFeatureEngineer, ItemFeatureEngineer")
    print(f"  3. Apply transformations to new data")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

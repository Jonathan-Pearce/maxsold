#!/usr/bin/env python3
"""
Monthly MaxSold Scraping Pipeline

This pipeline orchestrates the complete scraping workflow:
1. Scrape auction search data (30 days)
2. Scrape auction details from the auction IDs
3. Scrape item details from the auction IDs
4. Scrape bid history from the item IDs
5. Scrape enriched item details from the item IDs
6. Append new data to existing Kaggle datasets and upload
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
from utils.kaggle_pipeline import KaggleDataPipeline

# Dataset configuration
KAGGLE_DATASETS = {
    'auction': {
        'slug': 'pearcej/raw-maxsold-auction',
        'scraper_output': 'data/raw/auction',
        'temp_file': 'auction_details.parquet'
    },
    'item': {
        'slug': 'pearcej/raw-maxsold-item',
        'scraper_output': 'data/raw/item',
        'temp_file': 'items_details.parquet'
    },
    'bid': {
        'slug': 'pearcej/raw-maxsold-bid',
        'scraper_output': 'data/raw/item',
        'temp_file': 'bid_history.parquet'
    },
    'item_enriched': {
        'slug': 'pearcej/raw-maxsold-item-enriched',
        'scraper_output': 'data/raw/item_enriched',
        'temp_file': 'item_enriched_details.parquet'
    }
}

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed!")
        print(f"Error: {e.stderr}")
        raise


def scrape_auction_search(output_path):
    """Step 1: Scrape auction search data"""
    cmd = [
        sys.executable,
        'scrapers/01_extract_auction_search.py',
        '--output', output_path,
        '--days', '30'
    ]
    run_command(cmd, "Scraping auction search (30 days)")
    return output_path


def scrape_auction_details(input_parquet, output_path):
    """Step 2: Scrape auction details"""
    cmd = [
        sys.executable,
        'scrapers/02_extract_auction_details.py',
        '--input-parquet', input_parquet,
        '--output', output_path
    ]
    run_command(cmd, "Scraping auction details")
    return output_path


def scrape_item_details(input_parquet, output_path):
    """Step 3: Scrape item details"""
    cmd = [
        sys.executable,
        'scrapers/03_extract_items_details.py',
        '--input-parquet', input_parquet,
        '--output', output_path
    ]
    run_command(cmd, "Scraping item details")
    return output_path


def scrape_bid_history(input_parquet, output_path):
    """Step 4: Scrape bid history"""
    cmd = [
        sys.executable,
        'scrapers/04_extract_bid_history.py',
        '--input-parquet', input_parquet,
        '--output', output_path
    ]
    run_command(cmd, "Scraping bid history")
    return output_path


def scrape_item_enriched(input_parquet, output_path):
    """Step 5: Scrape enriched item details"""
    cmd = [
        sys.executable,
        'scrapers/05_extract_item_enriched_details.py',
        '--input-parquet', input_parquet,
        '--output', output_path
    ]
    run_command(cmd, "Scraping enriched item details")
    return output_path


def append_and_upload_to_kaggle(dataset_info, new_data_path, pipeline):
    """Download existing Kaggle data, append new data, and upload back"""
    print(f"\n{'='*80}")
    print(f"UPLOADING: {dataset_info['slug']}")
    print(f"{'='*80}")
    
    dataset_slug = dataset_info['slug']
    temp_dir = Path('data/temp_kaggle') / dataset_slug.split('/')[-1]
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load new scraped data
    print(f"Loading new data from: {new_data_path}")
    new_df = pd.read_parquet(new_data_path)
    print(f"  New data shape: {new_df.shape}")
    
    # Try to download existing data from Kaggle
    existing_df = None
    try:
        print(f"Downloading existing data from Kaggle: {dataset_slug}")
        downloaded_path = pipeline.download_dataset(
            dataset_name=dataset_slug,
            download_path=temp_dir#,
            #file_name=dataset_info['temp_file']
        )
        
        # Load existing data
        existing_files = list(temp_dir.glob('*.parquet'))
        if existing_files:
            existing_df = pd.read_parquet(existing_files[0])
            print(f"  Existing data shape: {existing_df.shape}")
    except Exception as e:
        print(f"  Note: Could not download existing data (may not exist yet): {e}")
        print(f"  Will create new dataset...")
    
    # Combine data
    if existing_df is not None:
        print("Appending new data to existing data...")
        
        # Remove duplicates if necessary based on identifying columns
        # For auction: auction_id, for items: id+auction_id, for bids: auction_id+item_id+bid_number
        id_columns = []
        if 'auction_id' in new_df.columns and 'id' not in new_df.columns:
            # This is auction details
            id_columns = ['auction_id']
        elif 'id' in new_df.columns and 'auction_id' in new_df.columns:
            # This is item details or enriched
            id_columns = ['id', 'auction_id']
        elif 'auction_id' in new_df.columns and 'item_id' in new_df.columns and 'bid_number' in new_df.columns:
            # This is bid history
            id_columns = ['auction_id', 'item_id', 'bid_number']
        
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates
        if id_columns:
            original_size = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=id_columns, keep='last')
            print(f"  Removed {original_size - len(combined_df)} duplicate rows")
        
        print(f"  Combined data shape: {combined_df.shape}")
    else:
        combined_df = new_df
        print(f"  Using new data only (no existing data found)")
    
    # Save combined data
    output_path = temp_dir / dataset_info['temp_file']
    combined_df.to_parquet(output_path, index=False, compression='snappy')
    print(f"✓ Saved combined data to: {output_path}")
    
    # Upload to Kaggle
    print(f"Uploading to Kaggle: {dataset_slug}")
    
    # Get dataset title from slug
    dataset_name = dataset_slug.split('/')[-1].replace('-', ' ').title()
    
    try:
        pipeline.upload_dataset(
            dataset_dir=str(temp_dir),
            dataset_slug=dataset_slug,
            title=dataset_name,
            subtitle=f"Updated {datetime.now().strftime('%Y-%m-%d')}",
            description=f"MaxSold scraping data - {dataset_name}. Updated monthly with new auction data.",
            is_public=True,
            version_notes=f"Monthly update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Added {len(new_df)} new rows"
        )
        print(f"✓ Successfully uploaded to Kaggle!")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main pipeline orchestrator"""
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                  MAXSOLD MONTHLY SCRAPING PIPELINE                        ║
║                                                                            ║
║  Timestamp: {TIMESTAMP}                                           ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Create output directories
    for dataset_info in KAGGLE_DATASETS.values():
        Path(dataset_info['scraper_output']).mkdir(parents=True, exist_ok=True)
    
    # Initialize Kaggle pipeline
    print("\n[SETUP] Initializing Kaggle API...")
    try:
        kaggle_pipeline = KaggleDataPipeline()
        print("✓ Kaggle API initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Kaggle API: {e}")
        print("Make sure KAGGLE_USERNAME and KAGGLE_KEY environment variables are set")
        sys.exit(1)
    
    # Define file paths
    #auction_search_path = f"data/temp/auction_search_{TIMESTAMP}.parquet"
    #auction_details_path = f"{KAGGLE_DATASETS['auction']['scraper_output']}/auction_details_{TIMESTAMP}.parquet"
    #item_details_path = f"{KAGGLE_DATASETS['item']['scraper_output']}/items_details_{TIMESTAMP}.parquet"
    #bid_history_path = f"{KAGGLE_DATASETS['bid']['scraper_output']}/bid_history_{TIMESTAMP}.parquet"
    #item_enriched_path = f"{KAGGLE_DATASETS['item_enriched']['scraper_output']}/item_enriched_{TIMESTAMP}.parquet"

    auction_search_path = f"data/temp/auction_search_20251223_175909.parquet"
    auction_details_path = f"{KAGGLE_DATASETS['auction']['scraper_output']}/auction_details_20251223_175909.parquet"
    item_details_path = f"{KAGGLE_DATASETS['item']['scraper_output']}/items_details_20251223_175909.parquet"
    bid_history_path = f"{KAGGLE_DATASETS['bid']['scraper_output']}/bid_history_20251223_175909.parquet"
    item_enriched_path = f"{KAGGLE_DATASETS['item_enriched']['scraper_output']}/item_enriched_20251223_175909.parquet"
    
    # Create temp directory
    Path('data/temp').mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Scrape auction search (this provides auction IDs)
        print("\n" + "█" * 80)
        print("PHASE 1: AUCTION SEARCH")
        print("█" * 80)
        #scrape_auction_search(auction_search_path)
        
        # Step 2 & 3: Scrape auction and item details in parallel (both use auction search output)
        print("\n" + "█" * 80)
        print("PHASE 2: AUCTION & ITEM DETAILS")
        print("█" * 80)
        #scrape_auction_details(auction_search_path, auction_details_path)
        #scrape_item_details(auction_search_path, item_details_path)
        
        # Step 4 & 5: Scrape bid history and enriched details (both use item details output)
        print("\n" + "█" * 80)
        print("PHASE 3: BID HISTORY & ENRICHED DETAILS")
        print("█" * 80)
        #scrape_bid_history(item_details_path, bid_history_path)
        #scrape_item_enriched(item_details_path, item_enriched_path)
        
        # Phase 4: Upload to Kaggle
        print("\n" + "█" * 80)
        print("PHASE 4: KAGGLE UPLOAD")
        print("█" * 80)
        
        append_and_upload_to_kaggle(
            KAGGLE_DATASETS['auction'],
            auction_details_path,
            kaggle_pipeline
        )
        
        append_and_upload_to_kaggle(
            KAGGLE_DATASETS['item'],
            item_details_path,
            kaggle_pipeline
        )
        
        append_and_upload_to_kaggle(
            KAGGLE_DATASETS['bid'],
            bid_history_path,
            kaggle_pipeline
        )
        
        append_and_upload_to_kaggle(
            KAGGLE_DATASETS['item_enriched'],
            item_enriched_path,
            kaggle_pipeline
        )
        
        print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     PIPELINE COMPLETED SUCCESSFULLY! ✓                     ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
        
    except Exception as e:
        print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                         PIPELINE FAILED! ✗                                 ║
║  Error: {str(e)[:65]:<65} ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)


if __name__ == "__main__":
    main()

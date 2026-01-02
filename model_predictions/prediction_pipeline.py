"""
Main prediction pipeline for ML model inference.
Fetches data from MaxSold APIs and prepares it for model prediction.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_predictions.data_fetcher import MaxSoldDataFetcher
from utils.json_extractors import (
    extract_enriched_data,
    extract_auction_from_json,
    extract_items_from_json,
    extract_bids_from_json
)
from utils.raw_data_format import (
    format_item_enriched_data,
    format_auction_details_data,
    format_item_data,
    format_bid_history_data,
)


class PredictionPipeline:
    """Pipeline to fetch and process MaxSold data for model prediction"""
    
    def __init__(self, output_dir: str = "data/model_predictions"):
        """
        Initialize prediction pipeline.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = MaxSoldDataFetcher(output_dir=str(self.output_dir))
        self.results = {}
    
    def process_item_url(self, item_url: str) -> Dict[str, Any]:
        """
        Process a MaxSold item URL through the complete pipeline.
        
        Args:
            item_url: Full MaxSold listing URL
        
        Returns:
            Dictionary containing all extracted data and any errors
        """
        print(f"\n{'='*60}")
        print(f"Processing item URL: {item_url}")
        print(f"{'='*60}\n")
        
        self.results = {
            'item_url': item_url,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'errors': [],
            'item_id': None,
            'auction_id': None,
            'enriched_data': None,
            'auction_data': None,
            'item_details': None,
            'bid_history': None,
        }
        
        # Step 1: Extract item ID from URL
        item_id = self.fetcher.extract_item_id_from_url(item_url)
        if not item_id:
            error = "Failed to extract item ID from URL"
            print(f"ERROR: {error}")
            self.results['errors'].append(error)
            return self.results
        
        self.results['item_id'] = item_id
        print(f"✓ Extracted item ID: {item_id}")
        
        # Step 2: Fetch enriched item data
        print(f"\nFetching enriched data for item {item_id}...")
        enriched_json, error = self.fetcher.fetch_enriched_data(item_id)
        if error:
            self.results['errors'].append(error)
            return self.results
        
        # Step 3: Extract enriched data
        enriched_data = extract_enriched_data(enriched_json, item_id)
        if not enriched_data:
            error = "Failed to extract enriched data"
            print(f"ERROR: {error}")
            self.results['errors'].append(error)
            return self.results
        
        self.results['enriched_data'] = enriched_data
        auction_id = enriched_data.get('amAuctionId')
        
        if not auction_id:
            error = "No auction ID found in enriched data"
            print(f"ERROR: {error}")
            self.results['errors'].append(error)
            return self.results
        
        self.results['auction_id'] = str(auction_id)
        print(f"✓ Extracted auction ID: {auction_id}")
        
        # Convert enriched data to DataFrame and format
        enriched_df = pd.DataFrame([enriched_data])
        try:
            enriched_df = format_item_enriched_data(enriched_df)
        except Exception:
            pass  # keep the raw DF if formatting fails

        # Step 4: Fetch auction data
        print(f"\nFetching auction data for auction {auction_id}...")
        auction_json, error = self.fetcher.fetch_auction_data(str(auction_id))
        auction_df = None
        item_df = None
        if error:
            self.results['errors'].append(error)
            # Continue even if auction fetch fails
        else:
            # Step 5: Extract auction details
            auction_data = extract_auction_from_json(auction_json, str(auction_id))
            if auction_data:
                self.results['auction_data'] = auction_data
                print(f"✓ Extracted auction details")
                auction_df = pd.DataFrame([auction_data])
                try:
                    auction_df = format_auction_details_data(auction_df)
                except Exception:
                    pass
            
            # Step 6: Extract item details (filtered to our specific item)
            item_details_list = extract_items_from_json(
                auction_json, 
                str(auction_id), 
                item_ids=[item_id]
            )
            if item_details_list:
                item_row = item_details_list[0] if item_details_list else None
                self.results['item_details'] = item_row
                print(f"✓ Extracted item details for item {item_id}")
                if item_row:
                    item_df = pd.DataFrame([item_row])
                    try:
                        item_df = format_item_data(item_df)
                    except Exception:
                        pass
        
        # Step 7: Fetch bid history
        print(f"\nFetching bid history for auction {auction_id}, item {item_id}...")
        bid_history_json, error = self.fetcher.fetch_bid_history(str(auction_id), item_id)
        bid_df = None
        if error:
            self.results['errors'].append(error)
            # Continue even if bid history fetch fails
        else:
            # Step 8: Extract bid history
            bid_history = extract_bids_from_json(bid_history_json, str(auction_id), item_id)
            if bid_history:
                self.results['bid_history'] = bid_history
                print(f"✓ Extracted {len(bid_history)} bid records")
                try:
                    bid_df = pd.DataFrame(bid_history)
                    bid_df = format_bid_history_data(bid_df)
                except Exception:
                    pass
            else:
                print(f"  No bid history found (item may have no bids)")
        
        # Mark as successful if we got at least enriched data
        if self.results['enriched_data']:
            self.results['success'] = True
        
        # Step 9: Save processed JSON results
        self._save_results()
        
        # Step 10: Save DataFrames (if present)
        self._save_dataframes(
            item_id=item_id,
            auction_id=str(auction_id),
            enriched_df=enriched_df,
            auction_df=auction_df,
            item_df=item_df,
            bid_df=bid_df
        )
        
        return self.results
    
    def _save_results(self):
        """Save processed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        item_id = self.results.get('item_id', 'unknown')
        filename = f"processed_item_{item_id}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved processed results to {filepath}")

    def _try_save_parquet(self, df: pd.DataFrame, path: Path) -> bool:
        try:
            df.to_parquet(path, index=False)
            return True
        except Exception:
            return False

    def _save_dataframes(
        self,
        item_id: str,
        auction_id: str,
        enriched_df: Optional[pd.DataFrame] = None,
        auction_df: Optional[pd.DataFrame] = None,
        item_df: Optional[pd.DataFrame] = None,
        bid_df: Optional[pd.DataFrame] = None,
    ):
        """Save provided DataFrames to files (parquet preferred, fallback to CSV)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def _save(df: pd.DataFrame, base_name: str):
            if df is None or df.empty:
                return None
            parquet_path = self.output_dir / f"{base_name}_{timestamp}.parquet"
            csv_path = self.output_dir / f"{base_name}_{timestamp}.csv"
            if self._try_save_parquet(df, parquet_path):
                print(f"✓ Saved {base_name} to {parquet_path}")
                return parquet_path
            else:
                df.to_csv(csv_path, index=False)
                print(f"✓ Saved {base_name} to {csv_path} (CSV fallback)")
                return csv_path
        
        paths = {}
        paths['enriched'] = _save(enriched_df, f"enriched_item_{item_id}") if enriched_df is not None else None
        paths['auction'] = _save(auction_df, f"auction_{auction_id}") if auction_df is not None else None
        paths['item_details'] = _save(item_df, f"item_{item_id}_details") if item_df is not None else None
        paths['bid_history'] = _save(bid_df, f"bid_history_auction_{auction_id}_item_{item_id}") if bid_df is not None else None
        
        # Add saved dataframe paths to results
        self.results.setdefault('saved_dataframes', {}).update({k: (str(v) if v else None) for k, v in paths.items()})
    
    def print_summary(self):
        """Print a summary of the pipeline execution"""
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Item ID: {self.results.get('item_id', 'N/A')}")
        print(f"Auction ID: {self.results.get('auction_id', 'N/A')}")
        print(f"Success: {self.results.get('success', False)}")
        
        if self.results.get('enriched_data'):
            print(f"✓ Enriched data extracted")
        if self.results.get('auction_data'):
            print(f"✓ Auction data extracted")
        if self.results.get('item_details'):
            print(f"✓ Item details extracted")
        if self.results.get('bid_history'):
            print(f"✓ Bid history extracted ({len(self.results['bid_history'])} bids)")
        
        if self.results.get('saved_dataframes'):
            print("\nSaved dataframes:")
            for k, v in self.results['saved_dataframes'].items():
                if v:
                    print(f"  - {k}: {v}")
        
        if self.results.get('errors'):
            print(f"\nErrors encountered: {len(self.results['errors'])}")
            for i, error in enumerate(self.results['errors'], 1):
                print(f"  {i}. {error}")
        
        print(f"{'='*60}\n")


def main():
    """Main entry point for the prediction pipeline"""
    parser = argparse.ArgumentParser(
        description='MaxSold ML Prediction Pipeline - Fetch and process data for model inference'
    )
    parser.add_argument(
        'item_url',
        type=str,
        help='MaxSold item listing URL (e.g., https://maxsold.com/listing/7561262/...)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/model_predictions',
        help='Directory to save output files (default: data/model_predictions)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = PredictionPipeline(output_dir=args.output_dir)
    
    try:
        results = pipeline.process_item_url(args.item_url)
        pipeline.print_summary()
        
        # Exit with appropriate code
        if results['success']:
            print("Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("Pipeline completed with errors.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

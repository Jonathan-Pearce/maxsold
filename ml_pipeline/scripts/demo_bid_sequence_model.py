#!/usr/bin/env python3
"""
Bid Sequence Model Demo with Synthetic Data

This script demonstrates the bid sequence model using synthetic bid data.
Useful for testing the model without requiring actual Kaggle data.

Usage:
    python demo_bid_sequence_model.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline.bid_sequence_model import BidSequenceDataLoader, BidSequencePredictor, BidSequenceTrainer


def generate_synthetic_bid_data(n_items=1000, avg_bids_per_item=15, seed=42):
    """
    Generate synthetic bid history data for testing.
    
    Parameters:
    -----------
    n_items : int
        Number of items to generate
    avg_bids_per_item : int
        Average number of bids per item
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Synthetic bid history data
    """
    print(f"\nGenerating synthetic bid data...")
    print(f"  Items: {n_items}")
    print(f"  Avg bids per item: {avg_bids_per_item}")
    
    np.random.seed(seed)
    
    bids = []
    
    for item_id in range(1, n_items + 1):
        # Random number of bids for this item (using Poisson distribution)
        n_bids = max(3, np.random.poisson(avg_bids_per_item))
        
        # Starting bid amount
        start_price = np.random.uniform(5, 50)
        
        # Simulate bid progression
        current_price = start_price
        base_time = pd.Timestamp('2026-01-01 10:00:00')
        
        for bid_num in range(1, n_bids + 1):
            # Bid increment (typically 5-20% of current price)
            increment = current_price * np.random.uniform(0.05, 0.20)
            current_price += increment
            
            # Time increment (1-60 minutes between bids)
            time_delta = pd.Timedelta(minutes=np.random.uniform(1, 60))
            bid_time = base_time + time_delta
            base_time = bid_time
            
            # Proxy bid (30% chance)
            is_proxy = np.random.random() < 0.3
            
            # Create bid record
            # Note: Using REVERSED numbering (as in raw data)
            # Winning bid = 1, first bid = n_bids
            bid = {
                'auction_id': (item_id - 1) // 100 + 1,  # Group items into auctions
                'item_id': item_id,
                'bid_number': n_bids - bid_num + 1,  # REVERSED!
                'time_of_bid': bid_time,
                'amount': current_price,
                'isproxy': is_proxy
            }
            bids.append(bid)
    
    df = pd.DataFrame(bids)
    
    print(f"\nGenerated {len(df):,} bids across {n_items:,} items")
    print(f"  Auctions: {df['auction_id'].nunique()}")
    print(f"  Avg bids per item: {len(df) / n_items:.1f}")
    print(f"  Price range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
    print(f"  Proxy bids: {df['isproxy'].sum():,} ({100*df['isproxy'].mean():.1f}%)")
    
    return df


def demo_full_pipeline():
    """
    Demonstrate the complete training pipeline with synthetic data.
    """
    print("="*80)
    print("BID SEQUENCE MODEL DEMO - SYNTHETIC DATA")
    print("="*80)
    
    # Create temporary directory for demo outputs
    output_dir = Path('data/models/bid_sequence_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    df = generate_synthetic_bid_data(n_items=1000, avg_bids_per_item=15)
    
    # Save to temporary file
    temp_data_path = output_dir / 'synthetic_bid_history.parquet'
    df.to_parquet(temp_data_path, index=False)
    print(f"\nSaved synthetic data to: {temp_data_path}")
    
    # Initialize trainer
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    trainer = BidSequenceTrainer(
        bid_history_path=str(temp_data_path),
        output_dir=str(output_dir),
        max_sequence_length=30,  # Shorter for demo
        lstm_units=32,            # Smaller for demo
        dropout_rate=0.2,
        model_type='gru'          # Faster for demo
    )
    
    # Run pipeline with fewer epochs for demo
    trainer.run_full_pipeline(epochs=20, batch_size=32)
    
    # Test predictions with partial sequences
    print("\n" + "="*80)
    print("TESTING PARTIAL SEQUENCE PREDICTIONS")
    print("="*80)
    
    # Load data again
    loader = BidSequenceDataLoader(bid_history_path=str(temp_data_path))
    bid_df = loader.load_bid_history()
    bid_df = loader.reverse_bid_ordering(bid_df)
    bid_df = loader.create_sequence_features(bid_df)
    
    # Create partial sequences (first 5 bids only)
    print("\nCreating partial sequences (first 5 bids)...")
    X_partial, y_true, item_ids = loader.create_partial_sequences(
        bid_df,
        partial_length=5,
        max_sequence_length=30
    )
    
    # Load trained model
    model = BidSequencePredictor()
    model.load(str(output_dir))
    
    # Make predictions
    predictions = model.predict(X_partial)
    
    # Show results for first 10 items
    print("\nPredictions from first 5 bids:")
    print("-" * 60)
    print(f"{'Item ID':<15} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
    print("-" * 60)
    
    for i in range(min(10, len(predictions))):
        item_id = item_ids[i]
        pred = predictions[i]
        actual = y_true[i]
        error = actual - pred
        error_pct = (error / actual) * 100 if actual != 0 else 0
        
        print(f"{item_id:<15} ${pred:<14.2f} ${actual:<14.2f} ${error:<8.2f} ({error_pct:>5.1f}%)")
    
    print("-" * 60)
    
    # Overall statistics
    errors = y_true - predictions
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / (y_true + 1e-10))) * 100
    
    print("\nOverall Performance (from first 5 bids):")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\nDemo outputs saved to: {output_dir}")
    print("\nYou can now use this model with real data by:")
    print("  1. Download bid history from Kaggle")
    print("  2. Run: python ml_pipeline/scripts/train_bid_sequence_model.py \\")
    print("          --bid_history <path_to_real_data>")


def demo_data_loader_only():
    """
    Demonstrate just the data loader functionality.
    """
    print("="*80)
    print("DATA LOADER DEMO")
    print("="*80)
    
    # Generate synthetic data
    df = generate_synthetic_bid_data(n_items=100, avg_bids_per_item=10)
    
    print("\nSample of raw data (reversed bid numbering):")
    print("-" * 60)
    sample_item = df[df['item_id'] == 1].sort_values('time_of_bid')
    print(sample_item[['item_id', 'bid_number', 'amount', 'time_of_bid', 'isproxy']])
    
    # Initialize data loader
    temp_path = Path('data/models/bid_sequence_demo/temp.parquet')
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(temp_path, index=False)
    
    loader = BidSequenceDataLoader(bid_history_path=str(temp_path))
    bid_df = loader.load_bid_history()
    
    # Reverse ordering
    bid_df = loader.reverse_bid_ordering(bid_df)
    
    print("\nAfter reversing bid numbering:")
    print("-" * 60)
    sample_item = bid_df[bid_df['item_id'] == 1].sort_values('bid_number')
    print(sample_item[['item_id', 'bid_number', 'amount', 'time_of_bid', 'total_bids']])
    
    # Create features
    bid_df = loader.create_sequence_features(bid_df)
    
    print("\nWith sequence features:")
    print("-" * 60)
    sample_item = bid_df[bid_df['item_id'] == 1].sort_values('bid_number')
    cols = ['item_id', 'bid_number', 'amount', 'bid_increment', 
            'hours_since_first', 'bid_position_pct', 'proxy_ratio_so_far']
    print(sample_item[cols])
    
    # Prepare sequences
    X, y, item_ids = loader.prepare_sequences(bid_df, max_sequence_length=20)
    
    print(f"\nPrepared sequences:")
    print(f"  Shape: {X.shape}")
    print(f"  Targets: {y.shape}")
    print(f"  Sample target (final price): ${y[0]:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bid sequence model demo')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'data_loader'],
        default='full',
        help='Demo mode: full pipeline or data loader only'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        demo_full_pipeline()
    else:
        demo_data_loader_only()

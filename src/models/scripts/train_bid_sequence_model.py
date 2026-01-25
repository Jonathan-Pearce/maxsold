#!/usr/bin/env python3
"""
Train Bid Sequence Prediction Model

This script trains a deep learning model to predict final auction prices
from partial bid sequences.

Usage:
    python train_bid_sequence_model.py --bid_history <path> [options]

Example:
    python train_bid_sequence_model.py \
        --bid_history data/bid_history.parquet \
        --item_metadata data/item_metadata.parquet \
        --epochs 50 \
        --model_type lstm
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline.bid_sequence_model.trainer import BidSequenceTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train bid sequence prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_bid_sequence_model.py --bid_history data/bid_history.parquet
  
  # Train with item metadata and custom settings
  python train_bid_sequence_model.py \
      --bid_history data/bid_history.parquet \
      --item_metadata data/item_metadata.parquet \
      --epochs 100 \
      --batch_size 64 \
      --model_type gru \
      --lstm_units 128
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--bid_history',
        type=str,
        required=True,
        help='Path to bid history data file (parquet or csv)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--item_metadata',
        type=str,
        default=None,
        help='Path to item metadata file (parquet or csv)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/models/bid_sequence',
        help='Directory for saving model and outputs (default: data/models/bid_sequence)'
    )
    
    parser.add_argument(
        '--max_sequence_length',
        type=int,
        default=50,
        help='Maximum sequence length (default: 50)'
    )
    
    parser.add_argument(
        '--lstm_units',
        type=int,
        default=64,
        help='Number of LSTM/GRU units (default: 64)'
    )
    
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help='Dropout rate for regularization (default: 0.2)'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['lstm', 'gru'],
        default='lstm',
        help='Type of recurrent layer: lstm or gru (default: lstm)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*80)
    print("BID SEQUENCE PREDICTION MODEL TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Bid history: {args.bid_history}")
    print(f"  Item metadata: {args.item_metadata or 'None'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max sequence length: {args.max_sequence_length}")
    print(f"  Model type: {args.model_type.upper()}")
    print(f"  LSTM units: {args.lstm_units}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print("="*80)
    
    # Check if files exist
    if not Path(args.bid_history).exists():
        print(f"\nError: Bid history file not found: {args.bid_history}")
        print("\nPlease ensure you have downloaded the bid history data from Kaggle.")
        print("You can download it using:")
        print("  kaggle datasets download -d <dataset-name>")
        sys.exit(1)
    
    if args.item_metadata and not Path(args.item_metadata).exists():
        print(f"\nWarning: Item metadata file not found: {args.item_metadata}")
        print("Continuing without item metadata...\n")
        args.item_metadata = None
    
    # Initialize trainer
    trainer = BidSequenceTrainer(
        bid_history_path=args.bid_history,
        item_metadata_path=args.item_metadata,
        output_dir=args.output_dir,
        max_sequence_length=args.max_sequence_length,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        model_type=args.model_type
    )
    
    # Run full training pipeline
    try:
        trainer.run_full_pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nModel and outputs saved to: {args.output_dir}")
        print("\nTo use the trained model:")
        print("  from ml_pipeline.bid_sequence_model import BidSequencePredictor")
        print(f"  model = BidSequencePredictor()")
        print(f"  model.load('{args.output_dir}')")
        print("  predictions = model.predict(X_new)")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR DURING TRAINING")
        print("="*80)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

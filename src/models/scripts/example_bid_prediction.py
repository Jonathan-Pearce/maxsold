"""
Example: Using the Bid Sequence Model for Production Predictions

This example shows how to use a trained bid sequence model to predict
final auction prices from partial bid data in a production scenario.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline.bid_sequence_model import BidSequencePredictor, BidSequenceDataLoader


def example_predict_from_live_bids():
    """
    Example: Predict final price from live bid data.
    
    Scenario: You have the first 10 bids for an item and want to predict
    what the final winning bid will be.
    """
    print("="*80)
    print("EXAMPLE: PREDICTING FINAL PRICE FROM LIVE BIDS")
    print("="*80)
    
    # Simulate receiving live bid data (first 10 bids for an item)
    live_bids = [
        {'bid_number': 1, 'amount': 10.00, 'time_of_bid': '2026-01-18 10:00:00', 'isproxy': False},
        {'bid_number': 2, 'amount': 12.50, 'time_of_bid': '2026-01-18 10:15:00', 'isproxy': True},
        {'bid_number': 3, 'amount': 15.00, 'time_of_bid': '2026-01-18 10:30:00', 'isproxy': False},
        {'bid_number': 4, 'amount': 17.50, 'time_of_bid': '2026-01-18 10:45:00', 'isproxy': True},
        {'bid_number': 5, 'amount': 20.00, 'time_of_bid': '2026-01-18 11:00:00', 'isproxy': False},
        {'bid_number': 6, 'amount': 23.00, 'time_of_bid': '2026-01-18 11:20:00', 'isproxy': True},
        {'bid_number': 7, 'amount': 26.00, 'time_of_bid': '2026-01-18 11:40:00', 'isproxy': False},
        {'bid_number': 8, 'amount': 30.00, 'time_of_bid': '2026-01-18 12:00:00', 'isproxy': True},
        {'bid_number': 9, 'amount': 34.00, 'time_of_bid': '2026-01-18 12:30:00', 'isproxy': False},
        {'bid_number': 10, 'amount': 38.00, 'time_of_bid': '2026-01-18 13:00:00', 'isproxy': True},
    ]
    
    print("\nReceived 10 live bids:")
    for bid in live_bids:
        print(f"  Bid {bid['bid_number']}: ${bid['amount']:.2f} {'(proxy)' if bid['isproxy'] else '(manual)'}")
    
    # Convert to DataFrame
    df = pd.DataFrame(live_bids)
    df['time_of_bid'] = pd.to_datetime(df['time_of_bid'])
    df['isproxy'] = df['isproxy'].astype(int)
    
    # Create features (simplified version for demo)
    df['first_bid_time'] = df['time_of_bid'].min()
    df['hours_since_first'] = (df['time_of_bid'] - df['first_bid_time']).dt.total_seconds() / 3600
    df['bid_increment'] = df['amount'].diff().fillna(0)
    df['bid_position_pct'] = df['bid_number'] / 10  # Assuming 10 is partial length
    df['proxy_count_so_far'] = df['isproxy'].cumsum()
    df['proxy_ratio_so_far'] = df['proxy_count_so_far'] / df['bid_number']
    
    # Prepare sequence (assuming model expects max_sequence_length=30 for demo)
    features = ['amount', 'bid_increment', 'hours_since_first', 
               'isproxy', 'bid_position_pct', 'proxy_ratio_so_far']
    
    feature_values = df[features].values
    
    # Pad to length 30 (demo model length)
    max_seq_len = 30
    padded_seq = np.zeros((max_seq_len, len(features)))
    padded_seq[:len(feature_values)] = feature_values
    
    # Add batch dimension
    X = padded_seq[np.newaxis, :, :]  # Shape: (1, 30, 6)
    
    # Load trained model
    model_path = 'data/models/bid_sequence'
    
    # Check if model exists (use demo model if production model not available)
    if not Path(model_path).exists():
        model_path = 'data/models/bid_sequence_demo'
        if not Path(model_path).exists():
            print(f"\nError: No trained model found at {model_path}")
            print("Please train a model first using:")
            print("  python ml_pipeline/scripts/train_bid_sequence_model.py --bid_history <path>")
            print("\nOr run the demo to create a demo model:")
            print("  python ml_pipeline/scripts/demo_bid_sequence_model.py")
            return
    
    print(f"\nLoading model from: {model_path}")
    model = BidSequencePredictor()
    model.load(model_path)
    
    # Make prediction
    predicted_final_price = model.predict(X)[0]
    
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    print(f"\nCurrent bid (after 10 bids): ${df['amount'].iloc[-1]:.2f}")
    print(f"Predicted final price: ${predicted_final_price:.2f}")
    print(f"Expected additional increase: ${predicted_final_price - df['amount'].iloc[-1]:.2f}")
    print(f"Expected increase: {((predicted_final_price / df['amount'].iloc[-1]) - 1) * 100:.1f}%")
    
    return predicted_final_price


def example_batch_predictions():
    """
    Example: Make predictions for multiple items at once.
    """
    print("\n" + "="*80)
    print("EXAMPLE: BATCH PREDICTIONS FOR MULTIPLE ITEMS")
    print("="*80)
    
    # Simulate having partial bid data for 5 different items
    print("\nProcessing 5 items with partial bid histories...")
    
    # For demo purposes, we'll use synthetic data
    # In production, this would come from your database or API
    
    model_path = 'data/models/bid_sequence_demo'
    if not Path(model_path).exists():
        print(f"\nSkipping: No model found at {model_path}")
        print("Run demo_bid_sequence_model.py first to create a demo model")
        return
    
    # Load the synthetic data used for training
    data_path = Path(model_path) / 'synthetic_bid_history.parquet'
    if not data_path.exists():
        print("No demo data available")
        return
    
    # Load and prepare data
    loader = BidSequenceDataLoader(bid_history_path=str(data_path))
    df = loader.load_bid_history()
    df = loader.reverse_bid_ordering(df)
    df = loader.create_sequence_features(df)
    
    # Get first 5 items with at least 10 bids
    item_counts = df.groupby(['auction_id', 'item_id']).size()
    valid_items = item_counts[item_counts >= 10].head(5).index
    
    # Create partial sequences (first 10 bids only)
    X_batch = []
    item_info = []
    
    for auction_id, item_id in valid_items:
        item_data = df[(df['auction_id'] == auction_id) & (df['item_id'] == item_id)]
        item_data = item_data.sort_values('bid_number').head(10)
        
        features = ['amount', 'bid_increment', 'hours_since_first', 
                   'isproxy', 'bid_position_pct', 'proxy_ratio_so_far']
        feature_values = item_data[features].values
        
        # Pad to length 30 (matching demo model)
        padded_seq = np.zeros((30, len(features)))
        padded_seq[:len(feature_values)] = feature_values
        
        X_batch.append(padded_seq)
        
        actual_final = df[(df['auction_id'] == auction_id) & (df['item_id'] == item_id)]['amount'].max()
        current_bid = item_data['amount'].iloc[-1]
        
        item_info.append({
            'auction_id': auction_id,
            'item_id': item_id,
            'current_bid': current_bid,
            'actual_final': actual_final
        })
    
    X_batch = np.array(X_batch)
    
    # Load model and predict
    model = BidSequencePredictor()
    model.load(str(model_path))
    
    predictions = model.predict(X_batch)
    
    # Display results
    print("\nBatch Prediction Results:")
    print("-" * 80)
    print(f"{'Item':<12} {'Current':<12} {'Predicted':<12} {'Actual':<12} {'Error':<12}")
    print("-" * 80)
    
    for i, info in enumerate(item_info):
        item_str = f"{info['auction_id']}_{info['item_id']}"
        current = info['current_bid']
        predicted = predictions[i]
        actual = info['actual_final']
        error = abs(actual - predicted)
        
        print(f"{item_str:<12} ${current:<11.2f} ${predicted:<11.2f} ${actual:<11.2f} ${error:<11.2f}")
    
    print("-" * 80)
    
    # Overall statistics
    actual_finals = np.array([info['actual_final'] for info in item_info])
    mae = np.mean(np.abs(actual_finals - predictions))
    mape = np.mean(np.abs((actual_finals - predictions) / actual_finals)) * 100
    
    print(f"\nBatch Statistics:")
    print(f"  Mean Absolute Error: ${mae:.2f}")
    print(f"  Mean Absolute Percentage Error: {mape:.2f}%")


if __name__ == "__main__":
    # Run examples
    try:
        # Example 1: Single prediction from live bids
        example_predict_from_live_bids()
        
        # Example 2: Batch predictions
        example_batch_predictions()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

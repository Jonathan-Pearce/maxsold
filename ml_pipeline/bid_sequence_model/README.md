# Bid Sequence Prediction Model

Deep learning model for predicting final auction prices from partial bid sequences.

## Overview

This model uses LSTM/GRU neural networks to predict what the final (winning) bid amount will be for an item auction, given only partial bid history data. This is designed for production use cases where only the first X bids are available.

## Features

- **Sequence Processing**: Handles variable-length bid sequences with automatic padding
- **Reversed Bid Numbering**: Automatically corrects the reversed bid ordering in raw data (winning bid = 1 → first bid = 1)
- **Item Metadata Integration**: Merges auction start/end times and other item metadata for richer features
- **Multiple Architectures**: Supports both LSTM and GRU models
- **Production Ready**: Can predict from partial sequences (e.g., only first 10 bids)
- **Comprehensive Evaluation**: Includes RMSE, MAE, R², and MAPE metrics

## Data Requirements

### Bid History Data
The model expects bid history data with the following columns:
- `auction_id`: Auction identifier
- `item_id`: Item identifier
- `bid_number`: Bid sequence number (can be reversed, will be corrected)
- `time_of_bid`: Timestamp of bid
- `amount`: Bid amount ($)
- `isproxy`: Boolean indicating if bid was automatic/proxy

**Note**: The raw data from Kaggle has reversed bid numbering (winning bid = 1, first bid = total_bids). The data loader automatically corrects this.

### Item Metadata (Optional)
Additional item-level data that can improve predictions:
- `item_id`: Item identifier (for merging)
- `auction_id`: Auction identifier (for merging)
- `start_time`: Auction start time
- `end_time`: Auction end time
- Other auction/item attributes

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow >= 2.15.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

## Quick Start

### Training a Model

```bash
python ml_pipeline/scripts/train_bid_sequence_model.py \
    --bid_history data/bid_history.parquet \
    --item_metadata data/item_metadata.parquet \
    --epochs 50 \
    --model_type lstm
```

### Using a Trained Model

```python
from ml_pipeline.bid_sequence_model import BidSequencePredictor

# Load trained model
model = BidSequencePredictor()
model.load('data/models/bid_sequence')

# Prepare new data (partial sequences)
# X_new shape: (n_items, max_seq_length, n_features)

# Make predictions
predictions = model.predict(X_new)
print(f"Predicted final prices: {predictions}")
```

## Model Architecture

### LSTM Model (Default)
```
Input (sequence_length, n_features)
  ↓
Masking Layer (handles padding)
  ↓
LSTM Layer (64 units, return_sequences=True)
  ↓
Dropout (0.2)
  ↓
LSTM Layer (32 units)
  ↓
Dropout (0.2)
  ↓
Dense Layer (32 units, ReLU)
  ↓
Dropout (0.1)
  ↓
Output Layer (1 unit, Linear)
```

### Features Used
1. **amount**: Current bid amount
2. **bid_increment**: Increase from previous bid
3. **hours_since_first**: Hours elapsed since first bid
4. **isproxy**: Whether bid is automatic (1) or manual (0)
5. **bid_position_pct**: Position in sequence (0-1)
6. **proxy_ratio_so_far**: Ratio of proxy bids up to this point

## Usage Examples

### Basic Training

```python
from ml_pipeline.bid_sequence_model import BidSequenceTrainer

trainer = BidSequenceTrainer(
    bid_history_path='data/bid_history.parquet',
    output_dir='data/models/bid_sequence',
    max_sequence_length=50,
    lstm_units=64,
    dropout_rate=0.2,
    model_type='lstm'
)

trainer.run_full_pipeline(epochs=50, batch_size=32)
```

### Training with Item Metadata

```python
trainer = BidSequenceTrainer(
    bid_history_path='data/bid_history.parquet',
    item_metadata_path='data/item_metadata.parquet',
    output_dir='data/models/bid_sequence'
)

trainer.run_full_pipeline()
```

### Making Predictions with Partial Data

```python
from ml_pipeline.bid_sequence_model import BidSequenceDataLoader, BidSequencePredictor

# Load and prepare data
loader = BidSequenceDataLoader(bid_history_path='data/bid_history.parquet')
bid_df = loader.load_bid_history()
bid_df = loader.reverse_bid_ordering(bid_df)
bid_df = loader.create_sequence_features(bid_df)

# Create partial sequences (e.g., first 10 bids only)
X_partial, y_true, item_ids = loader.create_partial_sequences(
    bid_df,
    partial_length=10,  # Only first 10 bids
    max_sequence_length=50
)

# Load model and predict
model = BidSequencePredictor()
model.load('data/models/bid_sequence')
predictions = model.predict(X_partial)

# Compare predictions to actual final prices
import pandas as pd
results = pd.DataFrame({
    'item_id': item_ids,
    'predicted_final_price': predictions,
    'actual_final_price': y_true,
    'error': y_true - predictions
})
print(results.head())
```

### Using GRU Instead of LSTM

```bash
python ml_pipeline/scripts/train_bid_sequence_model.py \
    --bid_history data/bid_history.parquet \
    --model_type gru \
    --lstm_units 128 \
    --epochs 100
```

### Custom Architecture

```python
from ml_pipeline.bid_sequence_model import BidSequencePredictor

model = BidSequencePredictor(
    sequence_length=100,      # Longer sequences
    n_features=6,
    lstm_units=128,           # More units
    dropout_rate=0.3,         # Higher dropout
    model_type='gru'          # Use GRU
)

model.build_model()
# ... train model ...
```

## Command Line Arguments

```
--bid_history PATH          Path to bid history data (required)
--item_metadata PATH        Path to item metadata (optional)
--output_dir DIR            Output directory (default: data/models/bid_sequence)
--max_sequence_length N     Max sequence length (default: 50)
--lstm_units N              Number of LSTM/GRU units (default: 64)
--dropout_rate FLOAT        Dropout rate (default: 0.2)
--model_type TYPE           'lstm' or 'gru' (default: lstm)
--epochs N                  Training epochs (default: 50)
--batch_size N              Batch size (default: 32)
```

## Output Files

After training, the following files are saved to the output directory:

```
data/models/bid_sequence/
├── bid_sequence_model.h5           # Trained model weights
├── model_config.json                # Model configuration
├── training_history.json            # Training history
├── metrics_summary.json             # Evaluation metrics
├── training_history.png             # Loss curves
├── predictions_comparison.png       # Predictions vs actual
└── error_distribution.png           # Error analysis
```

## Performance Metrics

The model reports the following metrics:
- **RMSE**: Root Mean Squared Error (in dollars)
- **MAE**: Mean Absolute Error (in dollars)
- **R²**: R-squared score (0-1, higher is better)
- **MAPE**: Mean Absolute Percentage Error (%)

## Production Deployment

### Scenario: Predict After First 10 Bids

In production, you'll want to predict the final price after seeing only the first few bids:

```python
# 1. Load trained model
model = BidSequencePredictor()
model.load('data/models/bid_sequence')

# 2. Receive new bid data (first 10 bids for an item)
new_bids = [
    {'bid_number': 1, 'amount': 5.0, 'time_of_bid': '2026-01-15 10:00:00', 'isproxy': False},
    {'bid_number': 2, 'amount': 7.5, 'time_of_bid': '2026-01-15 10:05:00', 'isproxy': True},
    # ... up to bid 10
]

# 3. Convert to features and sequence
# (implement feature extraction based on your data pipeline)
X_new = prepare_sequence_from_bids(new_bids)  # Your preprocessing

# 4. Predict final price
predicted_final_price = model.predict(X_new)[0]
print(f"Predicted final price: ${predicted_final_price:.2f}")
```

## Data Notes

### Reversed Bid Numbering
The raw bid history data from Kaggle has **reversed bid numbering**:
- Winning bid (last bid) = `bid_number = 1`
- First bid = `bid_number = total_bids`

The `BidSequenceDataLoader.reverse_bid_ordering()` method automatically corrects this to:
- First bid = `bid_number = 1`
- Winning bid = `bid_number = total_bids`

### Handling Proxy Bids
Proxy bids are automatic bids placed by MaxSold's bidding system. The `isproxy` field indicates whether a bid was:
- `True` / `1`: Automatic proxy bid
- `False` / `0`: Manual bid by user

The model uses proxy bid patterns as features (proxy count, proxy ratio).

## Troubleshooting

### "No bid history path provided"
- Ensure you provide the `--bid_history` argument with a valid file path
- Download bid history data from Kaggle if needed

### "Unsupported file format"
- The model supports `.parquet` and `.csv` files
- Convert your data to one of these formats

### Low Model Performance
- Try increasing `--lstm_units` (e.g., 128 or 256)
- Increase `--max_sequence_length` if items have many bids
- Use GRU instead of LSTM: `--model_type gru`
- Add item metadata if available
- Increase training data size
- Adjust `--epochs` (more epochs may help, but watch for overfitting)

### Out of Memory Errors
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--max_sequence_length`
- Sample your data to train on fewer items

## Model Comparison

| Model Type | Speed | Memory | Performance |
|------------|-------|---------|-------------|
| LSTM       | Slower | Higher  | Slightly better |
| GRU        | Faster | Lower   | Comparable |

**Recommendation**: Start with LSTM. If training is too slow or memory is an issue, try GRU.

## Future Improvements

Possible enhancements:
- [ ] Bidirectional LSTM/GRU layers
- [ ] Attention mechanism for focusing on important bids
- [ ] Incorporate item category/description embeddings
- [ ] Multi-task learning (predict final price + number of remaining bids)
- [ ] Ensemble models (combine LSTM + GRU + XGBoost)
- [ ] Time-aware features (time of day, day of week patterns)

## References

- MaxSold bid increments: https://support.maxsold.com/hc/en-us/articles/203144054-How-do-bid-increments-work
- MaxSold soft close: https://support.maxsold.com/hc/en-us/articles/203144064-What-does-soft-close-mean

## License

This code is part of the MaxSold data project. See repository LICENSE for details.

## Contact

For issues or questions, please open an issue on the GitHub repository.

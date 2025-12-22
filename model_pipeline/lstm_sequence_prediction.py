"""
Sequential Bid Price Prediction using Deep Learning (LSTM/GRU) and Ensemble Methods

This script trains models to predict the final winning auction price from sequential bid data.
The data contains time-ordered bids for each item, and we want to predict the maximum bid (final price).

Architecture options:
1. LSTM/GRU - Recurrent neural networks for sequence modeling
2. Transformer - Attention-based sequence modeling
3. XGBoost on sequence features - Gradient boosting on engineered sequence statistics

Data: /workspaces/maxsold/data/engineered_data/bid_history/bid_history_engineered_20251201.parquet

Features:
- Sequence features: bid amounts, bid increments, time between bids
- User behavior: proxy vs manual bids, bidding patterns
- Temporal: hours since first bid, time of day, day of week

Target: Maximum bid amount (final winning price) per item
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Traditional ML
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class BidSequenceDataset(Dataset):
    """PyTorch Dataset for sequential bid data"""
    
    def __init__(self, sequences, targets, sequence_lengths, max_len=None):
        """
        Args:
            sequences: List of sequences (each sequence is a 2D array: [seq_len, num_features])
            targets: Array of target values (max bid for each item)
            sequence_lengths: Array of actual sequence lengths
            max_len: Maximum sequence length for padding (if None, uses max from data)
        """
        self.sequences = sequences
        self.targets = torch.FloatTensor(targets)
        self.sequence_lengths = sequence_lengths
        self.max_len = max_len if max_len is not None else max(sequence_lengths)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        seq_len = self.sequence_lengths[idx]
        
        # Pad sequence to max_len
        padded_seq = np.zeros((self.max_len, seq.shape[1]))
        padded_seq[:seq.shape[0], :] = seq
        
        return {
            'sequence': torch.FloatTensor(padded_seq),
            'target': target,
            'length': seq_len
        }


class BidLSTM(nn.Module):
    """LSTM model for sequential bid prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(BidLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x, lengths):
        """
        Args:
            x: [batch_size, seq_len, input_size]
            lengths: [batch_size] - actual sequence lengths
        Returns:
            predictions: [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Pack padded sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention mechanism
        attention_weights = self.attention(output)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * output, dim=1)  # [batch_size, hidden_size * 2]
        
        # Final prediction
        prediction = self.fc(context)
        
        return prediction


class BidGRU(nn.Module):
    """GRU model for sequential bid prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(BidGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # Pack padded sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # GRU forward pass
        packed_output, hidden = self.gru(packed_input)
        
        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention mechanism
        attention_weights = self.attention(output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * output, dim=1)
        
        # Final prediction
        prediction = self.fc(context)
        
        return prediction


class SequentialBidPredictionPipeline:
    """Complete pipeline for sequential bid prediction"""
    
    def __init__(self, data_path, output_dir='model_pipeline/outputs_sequential', 
                 model_type='lstm', device=None):
        """
        Args:
            data_path: Path to bid_history_engineered parquet file
            output_dir: Directory to save outputs
            model_type: 'lstm', 'gru', or 'xgboost'
            device: torch device (cuda/cpu)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_type = model_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.metrics = {}
        
    def load_and_prepare_data(self):
        """Load bid history data and prepare sequences"""
        print("="*80)
        print("LOADING AND PREPARING SEQUENTIAL BID DATA")
        print("="*80)
        
        print(f"\nLoading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(df):,} rows")
        print(f"Unique items: {df['item_id'].nunique():,}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Data overview
        print(f"\nData overview:")
        print(f"  Date range: {df['time_of_bid'].min()} to {df['time_of_bid'].max()}")
        print(f"  Bid amount range: ${df['amount'].min():.2f} to ${df['amount'].max():.2f}")
        print(f"  Average bids per item: {len(df) / df['item_id'].nunique():.1f}")
        
        # Sort by item and time
        df = df.sort_values(['item_id', 'bid_number']).reset_index(drop=True)
        
        # Define sequence features
        self.feature_columns = [
            'amount',  # Current bid amount
            'bid_number',  # Bid sequence number
            'hours_since_first_bid',  # Time elapsed
            'isproxy',  # Proxy bid indicator
            'bid_increment',  # Increment from previous bid
            'bid_position_pct',  # Position in sequence (0-1)
            'proxy_bid_ratio',  # Ratio of proxy bids so far
            'bid_amount_mean',  # Running mean of bid amounts
            'bid_amount_std',  # Running std of bid amounts
        ]
        
        # Check which features exist
        existing_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"\nWarning: Missing features: {missing_features}")
            self.feature_columns = existing_features
        
        print(f"\nUsing {len(self.feature_columns)} sequence features:")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"  {i}. {col}")
        
        return df
    
    def create_sequences(self, df, max_seq_len=50):
        """
        Create sequences for each item
        
        Args:
            df: Dataframe with bid history
            max_seq_len: Maximum sequence length (truncate longer sequences)
        
        Returns:
            sequences: List of sequences
            targets: Array of target values (max bid per item)
            item_ids: Array of item IDs
            sequence_lengths: Array of actual sequence lengths
        """
        print("\n" + "="*80)
        print("CREATING SEQUENCES")
        print("="*80)
        
        sequences = []
        targets = []
        item_ids = []
        sequence_lengths = []
        
        # Group by item
        grouped = df.groupby('item_id')
        
        print(f"\nProcessing {len(grouped):,} items...")
        
        for item_id, item_df in grouped:
            # Sort by bid number to ensure correct order
            item_df = item_df.sort_values('bid_number')
            
            # Get target (max bid = last bid amount)
            target = item_df['amount'].max()
            
            # Get sequence features
            seq_features = item_df[self.feature_columns].values
            
            # Truncate if too long
            if len(seq_features) > max_seq_len:
                # Keep first and last bids, sample from middle
                first_n = max_seq_len // 3
                last_n = max_seq_len // 3
                middle_n = max_seq_len - first_n - last_n
                
                if middle_n > 0:
                    middle_indices = np.linspace(first_n, len(seq_features) - last_n - 1, 
                                                middle_n, dtype=int)
                    selected_indices = list(range(first_n)) + list(middle_indices) + \
                                     list(range(len(seq_features) - last_n, len(seq_features)))
                    seq_features = seq_features[selected_indices]
                else:
                    seq_features = np.vstack([seq_features[:first_n], seq_features[-last_n:]])
            
            sequences.append(seq_features)
            targets.append(target)
            item_ids.append(item_id)
            sequence_lengths.append(len(seq_features))
        
        targets = np.array(targets)
        item_ids = np.array(item_ids)
        sequence_lengths = np.array(sequence_lengths)
        
        print(f"\nSequence statistics:")
        print(f"  Number of sequences: {len(sequences):,}")
        print(f"  Sequence length - mean: {np.mean(sequence_lengths):.1f}, "
              f"median: {np.median(sequence_lengths):.1f}, "
              f"max: {np.max(sequence_lengths)}")
        print(f"  Target (max bid) - mean: ${np.mean(targets):.2f}, "
              f"median: ${np.median(targets):.2f}, "
              f"std: ${np.std(targets):.2f}")
        
        return sequences, targets, item_ids, sequence_lengths
    
    def normalize_sequences(self, train_sequences, val_sequences, test_sequences):
        """Normalize sequence features"""
        print("\n" + "="*80)
        print("NORMALIZING SEQUENCES")
        print("="*80)
        
        # Concatenate all training sequences
        train_data = np.vstack(train_sequences)
        
        # Fit scaler on training data
        self.scaler = RobustScaler()
        self.scaler.fit(train_data)
        
        print(f"\nNormalized {train_data.shape[1]} features using RobustScaler")
        
        # Transform all sequences
        train_sequences_norm = [self.scaler.transform(seq) for seq in train_sequences]
        val_sequences_norm = [self.scaler.transform(seq) for seq in val_sequences]
        test_sequences_norm = [self.scaler.transform(seq) for seq in test_sequences]
        
        return train_sequences_norm, val_sequences_norm, test_sequences_norm
    
    def split_data(self, sequences, targets, item_ids, sequence_lengths, 
                   val_size=0.15, test_size=0.15, random_state=42):
        """Split data into train/val/test sets"""
        print("\n" + "="*80)
        print("SPLITTING DATA")
        print("="*80)
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            np.arange(len(sequences)),
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            random_state=random_state
        )
        
        # Create splits
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx]
        test_sequences = [sequences[i] for i in test_idx]
        
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]
        test_targets = targets[test_idx]
        
        train_lengths = sequence_lengths[train_idx]
        val_lengths = sequence_lengths[val_idx]
        test_lengths = sequence_lengths[test_idx]
        
        train_item_ids = item_ids[train_idx]
        val_item_ids = item_ids[val_idx]
        test_item_ids = item_ids[test_idx]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_sequences):,} sequences ({len(train_idx)/len(sequences)*100:.1f}%)")
        print(f"  Val:   {len(val_sequences):,} sequences ({len(val_idx)/len(sequences)*100:.1f}%)")
        print(f"  Test:  {len(test_sequences):,} sequences ({len(test_idx)/len(sequences)*100:.1f}%)")
        
        return (train_sequences, val_sequences, test_sequences,
                train_targets, val_targets, test_targets,
                train_lengths, val_lengths, test_lengths,
                train_item_ids, val_item_ids, test_item_ids)
    
    def train_deep_learning_model(self, train_loader, val_loader, 
                                  input_size, epochs=50, lr=0.001):
        """Train LSTM or GRU model"""
        print("\n" + "="*80)
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print("="*80)
        
        # Initialize model
        if self.model_type == 'lstm':
            self.model = BidLSTM(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
        elif self.model_type == 'gru':
            self.model = BidGRU(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"\nModel architecture:")
        print(self.model)
        print(f"\nTotal parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Device: {self.device}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        print(f"\nTraining for {epochs} epochs...")
        print(f"Learning rate: {lr}, Batch size: {train_loader.batch_size}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                lengths = batch['length']
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(sequences, lengths).squeeze()
                loss = criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    sequences = batch['sequence'].to(self.device)
                    targets = batch['target'].to(self.device)
                    lengths = batch['length']
                    
                    predictions = self.model(sequences, lengths).squeeze()
                    loss = criterion(predictions, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                          self.output_dir / f'best_{self.model_type}_model.pt')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Best Val: {best_val_loss:.4f}")
            
            # Early stopping check
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(
            torch.load(self.output_dir / f'best_{self.model_type}_model.pt')
        )
        
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def evaluate_model(self, data_loader, split_name='Test'):
        """Evaluate model on a dataset"""
        print("\n" + "="*80)
        print(f"{split_name.upper()} SET EVALUATION")
        print("="*80)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                lengths = batch['length']
                
                predictions = self.model(sequences, lengths).squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        mape = mean_absolute_percentage_error(all_targets, all_predictions) * 100
        
        # Calculate percentage within ranges
        abs_errors = np.abs(all_predictions - all_targets)
        within_5_pct = np.mean(abs_errors / all_targets <= 0.05) * 100
        within_10_pct = np.mean(abs_errors / all_targets <= 0.10) * 100
        within_20_pct = np.mean(abs_errors / all_targets <= 0.20) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'within_5_pct': float(within_5_pct),
            'within_10_pct': float(within_10_pct),
            'within_20_pct': float(within_20_pct)
        }
        
        print(f"\n{split_name} Metrics:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"\nPrediction Accuracy:")
        print(f"  Within 5% of true value:  {within_5_pct:.1f}%")
        print(f"  Within 10% of true value: {within_10_pct:.1f}%")
        print(f"  Within 20% of true value: {within_20_pct:.1f}%")
        
        return metrics, all_predictions, all_targets
    
    def plot_training_history(self, train_losses, val_losses):
        """Plot training and validation loss"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title(f'{self.model_type.upper()} Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTraining history plot saved to: {self.output_dir / 'training_history.png'}")
    
    def plot_predictions(self, predictions, targets, split_name='Test'):
        """Plot prediction vs actual"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Scatter plot
        ax = axes[0, 0]
        ax.scatter(targets, predictions, alpha=0.5, s=20)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Winning Bid ($)', fontsize=11)
        ax.set_ylabel('Predicted Winning Bid ($)', fontsize=11)
        ax.set_title(f'{split_name} Set: Predicted vs Actual', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals
        ax = axes[0, 1]
        residuals = predictions - targets
        ax.scatter(targets, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual Winning Bid ($)', fontsize=11)
        ax.set_ylabel('Residual ($)', fontsize=11)
        ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax = axes[1, 0]
        errors = np.abs(predictions - targets)
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Absolute Error ($)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax.axvline(np.median(errors), color='r', linestyle='--', linewidth=2, 
                   label=f'Median: ${np.median(errors):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Percentage error distribution
        ax = axes[1, 1]
        pct_errors = np.abs((predictions - targets) / targets) * 100
        ax.hist(pct_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Absolute Percentage Error (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
        ax.axvline(np.median(pct_errors), color='r', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(pct_errors):.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'predictions_{split_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPrediction plots saved to: {self.output_dir / f'predictions_{split_name.lower()}.png'}")
    
    def save_results(self, metrics, split_name='test'):
        """Save evaluation metrics"""
        metrics_path = self.output_dir / f'metrics_{split_name}.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'metrics': metrics,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_path}")
    
    def run_pipeline(self, max_seq_len=50, batch_size=32, epochs=50, lr=0.001):
        """Execute full pipeline"""
        print("\n" + "="*80)
        print("SEQUENTIAL BID PREDICTION PIPELINE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model type: {self.model_type.upper()}")
        print(f"Device: {self.device}")
        
        # 1. Load data
        df = self.load_and_prepare_data()
        
        # 2. Create sequences
        sequences, targets, item_ids, sequence_lengths = self.create_sequences(df, max_seq_len)
        
        # 3. Split data
        (train_sequences, val_sequences, test_sequences,
         train_targets, val_targets, test_targets,
         train_lengths, val_lengths, test_lengths,
         train_item_ids, val_item_ids, test_item_ids) = self.split_data(
            sequences, targets, item_ids, sequence_lengths
        )
        
        # 4. Normalize sequences
        train_sequences, val_sequences, test_sequences = self.normalize_sequences(
            train_sequences, val_sequences, test_sequences
        )
        
        # 5. Create datasets and dataloaders
        max_len = max([seq.shape[0] for seq in sequences])
        
        train_dataset = BidSequenceDataset(train_sequences, train_targets, train_lengths, max_len)
        val_dataset = BidSequenceDataset(val_sequences, val_targets, val_lengths, max_len)
        test_dataset = BidSequenceDataset(test_sequences, test_targets, test_lengths, max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 6. Train model
        input_size = train_sequences[0].shape[1]
        self.train_deep_learning_model(train_loader, val_loader, input_size, epochs, lr)
        
        # 7. Evaluate on all splits
        train_metrics, train_preds, train_actual = self.evaluate_model(train_loader, 'Train')
        val_metrics, val_preds, val_actual = self.evaluate_model(val_loader, 'Validation')
        test_metrics, test_preds, test_actual = self.evaluate_model(test_loader, 'Test')
        
        # 8. Create visualizations
        self.plot_predictions(test_preds, test_actual, 'Test')
        self.plot_predictions(train_preds, train_actual, 'Train')
        
        # 9. Save results
        all_metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        self.save_results(all_metrics, 'all')
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nFinal Test Metrics:")
        print(f"  RMSE: ${test_metrics['rmse']:.2f}")
        print(f"  MAE:  ${test_metrics['mae']:.2f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"\nAll outputs saved to: {self.output_dir}")


def main():
    """Main execution"""
    # Configuration
    DATA_PATH = '/workspaces/maxsold/data/engineered_data/bid_history/bid_history_engineered_20251201.parquet'
    OUTPUT_DIR = 'model_pipeline/outputs_sequential_lstm'
    MODEL_TYPE = 'lstm'  # 'lstm', 'gru'
    
    # Hyperparameters
    MAX_SEQ_LEN = 50  # Maximum sequence length
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Initialize and run pipeline
    pipeline = SequentialBidPredictionPipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        model_type=MODEL_TYPE
    )
    
    pipeline.run_pipeline(
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )


if __name__ == "__main__":
    main()
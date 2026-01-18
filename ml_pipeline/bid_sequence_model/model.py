"""
Bid Sequence Prediction Model Architecture

LSTM/GRU-based deep learning model for predicting final auction prices from
partial bid sequences.

Uses PyTorch for the deep learning implementation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Small epsilon value to avoid division by zero
EPSILON = 1e-10


class BidSequenceNN(nn.Module):
    """
    PyTorch neural network module for bid sequence prediction.
    """
    
    def __init__(self, n_features: int, lstm_units: int, dropout_rate: float, model_type: str = 'lstm'):
        super(BidSequenceNN, self).__init__()
        
        self.model_type = model_type.lower()
        self.lstm_units = lstm_units
        
        # Recurrent layers
        if self.model_type == 'gru':
            self.rnn1 = nn.GRU(n_features, lstm_units, batch_first=True)
            self.rnn2 = nn.GRU(lstm_units, lstm_units // 2, batch_first=True)
        else:
            self.rnn1 = nn.LSTM(n_features, lstm_units, batch_first=True)
            self.rnn2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate / 2)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_units // 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x, lengths=None):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, n_features)
        lengths : torch.Tensor, optional
            Actual lengths of sequences (before padding)
        
        Returns:
        --------
        torch.Tensor
            Output predictions of shape (batch_size, 1)
        """
        # First RNN layer
        if lengths is not None:
            # Pack sequences for efficiency
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn1_out, _ = self.rnn1(x_packed)
            rnn1_out, _ = pad_packed_sequence(rnn1_out, batch_first=True)
        else:
            rnn1_out, _ = self.rnn1(x)
        
        x = self.dropout1(rnn1_out)
        
        # Second RNN layer
        if lengths is not None:
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn2_out, _ = self.rnn2(x_packed)
            rnn2_out, _ = pad_packed_sequence(rnn2_out, batch_first=True)
        else:
            rnn2_out, _ = self.rnn2(x)
        
        # Take the last output
        if lengths is not None:
            # Get the last actual output for each sequence
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, rnn2_out.size(2))
            x = rnn2_out.gather(1, idx).squeeze(1)
        else:
            x = rnn2_out[:, -1, :]
        
        x = self.dropout2(x)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x


class BidSequencePredictor:
    """
    Deep learning model for predicting final auction prices from bid sequences.
    
    Uses LSTM/GRU layers to process sequential bid data and predict the final
    winning bid amount.
    """
    
    def __init__(self, 
                 sequence_length: int = 50,
                 n_features: int = 6,
                 lstm_units: int = 64,
                 dropout_rate: float = 0.2,
                 model_type: str = 'lstm',
                 device: Optional[str] = None):
        """
        Initialize the bid sequence predictor.
        
        Parameters:
        -----------
        sequence_length : int
            Maximum length of bid sequences
        n_features : int
            Number of features per bid
        lstm_units : int
            Number of LSTM/GRU units in hidden layers
        dropout_rate : float
            Dropout rate for regularization
        model_type : str
            Type of recurrent layer: 'lstm' or 'gru'
        device : str, optional
            Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model_type = model_type.lower()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.history = None
        
        # Normalization parameters (fit during training)
        self.feature_mean = None
        self.feature_std = None
        
    def build_model(self) -> nn.Module:
        """
        Build the neural network architecture.
        
        Returns:
        --------
        nn.Module
            PyTorch model ready for training
        """
        print("\nBuilding model architecture...")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Features: {self.n_features}")
        print(f"  Hidden units: {self.lstm_units}")
        print(f"  Dropout: {self.dropout_rate}")
        print(f"  Device: {self.device}")
        
        # Create model
        self.model = BidSequenceNN(
            n_features=self.n_features,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            model_type=self.model_type
        ).to(self.device)
        
        print("\n" + "="*60)
        print("Model Architecture:")
        print("="*60)
        print(self.model)
        print("="*60)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*60)
        
        return self.model
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using z-score normalization.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences (n_samples, sequence_length, n_features)
        fit : bool
            If True, compute and store normalization parameters
            
        Returns:
        --------
        np.ndarray
            Normalized sequences
        """
        # Ensure X is float64 to avoid object dtype issues
        X = X.astype(np.float64)
        
        if fit:
            # Compute mean and std across all non-zero values
            # (to avoid including padding in statistics)
            mask = X != 0
            self.feature_mean = np.zeros(self.n_features, dtype=np.float64)
            self.feature_std = np.ones(self.n_features, dtype=np.float64)
            
            for i in range(self.n_features):
                feature_values = X[:, :, i][mask[:, :, i]]
                if len(feature_values) > 0:
                    self.feature_mean[i] = float(feature_values.mean())
                    self.feature_std[i] = float(feature_values.std())
                    if self.feature_std[i] == 0:
                        self.feature_std[i] = 1.0
        
        # Apply normalization
        X_normalized = X.astype(np.float64).copy()
        for i in range(self.n_features):
            X_normalized[:, :, i] = (X[:, :, i] - self.feature_mean[i]) / self.feature_std[i]
            # Reset padding to zero
            X_normalized[:, :, i][X[:, :, i] == 0] = 0
        
        # Ensure output is float64
        X_normalized = X_normalized.astype(np.float64)
            
        return X_normalized
    
    def _get_sequence_lengths(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate actual sequence lengths (non-padded) for each sample.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences (n_samples, sequence_length, n_features)
            
        Returns:
        --------
        np.ndarray
            Array of sequence lengths
        """
        # Find the last non-zero position in each sequence
        # Check if any feature is non-zero
        non_zero_mask = np.any(X != 0, axis=2)  # Shape: (n_samples, sequence_length)
        lengths = np.sum(non_zero_mask, axis=1)
        # Ensure at least length 1
        lengths = np.maximum(lengths, 1)
        return lengths
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: int = 1) -> Dict:
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training sequences (n_samples, sequence_length, n_features)
        y_train : np.ndarray
            Training targets (final prices)
        X_val : np.ndarray, optional
            Validation sequences
        y_val : np.ndarray, optional
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level (0, 1, or 2)
            
        Returns:
        --------
        Dict
            Training history
        """
        if self.model is None:
            self.build_model()
        
        print("\nNormalizing features...")
        # Ensure input is float64 before normalization
        X_train = X_train.astype(np.float64)
        X_train_norm = self.normalize_features(X_train, fit=True)
        print(f"  X_train_norm dtype: {X_train_norm.dtype}, shape: {X_train_norm.shape}")
        train_lengths = self._get_sequence_lengths(X_train)
        
        if X_val is not None:
            X_val = X_val.astype(np.float64)
            X_val_norm = self.normalize_features(X_val, fit=False)
            print(f"  X_val_norm dtype: {X_val_norm.dtype}, shape: {X_val_norm.shape}")
            val_lengths = self._get_sequence_lengths(X_val)
        
        print(f"\nTraining model...")
        print(f"  Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"  Validation samples: {len(X_val)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        train_lengths_tensor = torch.LongTensor(train_lengths).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_lengths_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val_norm).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            val_lengths_tensor = torch.LongTensor(val_lengths).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, val_lengths_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Training history
        history = {
            'loss': [],
            'mae': [],
            'val_loss': [],
            'val_mae': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for batch_X, batch_y, batch_lengths in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X, batch_lengths)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                train_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
            
            train_loss /= len(train_dataset)
            train_mae /= len(train_dataset)
            history['loss'].append(train_loss)
            history['mae'].append(train_mae)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0.0
                val_mae = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y, batch_lengths in val_loader:
                        outputs = self.model(batch_X, batch_lengths)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item() * batch_X.size(0)
                        val_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
                
                val_loss /= len(val_dataset)
                val_mae /= len(val_dataset)
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model weights
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose >= 1 and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {train_loss:.4f} - mae: {train_mae:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Restoring best weights from epoch {epoch+1-patience}")
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if verbose >= 1 and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - mae: {train_mae:.4f}")
        
        self.history = history
        print("\nTraining complete!")
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new sequences.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences (n_samples, sequence_length, n_features)
            
        Returns:
        --------
        np.ndarray
            Predicted final prices
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load() first.")
        
        # Normalize features
        X_norm = self.normalize_features(X, fit=False)
        lengths = self._get_sequence_lengths(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        lengths_tensor = torch.LongTensor(lengths).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor, lengths_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        y : np.ndarray
            True final prices
            
        Returns:
        --------
        Dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y - y_pred) / (y + EPSILON))) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        return metrics
    
    def save(self, save_dir: str):
        """
        Save model and configuration.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save model files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to: {save_dir}")
        
        # Save model weights (PyTorch format)
        model_path = save_dir / 'bid_sequence_model.pt'
        torch.save(self.model.state_dict(), model_path)
        print(f"  Saved model: {model_path.name}")
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'model_type': self.model_type,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
        }
        
        config_path = save_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Saved config: {config_path.name}")
        
        # Save training history if available
        if self.history is not None:
            history_path = save_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"  Saved history: {history_path.name}")
        
        print("Model saved successfully!")
    
    def load(self, save_dir: str):
        """
        Load model and configuration.
        
        Parameters:
        -----------
        save_dir : str
            Directory containing saved model files
        """
        save_dir = Path(save_dir)
        
        print(f"\nLoading model from: {save_dir}")
        
        # Load configuration
        config_path = save_dir / 'model_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.lstm_units = config['lstm_units']
        self.dropout_rate = config['dropout_rate']
        self.model_type = config['model_type']
        self.feature_mean = np.array(config['feature_mean']) if config['feature_mean'] else None
        self.feature_std = np.array(config['feature_std']) if config['feature_std'] else None
        
        # Build model architecture
        self.build_model()
        
        # Load model weights (PyTorch format)
        model_path = save_dir / 'bid_sequence_model.pt'
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"  Loaded model: {model_path.name}")
        else:
            raise FileNotFoundError(f"No model file found in {save_dir}")
        
        print(f"  Model type: {self.model_type.upper()}")
        print("Model loaded successfully!")

"""
Bid Sequence Prediction Model Architecture

LSTM/GRU-based deep learning model for predicting final auction prices from
partial bid sequences.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, Dict
import json
from pathlib import Path


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
                 model_type: str = 'lstm'):
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
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model_type = model_type.lower()
        self.model = None
        self.history = None
        
        # Normalization parameters (fit during training)
        self.feature_mean = None
        self.feature_std = None
        
    def build_model(self) -> keras.Model:
        """
        Build the neural network architecture.
        
        Returns:
        --------
        keras.Model
            Compiled model ready for training
        """
        print("\nBuilding model architecture...")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Features: {self.n_features}")
        print(f"  Hidden units: {self.lstm_units}")
        print(f"  Dropout: {self.dropout_rate}")
        
        # Input layer
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Masking layer to handle padded sequences
        x = layers.Masking(mask_value=0.0)(inputs)
        
        # Choose recurrent layer type
        if self.model_type == 'gru':
            # GRU layers (faster, often similar performance to LSTM)
            x = layers.GRU(self.lstm_units, return_sequences=True)(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.GRU(self.lstm_units // 2)(x)
        else:
            # LSTM layers (default)
            x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
            x = layers.Dropout(self.dropout_rate)(x)
            x = layers.LSTM(self.lstm_units // 2)(x)
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        
        # Output layer (final price prediction)
        outputs = layers.Dense(1, activation='linear')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("\n" + "="*60)
        print("Model Architecture:")
        print("="*60)
        model.summary()
        print("="*60)
        
        return model
    
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
        if fit:
            # Compute mean and std across all non-zero values
            # (to avoid including padding in statistics)
            mask = X != 0
            self.feature_mean = np.zeros(self.n_features)
            self.feature_std = np.ones(self.n_features)
            
            for i in range(self.n_features):
                feature_values = X[:, :, i][mask[:, :, i]]
                if len(feature_values) > 0:
                    self.feature_mean[i] = feature_values.mean()
                    self.feature_std[i] = feature_values.std()
                    if self.feature_std[i] == 0:
                        self.feature_std[i] = 1.0
        
        # Apply normalization
        X_normalized = X.copy()
        for i in range(self.n_features):
            X_normalized[:, :, i] = (X[:, :, i] - self.feature_mean[i]) / self.feature_std[i]
            # Reset padding to zero
            X_normalized[:, :, i][X[:, :, i] == 0] = 0
            
        return X_normalized
    
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
        X_train_norm = self.normalize_features(X_train, fit=True)
        
        if X_val is not None:
            X_val_norm = self.normalize_features(X_val, fit=False)
            validation_data = (X_val_norm, y_val)
        else:
            validation_data = None
        
        print(f"\nTraining model...")
        print(f"  Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"  Validation samples: {len(X_val)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train_norm, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        
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
        
        # Predict
        predictions = self.model.predict(X_norm, verbose=0)
        
        return predictions.flatten()
    
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
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
        
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
        
        # Save model weights
        model_path = save_dir / 'bid_sequence_model.h5'
        self.model.save(model_path)
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
        
        # Load model
        model_path = save_dir / 'bid_sequence_model.h5'
        self.model = keras.models.load_model(model_path)
        
        print(f"  Loaded model: {model_path.name}")
        print(f"  Model type: {self.model_type.upper()}")
        print("Model loaded successfully!")

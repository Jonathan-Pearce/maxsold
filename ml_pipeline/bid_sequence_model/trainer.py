"""
Trainer for Bid Sequence Model

Handles the complete training pipeline including data splitting, training,
evaluation, and visualization.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Optional, Dict, Tuple

from .data_loader import BidSequenceDataLoader
from .model import BidSequencePredictor


class BidSequenceTrainer:
    """
    Complete training pipeline for bid sequence prediction model.
    """
    
    def __init__(self, 
                 bid_history_path: str,
                 item_metadata_path: Optional[str] = None,
                 output_dir: str = 'data/models/bid_sequence',
                 max_sequence_length: int = 50,
                 lstm_units: int = 64,
                 dropout_rate: float = 0.2,
                 model_type: str = 'lstm'):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        bid_history_path : str
            Path to bid history data
        item_metadata_path : str, optional
            Path to item metadata
        output_dir : str
            Directory for saving outputs
        max_sequence_length : int
            Maximum sequence length
        lstm_units : int
            Number of LSTM/GRU units
        dropout_rate : float
            Dropout rate
        model_type : str
            'lstm' or 'gru'
        """
        self.bid_history_path = bid_history_path
        self.item_metadata_path = item_metadata_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_sequence_length = max_sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        
        self.data_loader = None
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_and_prepare_data(self, test_size: float = 0.2, val_size: float = 0.1):
        """
        Load and prepare data for training.
        
        Parameters:
        -----------
        test_size : float
            Fraction of data for testing
        val_size : float
            Fraction of training data for validation
        """
        print("="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)
        
        # Initialize data loader
        self.data_loader = BidSequenceDataLoader(
            bid_history_path=self.bid_history_path,
            item_metadata_path=self.item_metadata_path
        )
        
        # Load bid history
        bid_df = self.data_loader.load_bid_history()
        
        # Load and merge item metadata if available
        if self.item_metadata_path:
            metadata_df = self.data_loader.load_item_metadata()
            if metadata_df is not None:
                bid_df = self.data_loader.merge_item_metadata(bid_df, metadata_df)
        
        # Reverse bid ordering
        bid_df = self.data_loader.reverse_bid_ordering(bid_df)
        
        # Create sequence features
        bid_df = self.data_loader.create_sequence_features(bid_df)
        
        # Print data summary
        summary = self.data_loader.get_data_summary(bid_df)
        print("\nData Summary:")
        print("-" * 40)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Prepare sequences
        X, y, item_ids = self.data_loader.prepare_sequences(
            bid_df,
            max_sequence_length=self.max_sequence_length
        )
        
        # Split data
        print("\nSplitting data...")
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )
        
        print(f"  Training set: {len(self.X_train):,} samples")
        print(f"  Validation set: {len(self.X_val):,} samples")
        print(f"  Test set: {len(self.X_test):,} samples")
        
        return bid_df
    
    def train(self, epochs: int = 50, batch_size: int = 32, verbose: int = 1):
        """
        Train the model.
        
        Parameters:
        -----------
        epochs : int
            Maximum number of epochs
        batch_size : int
            Batch size
        verbose : int
            Verbosity level
        """
        print("\n" + "="*80)
        print("TRAINING MODEL")
        print("="*80)
        
        # Get number of features from data
        n_features = self.X_train.shape[2]
        
        # Initialize model
        self.model = BidSequencePredictor(
            sequence_length=self.max_sequence_length,
            n_features=n_features,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            model_type=self.model_type
        )
        
        # Build and train
        self.model.build_model()
        history = self.model.fit(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return history
    
    def evaluate(self):
        """
        Evaluate model on test set.
        
        Returns:
        --------
        Dict
            Test metrics
        """
        print("\n" + "="*80)
        print("EVALUATING MODEL")
        print("="*80)
        
        # Evaluate on test set
        test_metrics = self.model.evaluate(self.X_test, self.y_test)
        
        print("\nTest Set Performance:")
        print("-" * 40)
        print(f"  RMSE: ${test_metrics['rmse']:.2f}")
        print(f"  MAE:  ${test_metrics['mae']:.2f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        
        # Also evaluate on training and validation sets
        train_metrics = self.model.evaluate(self.X_train, self.y_train)
        val_metrics = self.model.evaluate(self.X_val, self.y_val)
        
        print("\nTraining Set Performance:")
        print("-" * 40)
        print(f"  RMSE: ${train_metrics['rmse']:.2f}")
        print(f"  R²:   {train_metrics['r2']:.4f}")
        
        print("\nValidation Set Performance:")
        print("-" * 40)
        print(f"  RMSE: ${val_metrics['rmse']:.2f}")
        print(f"  R²:   {val_metrics['r2']:.4f}")
        
        # Save metrics
        metrics_summary = {
            'test': test_metrics,
            'validation': val_metrics,
            'training': train_metrics
        }
        
        metrics_path = self.output_dir / 'metrics_summary.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return test_metrics
    
    def plot_training_history(self):
        """
        Plot training history (loss curves).
        """
        if self.model.history is None:
            print("No training history available")
            return
        
        print("\nGenerating training history plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.model.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.model.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.model.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(self.model.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE ($)', fontsize=12)
        axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'training_history.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def plot_predictions(self):
        """
        Plot predicted vs actual prices.
        """
        print("\nGenerating prediction plots...")
        
        # Get predictions
        y_pred = self.model.predict(self.X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(self.y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Final Price ($)', fontsize=12)
        axes[0].set_ylabel('Predicted Final Price ($)', fontsize=12)
        axes[0].set_title('Predicted vs Actual Prices', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = self.y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Final Price ($)', fontsize=12)
        axes[1].set_ylabel('Residual ($)', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'predictions_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def plot_error_distribution(self):
        """
        Plot distribution of prediction errors.
        """
        print("\nGenerating error distribution plots...")
        
        # Get predictions and errors
        y_pred = self.model.predict(self.X_test)
        errors = self.y_test - y_pred
        percent_errors = (errors / (self.y_test + 1e-10)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute error distribution
        axes[0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.median(np.abs(errors)), color='r', linestyle='--', 
                       linewidth=2, label=f'Median: ${np.median(np.abs(errors)):.2f}')
        axes[0].set_xlabel('Absolute Error ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Percentage error distribution
        axes[1].hist(percent_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.median(percent_errors), color='r', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(percent_errors):.1f}%')
        axes[1].set_xlabel('Percentage Error (%)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Percentage Errors', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'error_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def save_model(self):
        """
        Save the trained model.
        """
        print("\nSaving model...")
        self.model.save(self.output_dir)
    
    def run_full_pipeline(self, epochs: int = 50, batch_size: int = 32):
        """
        Run the complete training pipeline.
        
        Parameters:
        -----------
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        """
        print("\n" + "="*80)
        print("BID SEQUENCE PREDICTION MODEL - FULL TRAINING PIPELINE")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train model
        self.train(epochs=epochs, batch_size=batch_size)
        
        # Evaluate
        self.evaluate()
        
        # Generate visualizations
        self.plot_training_history()
        self.plot_predictions()
        self.plot_error_distribution()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nFiles created:")
        print(f"  - bid_sequence_model.pt (trained model - PyTorch)")
        print(f"  - model_config.json (model configuration)")
        print(f"  - training_history.json (training history)")
        print(f"  - metrics_summary.json (evaluation metrics)")
        print(f"  - training_history.png (loss curves)")
        print(f"  - predictions_comparison.png (predictions vs actual)")
        print(f"  - error_distribution.png (error analysis)")

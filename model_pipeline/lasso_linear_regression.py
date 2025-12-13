"""
Lasso Linear Regression Pipeline for Bid Prediction

This script trains a Lasso (L1-regularized) linear regression model to predict
the log-transformed current_bid, then transforms predictions back to original scale.

Target variable: log_current_bid (log(current_bid + 1))
Features: All available features except minimum_bid, bid_count, and current_bid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)


class LassoRegressionPipeline:
    """Pipeline for Lasso linear regression"""
    
    def __init__(self, data_path, output_dir='model_pipeline/outputs_lasso_regression'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.best_params = None
        
    def load_and_prepare_data(self):
        """Load and prepare dataset"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        print(f"\nLoading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        print(f"Original shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Check for target variable
        if 'log_current_bid' not in df.columns:
            print("\nCreating target variable 'log_current_bid'...")
            if 'current_bid' in df.columns:
                df['log_current_bid'] = np.log1p(df['current_bid'])
                print(f"Target created using log(current_bid + 1)")
            else:
                raise ValueError("Cannot create target: 'current_bid' column not found")
        
        # Check target distribution
        print(f"\nTarget statistics (log_current_bid):")
        print(df['log_current_bid'].describe())
        
        # Check original current_bid if available
        if 'current_bid' in df.columns:
            print(f"\nOriginal current_bid statistics:")
            print(df['current_bid'].describe())
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for modeling"""
        print("\n" + "="*70)
        print("PREPROCESSING FEATURES")
        print("="*70)
        
        # Create target
        if 'log_current_bid' not in df.columns:
            raise ValueError("Target variable 'log_current_bid' not found")
        
        y = df['log_current_bid'].copy()
        
        # Drop target and related columns
        columns_to_drop = [
            'log_current_bid',
            'current_bid',  # Don't use actual bid amount
            'minimum_bid',  # Don't use minimum bid
            'bid_count',    # Don't use bid count
            'current_bid_le_10_binary',  # Drop binary target if exists
            'id', 'auction_id', 'totalBids', 'average_bids_per_lot',
            'bidding_extended', 'viewed'
        ]
        
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        print(f"\nDropped columns: {[col for col in columns_to_drop if col in df.columns]}")
        
        # Identify datetime columns
        datetime_cols = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                datetime_cols.append(col)
            elif X[col].dtype == 'object':
                # Try to convert string to datetime
                try:
                    sample = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
                    if sample and isinstance(sample, str):
                        # Check if it looks like a datetime string
                        if any(char in str(sample) for char in ['-', ':', 'T', 'Z']):
                            X[col] = pd.to_datetime(X[col], errors='coerce')
                            if X[col].notna().sum() > len(X) * 0.5:  # If >50% converted successfully
                                datetime_cols.append(col)
                except:
                    pass
        
        print(f"\nDatetime columns found: {datetime_cols}")
        
        # Extract datetime features
        for col in datetime_cols:
            if col in X.columns:
                print(f"  Extracting features from {col}...")
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                X[f'{col}_timestamp'] = X[col].astype(np.int64) // 10**9  # Unix timestamp
                X = X.drop(columns=[col])
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"\nCategorical columns: {len(categorical_cols)}")
        
        # Label encode categorical columns (keep only top categories)
        for col in categorical_cols:
            if col in X.columns:
                # Get top 100 categories
                top_categories = X[col].value_counts().head(100).index
                X[col] = X[col].where(X[col].isin(top_categories), other='Other')
                
                # Label encode
                le = LabelEncoder()
                X[col] = X[col].fillna('Missing')
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"  Encoded {len(self.label_encoders)} categorical columns")
        
        # Handle missing values in numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric columns: {len(numeric_cols)}")
        
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        
        # Check for any remaining non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"\nWarning: Dropping non-numeric columns: {non_numeric}")
            X = X.drop(columns=non_numeric)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"\nFinal feature set: {len(self.feature_names)} features")
        print(f"Final X shape: {X.shape}")
        print(f"Final y shape: {y.shape}")
        
        # Check for infinite values
        inf_cols = X.columns[np.isinf(X).any()].tolist()
        if inf_cols:
            print(f"\nReplacing infinite values in columns: {inf_cols}")
            X = X.replace([np.inf, -np.inf], np.nan)
            for col in inf_cols:
                X[col] = X[col].fillna(X[col].median())
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTrain set: {X_train.shape[0]:,} samples")
        print(f"Test set:  {X_test.shape[0]:,} samples")
        
        print(f"\nTrain target (log_current_bid) statistics:")
        print(f"  Mean: {y_train.mean():.4f}")
        print(f"  Median: {y_train.median():.4f}")
        print(f"  Std: {y_train.std():.4f}")
        print(f"  Min: {y_train.min():.4f}")
        print(f"  Max: {y_train.max():.4f}")
        
        print(f"\nTest target (log_current_bid) statistics:")
        print(f"  Mean: {y_test.mean():.4f}")
        print(f"  Median: {y_test.median():.4f}")
        print(f"  Std: {y_test.std():.4f}")
        print(f"  Min: {y_test.min():.4f}")
        print(f"  Max: {y_test.max():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train Lasso regression with hyperparameter tuning"""
        print("\n" + "="*70)
        print("TRAINING LASSO REGRESSION")
        print("="*70)
        
        # Standardize features (important for L1 regularization)
        print("\nStandardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Features standardized (mean=0, std=1)")
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]  # Regularization strength
        }
        
        print(f"\nHyperparameter search space:")
        print(f"  alpha values: {param_grid['alpha']}")
        
        # Create base model with L1 penalty (Lasso)
        base_model = Lasso(
            max_iter=10000,
            random_state=42,
            selection='random'
        )
        
        # Perform grid search with cross-validation
        print(f"\nPerforming 5-fold cross-validation...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\n{'='*70}")
        print(f"Best hyperparameters:")
        print(f"  alpha: {self.best_params['alpha']}")
        print(f"  Best CV MSE: {-grid_search.best_score_:.4f}")
        print(f"  Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        print(f"{'='*70}")
        
        # Count non-zero coefficients (feature selection by Lasso)
        n_nonzero = np.sum(self.model.coef_ != 0)
        n_total = len(self.model.coef_)
        print(f"\nLasso feature selection:")
        print(f"  Non-zero coefficients: {n_nonzero} / {n_total}")
        print(f"  Features selected: {100 * n_nonzero / n_total:.1f}%")
        print(f"  Features eliminated: {n_total - n_nonzero}")
        
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*70)
        print("EVALUATING MODEL")
        print("="*70)
        
        # Scale data
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions (log scale)
        y_train_pred_log = self.model.predict(X_train_scaled)
        y_test_pred_log = self.model.predict(X_test_scaled)
        
        # Calculate metrics on log scale
        train_metrics_log = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_log)),
            'mae': mean_absolute_error(y_train, y_train_pred_log),
            'r2': r2_score(y_train, y_train_pred_log)
        }
        
        test_metrics_log = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_log)),
            'mae': mean_absolute_error(y_test, y_test_pred_log),
            'r2': r2_score(y_test, y_test_pred_log)
        }
        
        print("\nMETRICS ON LOG SCALE:")
        print("-" * 70)
        print("TRAIN SET:")
        print(f"  RMSE: {train_metrics_log['rmse']:.4f}")
        print(f"  MAE:  {train_metrics_log['mae']:.4f}")
        print(f"  R²:   {train_metrics_log['r2']:.4f}")
        
        print("\nTEST SET:")
        print(f"  RMSE: {test_metrics_log['rmse']:.4f}")
        print(f"  MAE:  {test_metrics_log['mae']:.4f}")
        print(f"  R²:   {test_metrics_log['r2']:.4f}")
        
        # Transform predictions back to original scale using expm1 (inverse of log1p)
        print("\n" + "="*70)
        print("TRANSFORMING PREDICTIONS TO ORIGINAL SCALE")
        print("="*70)
        
        y_train_pred_original = np.expm1(y_train_pred_log)
        y_test_pred_original = np.expm1(y_test_pred_log)
        
        # Need original scale targets for comparison
        y_train_original = np.expm1(y_train)
        y_test_original = np.expm1(y_test)
        
        # Calculate metrics on original scale
        train_metrics_original = {
            'rmse': np.sqrt(mean_squared_error(y_train_original, y_train_pred_original)),
            'mae': mean_absolute_error(y_train_original, y_train_pred_original),
            'r2': r2_score(y_train_original, y_train_pred_original),
            'mape': mean_absolute_percentage_error(y_train_original, y_train_pred_original) * 100
        }
        
        test_metrics_original = {
            'rmse': np.sqrt(mean_squared_error(y_test_original, y_test_pred_original)),
            'mae': mean_absolute_error(y_test_original, y_test_pred_original),
            'r2': r2_score(y_test_original, y_test_pred_original),
            'mape': mean_absolute_percentage_error(y_test_original, y_test_pred_original) * 100
        }
        
        print("\nMETRICS ON ORIGINAL SCALE (Current Bid in $):")
        print("=" * 70)
        print("TRAIN SET:")
        print(f"  RMSE: ${train_metrics_original['rmse']:.2f}")
        print(f"  MAE:  ${train_metrics_original['mae']:.2f}")
        print(f"  R²:   {train_metrics_original['r2']:.4f}")
        print(f"  MAPE: {train_metrics_original['mape']:.2f}%")
        
        print("\nTEST SET:")
        print(f"  RMSE: ${test_metrics_original['rmse']:.2f}")
        print(f"  MAE:  ${test_metrics_original['mae']:.2f}")
        print(f"  R²:   {test_metrics_original['r2']:.4f}")
        print(f"  MAPE: {test_metrics_original['mape']:.2f}%")
        print("=" * 70)
        
        # Print sample predictions
        print("\nSample predictions (first 10 test samples):")
        print(f"{'Actual ($)':>12} {'Predicted ($)':>15} {'Error ($)':>12}")
        print("-" * 42)
        for i in range(min(10, len(y_test_original))):
            actual = y_test_original.iloc[i]
            pred = y_test_pred_original[i]
            error = pred - actual
            print(f"${actual:>11.2f} ${pred:>14.2f} ${error:>11.2f}")
        
        # Save metrics
        metrics_path = self.output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'train_log_scale': train_metrics_log,
                'test_log_scale': test_metrics_log,
                'train_original_scale': train_metrics_original,
                'test_original_scale': test_metrics_original,
                'best_params': self.best_params
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return (train_metrics_log, test_metrics_log, train_metrics_original, test_metrics_original,
                y_train_pred_log, y_test_pred_log, y_train_pred_original, y_test_pred_original,
                y_train_original, y_test_original)
    
    def plot_results(self, y_train_original, y_train_pred_original, 
                     y_test_original, y_test_pred_original,
                     y_train_log, y_train_pred_log,
                     y_test_log, y_test_pred_log):
        """Create visualization plots"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Actual vs Predicted (Original Scale)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Train set
        axes[0].scatter(y_train_original, y_train_pred_original, alpha=0.3, s=10)
        axes[0].plot([y_train_original.min(), y_train_original.max()], 
                     [y_train_original.min(), y_train_original.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Current Bid ($)', fontsize=12)
        axes[0].set_ylabel('Predicted Current Bid ($)', fontsize=12)
        axes[0].set_title('Train Set - Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test set
        axes[1].scatter(y_test_original, y_test_pred_original, alpha=0.3, s=10, color='orange')
        axes[1].plot([y_test_original.min(), y_test_original.max()], 
                     [y_test_original.min(), y_test_original.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Current Bid ($)', fontsize=12)
        axes[1].set_ylabel('Predicted Current Bid ($)', fontsize=12)
        axes[1].set_title('Test Set - Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pred_path = self.output_dir / 'actual_vs_predicted_original.png'
        plt.savefig(pred_path, dpi=150, bbox_inches='tight')
        print(f"Actual vs Predicted plot (original scale) saved to: {pred_path}")
        plt.close()
        
        # 2. Residual plots (Original Scale)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Train residuals
        train_residuals = y_train_pred_original - y_train_original
        axes[0, 0].scatter(y_train_pred_original, train_residuals, alpha=0.3, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 0].set_xlabel('Predicted Current Bid ($)', fontsize=12)
        axes[0, 0].set_ylabel('Residuals ($)', fontsize=12)
        axes[0, 0].set_title('Train Set - Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test residuals
        test_residuals = y_test_pred_original - y_test_original
        axes[0, 1].scatter(y_test_pred_original, test_residuals, alpha=0.3, s=10, color='orange')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Current Bid ($)', fontsize=12)
        axes[0, 1].set_ylabel('Residuals ($)', fontsize=12)
        axes[0, 1].set_title('Test Set - Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Train residual histogram
        axes[1, 0].hist(train_residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals ($)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Train Set - Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Test residual histogram
        axes[1, 1].hist(test_residuals, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Residuals ($)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Test Set - Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        residual_path = self.output_dir / 'residual_plots.png'
        plt.savefig(residual_path, dpi=150, bbox_inches='tight')
        print(f"Residual plots saved to: {residual_path}")
        plt.close()
        
        # 3. Log scale predictions
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Train set (log scale)
        axes[0].scatter(y_train_log, y_train_pred_log, alpha=0.3, s=10, color='green')
        axes[0].plot([y_train_log.min(), y_train_log.max()], 
                     [y_train_log.min(), y_train_log.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Log(Current Bid + 1)', fontsize=12)
        axes[0].set_ylabel('Predicted Log(Current Bid + 1)', fontsize=12)
        axes[0].set_title('Train Set - Log Scale Predictions', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test set (log scale)
        axes[1].scatter(y_test_log, y_test_pred_log, alpha=0.3, s=10, color='purple')
        axes[1].plot([y_test_log.min(), y_test_log.max()], 
                     [y_test_log.min(), y_test_log.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Log(Current Bid + 1)', fontsize=12)
        axes[1].set_ylabel('Predicted Log(Current Bid + 1)', fontsize=12)
        axes[1].set_title('Test Set - Log Scale Predictions', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        log_pred_path = self.output_dir / 'actual_vs_predicted_log.png'
        plt.savefig(log_pred_path, dpi=150, bbox_inches='tight')
        print(f"Log scale predictions plot saved to: {log_pred_path}")
        plt.close()
    
    def plot_feature_importance(self, top_n=30):
        """Plot top features by absolute coefficient value"""
        print("\n" + "="*70)
        print(f"PLOTTING TOP {top_n} FEATURES")
        print("="*70)
        
        # Get coefficients
        coefficients = self.model.coef_
        
        # Create dataframe with feature names and coefficients
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient value
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        # Get top N
        top_features = importance_df.head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
        
        colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]
        
        ax.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Coefficient Magnitude\n(Green = Positive, Red = Negative)', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        importance_path = self.output_dir / 'feature_importance.png'
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {importance_path}")
        plt.close()
        
        # Save full coefficients
        importance_csv = self.output_dir / 'feature_coefficients.csv'
        importance_df.to_csv(importance_csv, index=False)
        print(f"Feature coefficients CSV saved to: {importance_csv}")
        
        # Print top features
        print(f"\nTop 10 positive coefficients (increase log_current_bid):")
        positive_features = importance_df[importance_df['coefficient'] > 0].head(10)
        for idx, row in positive_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.6f}")
        
        print(f"\nTop 10 negative coefficients (decrease log_current_bid):")
        negative_features = importance_df[importance_df['coefficient'] < 0].head(10)
        for idx, row in negative_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.6f}")
    
    def save_model(self):
        """Save trained model, scaler, and encoders"""
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        # Save full pipeline (model + scaler)
        pipeline_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        
        model_path = self.output_dir / 'lasso_regression.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        print(f"\nFull pipeline saved to: {model_path}")
        
        # Save feature names
        features_path = self.output_dir / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"Feature names saved to: {features_path}")
    
    def run_pipeline(self):
        """Execute full regression pipeline"""
        print("\n" + "="*70)
        print("LASSO REGRESSION PIPELINE - LOG TRANSFORMED TARGET")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Preprocess
        X, y = self.preprocess_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train model
        self.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate (includes transformation back to original scale)
        results = self.evaluate_model(X_train, y_train, X_test, y_test)
        (train_metrics_log, test_metrics_log, train_metrics_original, test_metrics_original,
         y_train_pred_log, y_test_pred_log, y_train_pred_original, y_test_pred_original,
         y_train_original, y_test_original) = results
        
        # Visualize
        self.plot_results(y_train_original, y_train_pred_original,
                         y_test_original, y_test_pred_original,
                         y_train, y_train_pred_log,
                         y_test, y_test_pred_log)
        
        # Plot feature importance
        self.plot_feature_importance(top_n=30)
        
        # Save model
        self.save_model()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutputs saved to: {self.output_dir}")


if __name__ == "__main__":
    # Initialize and run pipeline
    data_path = '/workspaces/maxsold/data/final_data/item_details/items_merged_20251201.parquet'
    
    pipeline = LassoRegressionPipeline(
        data_path=data_path,
        output_dir='model_pipeline/outputs_lasso_regression'
    )
    
    pipeline.run_pipeline()
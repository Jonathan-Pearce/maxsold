"""
XGBoost model pipeline for predicting current_bid on MaxSold auction items.
Target: current_bid
Features: All columns except current_bid, minimum_bid, and bid_count
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


class BidPredictionPipeline:
    """XGBoost pipeline for bid prediction"""
    
    def __init__(self, data_path, output_dir='model_pipeline/outputs'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_col = 'log_current_bid'
        self.exclude_cols = [
            'log_current_bid',
            'current_bid',  # Don't use actual bid amount
            'minimum_bid',  # Don't use minimum bid
            'bid_count',    # Don't use bid count
            'current_bid_le_10_binary',  # Drop binary target if exists
            'id', 'auction_id', 'totalBids', 'average_bids_per_lot',
            'bidding_extended', 'viewed'
        ]
        
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print("="*70)
        print("LOADING AND PREPARING DATA")
        print("="*70)
        
        print(f"\nLoading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        print(f"Original shape: {df.shape}")
        
        # Check if target column exists
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        print(f"\nTarget variable: {self.target_col}")
        print(f"Target statistics:")
        print(df[self.target_col].describe())
        
        # Remove rows where target is null
        before = len(df)
        df = df[df[self.target_col].notna()].copy()
        after = len(df)
        if before > after:
            print(f"\nRemoved {before - after:,} rows with null target values")
        
        # Identify feature columns
        self.feature_names = [col for col in df.columns 
                             if col not in self.exclude_cols]
        
        print(f"\nTotal columns: {len(df.columns)}")
        print(f"Excluded columns: {self.exclude_cols}")
        print(f"Feature columns: {len(self.feature_names)}")
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for XGBoost"""
        print("\n" + "="*70)
        print("PREPROCESSING FEATURES")
        print("="*70)
        
        X = df[self.feature_names].copy()
        y = df[self.target_col].copy()
        
        print(f"\nInitial feature matrix shape: {X.shape}")
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        print(f"\nNumeric columns: {len(numeric_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        print(f"Datetime columns: {len(datetime_cols)}")
        
        # Handle datetime columns - extract features
        # First convert any object columns that might be datetime strings
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try to convert to datetime
                    X[col] = pd.to_datetime(X[col], errors='coerce')
                except:
                    pass
        
        # Re-identify datetime columns after conversion
        datetime_cols = X.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
        
        if datetime_cols:
            print(f"\nExtracting features from {len(datetime_cols)} datetime columns...")
            for col in datetime_cols:
                try:
                    X[f'{col}_year'] = X[col].dt.year.fillna(0).astype(int)
                    X[f'{col}_month'] = X[col].dt.month.fillna(0).astype(int)
                    X[f'{col}_day'] = X[col].dt.day.fillna(0).astype(int)
                    X[f'{col}_hour'] = X[col].dt.hour.fillna(0).astype(int)
                    X[f'{col}_dayofweek'] = X[col].dt.dayofweek.fillna(-1).astype(int)
                    X[f'{col}_timestamp'] = X[col].astype('int64') // 10**9  # Unix timestamp
                except Exception as e:
                    print(f"Warning: Could not extract features from {col}: {e}")
                X = X.drop(columns=[col])
        
        # Update column lists after datetime processing
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Label encode categorical features
        if categorical_cols:
            print(f"\nEncoding {len(categorical_cols)} categorical columns...")
            categorical_sample = categorical_cols[:5]
            print(f"Sample categorical columns: {categorical_sample}")
            
            for col in categorical_cols:
                # Handle high cardinality - limit to top categories
                value_counts = X[col].value_counts()
                if len(value_counts) > 100:
                    top_categories = value_counts.head(100).index
                    X[col] = X[col].apply(lambda x: x if x in top_categories else 'OTHER')
                
                # Label encode
                le = LabelEncoder()
                X[col] = X[col].fillna('MISSING')
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle missing values in numeric columns
        if numeric_cols:
            print(f"\nHandling missing values in numeric columns...")
            null_counts = X[numeric_cols].isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            
            if len(cols_with_nulls) > 0:
                print(f"Columns with nulls: {len(cols_with_nulls)}")
                # Fill with median
                for col in cols_with_nulls.index:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"\nFinal feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTrain set: {X_train.shape[0]:,} samples ({100*(1-test_size):.0f}%)")
        print(f"Test set:  {X_test.shape[0]:,} samples ({100*test_size:.0f}%)")
        print(f"\nTrain target distribution:")
        print(y_train.describe())
        print(f"\nTest target distribution:")
        print(y_test.describe())
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n" + "="*70)
        print("TRAINING XGBOOST MODEL")
        print("="*70)
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'early_stopping_rounds': 10
        }
        
        print(f"\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create and train model
        self.model = xgb.XGBRegressor(**params)
        
        print(f"\nTraining model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Check if early stopping was used
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            print(f"Training complete! Best iteration: {self.model.best_iteration}")
        else:
            print("Training complete!")
        
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Predictions on log scale
        y_train_pred_log = self.model.predict(X_train)
        y_test_pred_log = self.model.predict(X_test)
        
        # Calculate metrics on log scale
        train_metrics_log = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_log)),
            'mae': mean_absolute_error(y_train, y_train_pred_log),
            'r2': r2_score(y_train, y_train_pred_log)
        };
        
        test_metrics_log = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_log)),
            'mae': mean_absolute_error(y_test, y_test_pred_log),
            'r2': r2_score(y_test, y_test_pred_log)
        };
        
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
        
        # Transform targets back to original scale
        y_train_original = np.expm1(y_train)
        y_test_original = np.expm1(y_test)
        
        # Calculate metrics on original scale
        train_metrics_original = {
            'rmse': np.sqrt(mean_squared_error(y_train_original, y_train_pred_original)),
            'mae': mean_absolute_error(y_train_original, y_train_pred_original),
            'r2': r2_score(y_train_original, y_train_pred_original)
        };
        
        test_metrics_original = {
            'rmse': np.sqrt(mean_squared_error(y_test_original, y_test_pred_original)),
            'mae': mean_absolute_error(y_test_original, y_test_pred_original),
            'r2': r2_score(y_test_original, y_test_pred_original)
        };
        
        print("\nMETRICS ON ORIGINAL SCALE (Current Bid in $):")
        print("=" * 70)
        print("TRAIN SET:")
        print(f"  RMSE: ${train_metrics_original['rmse']:,.2f}")
        print(f"  MAE:  ${train_metrics_original['mae']:,.2f}")
        print(f"  R²:   {train_metrics_original['r2']:.4f}")
        
        print("\nTEST SET:")
        print(f"  RMSE: ${test_metrics_original['rmse']:,.2f}")
        print(f"  MAE:  ${test_metrics_original['mae']:,.2f}")
        print(f"  R²:   {test_metrics_original['r2']:.4f}")
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
        
        # Get best iteration if available
        best_iter = None
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            best_iter = int(self.model.best_iteration)
        
        # Save metrics
        metrics_path = self.output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'train_log_scale': train_metrics_log,
                'test_log_scale': test_metrics_log,
                'train_original_scale': train_metrics_original,
                'test_original_scale': test_metrics_original,
                'best_iteration': best_iter,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return (train_metrics_log, test_metrics_log, train_metrics_original, test_metrics_original,
                y_train_pred_log, y_test_pred_log, y_train_pred_original, y_test_pred_original,
                y_train_original, y_test_original)
    
    def plot_results(self, y_train_original, y_train_pred_original, 
                     y_test_original, y_test_pred_original,
                     y_train_log, y_train_pred_log,
                     y_test, y_test_pred_log):
        """Create visualization plots"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
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
        
        # Feature importance plot
        self.plot_feature_importance()
    
    def plot_feature_importance(self, top_n=20):
        """Plot top feature importances"""
        importance_df = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (XGBoost)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        importance_path = self.output_dir / 'feature_importance.png'
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {importance_path}")
        plt.close()
        
        # Save full importance scores
        importance_csv = self.output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_csv, index=False)
        print(f"Feature importance CSV saved to: {importance_csv}")
    
    def save_model(self):
        """Save trained model and encoders"""
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        # Save XGBoost model using pickle (sklearn wrapper)
        model_path = self.output_dir / 'xgboost_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to: {model_path}")
        
        # Also save using XGBoost's native format (booster only)
        model_json_path = self.output_dir / 'xgboost_model.json'
        self.model.get_booster().save_model(str(model_json_path))
        print(f"Model (JSON format) saved to: {model_json_path}")
        
        # Save label encoders
        encoders_path = self.output_dir / 'label_encoders.pkl'
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"Label encoders saved to: {encoders_path}")
        
        # Save feature names
        features_path = self.output_dir / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"Feature names saved to: {features_path}")
    
    def run_pipeline(self):
        """Execute full pipeline"""
        print("\n" + "="*70)
        print("XGBOOST BID PREDICTION PIPELINE")
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
        
        # Evaluate
        results = self.evaluate_model(X_train, y_train, X_test, y_test)
        (train_metrics_log, test_metrics_log, train_metrics_original, test_metrics_original,
         y_train_pred_log, y_test_pred_log, y_train_pred_original, y_test_pred_original,
         y_train_original, y_test_original) = results
        
        # Visualize
        self.plot_results(y_train_original, y_train_pred_original,
                         y_test_original, y_test_pred_original,
                         y_train, y_train_pred_log,
                         y_test, y_test_pred_log)
        
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
    
    pipeline = BidPredictionPipeline(
        data_path=data_path,
        output_dir='model_pipeline/outputs'
    )
    
    pipeline.run_pipeline()
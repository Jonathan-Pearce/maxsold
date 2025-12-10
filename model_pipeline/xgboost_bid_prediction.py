"""
LightGBM model pipeline for predicting current_bid on MaxSold auction items.
Target: current_bid
Features: All columns except current_bid, minimum_bid, and bid_count
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


class BidPredictionPipeline:
    """LightGBM pipeline for bid prediction"""
    
    def __init__(self, data_path, output_dir='model_pipeline/outputs'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_col = 'current_bid'
        self.exclude_cols = ['current_bid', 'minimum_bid', 'bid_count']
        
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
        """Preprocess features for LightGBM"""
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
        if datetime_cols:
            print(f"\nExtracting features from {len(datetime_cols)} datetime columns...")
            for col in datetime_cols:
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                X = X.drop(columns=[col])
        
        # Update column lists after datetime processing
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # LightGBM can handle categorical features natively, but we'll use label encoding
        # to reduce memory usage for high cardinality features
        categorical_features = []
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
                categorical_features.append(col)
        
        self.categorical_features = categorical_features
        
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
        print(f"Categorical features: {len(self.categorical_features)}")
        
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
        """Train LightGBM model"""
        print("\n" + "="*70)
        print("TRAINING LIGHTGBM MODEL")
        print("="*70)
        
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        print(f"\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create and train model
        self.model = lgb.LGBMRegressor(**params)
        
        print(f"\nTraining model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        print(f"Training complete! Best iteration: {self.model.best_iteration_}")
        
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }
        
        print("\nTRAIN METRICS:")
        print(f"  RMSE: ${train_metrics['rmse']:,.2f}")
        print(f"  MAE:  ${train_metrics['mae']:,.2f}")
        print(f"  R²:   {train_metrics['r2']:.4f}")
        
        print("\nTEST METRICS:")
        print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
        print(f"  MAE:  ${test_metrics['mae']:,.2f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        
        # Save metrics
        metrics_path = self.output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'train': train_metrics,
                'test': test_metrics,
                'best_iteration': int(self.model.best_iteration_) if self.model.best_iteration_ else None,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return train_metrics, test_metrics, y_train_pred, y_test_pred
    
    def plot_results(self, y_train, y_train_pred, y_test, y_test_pred):
        """Create visualization plots"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted (Train)
        axes[0, 0].scatter(y_train, y_train_pred, alpha=0.3, s=10)
        axes[0, 0].plot([y_train.min(), y_train.max()], 
                        [y_train.min(), y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Bid ($)')
        axes[0, 0].set_ylabel('Predicted Bid ($)')
        axes[0, 0].set_title('Train: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Actual vs Predicted (Test)
        axes[0, 1].scatter(y_test, y_test_pred, alpha=0.3, s=10)
        axes[0, 1].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Bid ($)')
        axes[0, 1].set_ylabel('Predicted Bid ($)')
        axes[0, 1].set_title('Test: Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals (Train)
        train_residuals = y_train - y_train_pred
        axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.3, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Bid ($)')
        axes[1, 0].set_ylabel('Residuals ($)')
        axes[1, 0].set_title('Train: Residual Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals (Test)
        test_residuals = y_test - y_test_pred
        axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.3, s=10)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted Bid ($)')
        axes[1, 1].set_ylabel('Residuals ($)')
        axes[1, 1].set_title('Test: Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'predictions_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPredictions plot saved to: {plot_path}")
        plt.close()
        
        # Feature importance plot
        self.plot_feature_importance()
    
    def plot_feature_importance(self, top_n=20):
        """Plot top feature importances"""
        importance_df = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (LightGBM)')
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
        
        # Save LightGBM model
        model_path = self.output_dir / 'lightgbm_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to: {model_path}")
        
        # Also save as text for interpretability
        model_txt_path = self.output_dir / 'lightgbm_model.txt'
        self.model.booster_.save_model(str(model_txt_path))
        print(f"Model (text format) saved to: {model_txt_path}")
        
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
        print("LIGHTGBM BID PREDICTION PIPELINE")
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
        train_metrics, test_metrics, y_train_pred, y_test_pred = self.evaluate_model(
            X_train, y_train, X_test, y_test
        )
        
        # Visualize
        self.plot_results(y_train, y_train_pred, y_test, y_test_pred)
        
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
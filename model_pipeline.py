"""
Machine Learning Pipeline for MaxSold Current Bid Prediction
Target: current_bid
Model: XGBoost Regression
Excluded Features: bid_count
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MaxSoldModelPipeline:
    """Complete ML pipeline for MaxSold bid prediction"""
    
    def __init__(self, data_path, model_dir='data/models', output_dir='data/models/output'):
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.label_encoders = {}
        self.results = {}
        
    def load_data(self):
        """Load the dataset from parquet file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n" + "="*80)
        print("DATA EXPLORATION")
        print("="*80)
        
        print(f"\nDataset Info:")
        print(f"  Total rows: {len(self.df):,}")
        print(f"  Total columns: {len(self.df.columns)}")
        
        print(f"\nTarget Variable (current_bid) Statistics:")
        print(self.df['current_bid'].describe())
        
        print(f"\nMissing values:")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            print(missing)
        else:
            print("  No missing values found")
        
        print(f"\nData types:")
        print(self.df.dtypes.value_counts())
        
        return self
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        # Make a copy
        df = self.df.copy()
        
        # Remove bid_count as requested
        if 'bid_count' in df.columns:
            print(f"\nRemoving 'bid_count' column as requested")
            df = df.drop(columns=['bid_count'])
        
        # Check for target variable
        if 'current_bid' not in df.columns:
            raise ValueError("Target column 'current_bid' not found in dataset")
        
        # Remove rows with missing target
        initial_rows = len(df)
        df = df.dropna(subset=['current_bid'])
        print(f"Removed {initial_rows - len(df)} rows with missing target")
        
        # Separate features and target
        y = df['current_bid']
        X = df.drop(columns=['current_bid', 'log_current_bid', 'current_bid_le_10_binary'])
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        print(f"\nColumn types identified:")
        print(f"  Numeric: {len(numeric_cols)}")
        print(f"  Categorical: {len(categorical_cols)}")
        print(f"  Datetime: {len(datetime_cols)}")
        
        # Handle datetime columns - extract useful features
        for col in datetime_cols:
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X[f'{col}_hour'] = X[col].dt.hour
            X = X.drop(columns=[col])
            print(f"  Extracted features from datetime column: {col}")
        
        # Handle categorical columns - label encoding
        for col in categorical_cols:
            # Handle missing values
            X[col] = X[col].fillna('MISSING')
            
            # Limit cardinality - keep top N categories
            if X[col].nunique() > 50:
                top_categories = X[col].value_counts().head(50).index
                X[col] = X[col].apply(lambda x: x if x in top_categories else 'OTHER')
                print(f"  Reduced cardinality for {col} to top 50 categories")
            
            # Label encode
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle numeric columns - fill missing with median
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"  Filled {col} missing values with median: {median_val:.2f}")
        
        # Handle any remaining infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"\nFinal preprocessed shape: {X.shape}")
        print(f"Features: {X.shape[1]}")
        
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "="*80)
        print("TRAIN/TEST SPLIT")
        print("="*80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples")
        print(f"Test size: {test_size*100:.0f}%")
        
        return self
    
    def train_model(self, params=None):
        """Train XGBoost regression model"""
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
        
        print(f"Model parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print(f"\nTraining XGBoost model...")
        self.model = xgb.XGBRegressor(**params)
        
        # Train with evaluation set
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=50
        )
        
        print("Model training complete!")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_metrics = {
            'RMSE': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'MAE': mean_absolute_error(self.y_train, y_train_pred),
            'R2': r2_score(self.y_train, y_train_pred),
            'MAPE': np.mean(np.abs((self.y_train - y_train_pred) / self.y_train)) * 100
        }
        
        test_metrics = {
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'MAE': mean_absolute_error(self.y_test, y_test_pred),
            'R2': r2_score(self.y_test, y_test_pred),
            'MAPE': np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100
        }
        
        print("\nTraining Set Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nTest Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Store results
        self.results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        return self
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        print("\nGenerating feature importance plot...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Save full importance to CSV
        csv_path = self.output_dir / 'feature_importance.csv'
        feature_importance.to_csv(csv_path, index=False)
        print(f"  Saved feature importance to: {csv_path}")
        
        # Plot top N features
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        
        plot_path = self.output_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to: {plot_path}")
        
        return self
    
    def plot_predictions(self):
        """Plot actual vs predicted values"""
        print("\nGenerating prediction plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training set
        axes[0].scatter(self.y_train, self.results['y_train_pred'], alpha=0.5, s=10)
        axes[0].plot([self.y_train.min(), self.y_train.max()], 
                     [self.y_train.min(), self.y_train.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Current Bid', fontsize=12)
        axes[0].set_ylabel('Predicted Current Bid', fontsize=12)
        axes[0].set_title('Training Set: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test set
        axes[1].scatter(self.y_test, self.results['y_test_pred'], alpha=0.5, s=10)
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Current Bid', fontsize=12)
        axes[1].set_ylabel('Predicted Current Bid', fontsize=12)
        axes[1].set_title('Test Set: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'predictions_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to: {plot_path}")
        
        return self
    
    def plot_residuals(self):
        """Plot residual analysis"""
        print("\nGenerating residual plots...")
        
        # Calculate residuals
        train_residuals = self.y_train - self.results['y_train_pred']
        test_residuals = self.y_test - self.results['y_test_pred']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Training residuals scatter
        axes[0, 0].scatter(self.results['y_train_pred'], train_residuals, alpha=0.5, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
        axes[0, 0].set_ylabel('Residuals', fontsize=12)
        axes[0, 0].set_title('Training Set: Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test residuals scatter
        axes[0, 1].scatter(self.results['y_test_pred'], test_residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Test Set: Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training residuals distribution
        axes[1, 0].hist(train_residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Training Set: Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Test residuals distribution
        axes[1, 1].hist(test_residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Residuals', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Test Set: Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'residual_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to: {plot_path}")
        
        return self
    
    def plot_learning_curve(self):
        """Plot training learning curve"""
        print("\nGenerating learning curve...")
        
        results = self.model.evals_result()
        train_rmse = [np.sqrt(x) for x in results['validation_0']['rmse']]
        test_rmse = [np.sqrt(x) for x in results['validation_1']['rmse']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train_rmse, label='Training RMSE', linewidth=2)
        ax.plot(test_rmse, label='Test RMSE', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('XGBoost Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'learning_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to: {plot_path}")
        
        return self
    
    def plot_error_distribution(self):
        """Plot error distribution analysis"""
        print("\nGenerating error distribution plots...")
        
        # Calculate percentage errors, avoiding division by zero
        # Only calculate for non-zero actual values
        train_mask = self.y_train != 0
        test_mask = self.y_test != 0
        
        train_pct_error = ((self.results['y_train_pred'][train_mask] - self.y_train[train_mask]) / self.y_train[train_mask] * 100)
        test_pct_error = ((self.results['y_test_pred'][test_mask] - self.y_test[test_mask]) / self.y_test[test_mask] * 100)
        
        # Remove any remaining infinite values
        train_pct_error = train_pct_error[np.isfinite(train_pct_error)]
        test_pct_error = test_pct_error[np.isfinite(test_pct_error)]
        
        # Cap extreme values for better visualization (keep within -500% to 500%)
        train_pct_error = np.clip(train_pct_error, -500, 500)
        test_pct_error = np.clip(test_pct_error, -500, 500)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training error distribution
        axes[0].hist(train_pct_error, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        axes[0].set_xlabel('Percentage Error (%) [capped at ±500%]', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Training Set: Percentage Error Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test error distribution
        axes[1].hist(test_pct_error, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        axes[1].set_xlabel('Percentage Error (%) [capped at ±500%]', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Test Set: Percentage Error Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'error_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to: {plot_path}")
        
        return self
    
    def save_model(self):
        """Save the trained model and artifacts"""
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)
        
        # Save model
        model_path = self.model_dir / 'xgboost_model.pkl'
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save label encoders
        if self.label_encoders:
            encoders_path = self.model_dir / 'label_encoders.pkl'
            joblib.dump(self.label_encoders, encoders_path)
            print(f"Label encoders saved to: {encoders_path}")
        
        # Save feature names
        features_path = self.model_dir / 'feature_names.pkl'
        joblib.dump(self.feature_names, features_path)
        print(f"Feature names saved to: {features_path}")
        
        # Save metrics summary
        metrics_path = self.output_dir / 'metrics_summary.txt'
        with open(metrics_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MAXSOLD CURRENT BID PREDICTION MODEL - EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: XGBoost Regression\n")
            f.write(f"Target: current_bid\n")
            f.write(f"Excluded Features: bid_count\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TRAINING SET METRICS:\n")
            f.write("-" * 40 + "\n")
            for metric, value in self.results['train_metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nTEST SET METRICS:\n")
            f.write("-" * 40 + "\n")
            for metric, value in self.results['test_metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nDATASET INFO:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total samples: {len(self.df):,}\n")
            f.write(f"  Training samples: {len(self.X_train):,}\n")
            f.write(f"  Test samples: {len(self.X_test):,}\n")
            f.write(f"  Number of features: {len(self.feature_names)}\n")
        
        print(f"Metrics summary saved to: {metrics_path}")
        
        return self
    
    def run_full_pipeline(self):
        """Execute the complete ML pipeline"""
        print("\n" + "="*80)
        print("MAXSOLD ML PIPELINE - CURRENT BID PREDICTION")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        try:
            self.load_data()
            self.explore_data()
            self.preprocess_data()
            self.split_data()
            self.train_model()
            self.evaluate_model()
            self.plot_feature_importance()
            self.plot_predictions()
            self.plot_residuals()
            self.plot_learning_curve()
            self.plot_error_distribution()
            self.save_model()
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nModel and results saved to:")
            print(f"  Model directory: {self.model_dir}")
            print(f"  Output directory: {self.output_dir}")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Pipeline failed with exception:")
            print(f"{'='*80}")
            print(f"{str(e)}")
            raise


def main():
    """Main execution function"""
    # Configuration
    DATA_PATH = 'data/final_data/maxsold_final_dataset.parquet'
    MODEL_DIR = 'data/models'
    OUTPUT_DIR = 'data/models/output'
    
    # Initialize and run pipeline
    pipeline = MaxSoldModelPipeline(
        data_path=DATA_PATH,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )
    
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()

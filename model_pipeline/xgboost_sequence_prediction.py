"""
XGBoost Sequential Bid Price Prediction

This script uses XGBoost on engineered sequence features to predict winning auction prices.
Unlike the LSTM approach, this creates aggregate statistics from the bid sequence rather than
modeling the sequence directly.

This is computationally cheaper and often performs comparably well for tabular data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostSequencePipeline:
    """XGBoost pipeline using sequence aggregate features"""
    
    def __init__(self, data_path, output_dir='model_pipeline/outputs_sequential_xgb'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        
    def load_and_engineer_features(self):
        """Load data and create item-level aggregate features"""
        print("="*80)
        print("LOADING DATA AND ENGINEERING SEQUENCE FEATURES")
        print("="*80)
        
        print(f"\nLoading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(df):,} rows")
        print(f"Unique items: {df['item_id'].nunique():,}")
        
        # Sort by item and bid number
        df = df.sort_values(['item_id', 'bid_number']).reset_index(drop=True)
        
        print("\nEngineering item-level features from bid sequences...")
        
        # Group by item and create features
        item_features = []
        
        for item_id, item_df in df.groupby('item_id'):
            item_df = item_df.sort_values('bid_number')
            
            # Basic statistics
            features = {
                'item_id': item_id,
                'auction_id': item_df['auction_id'].iloc[0],
                
                # Target: maximum bid (winning price)
                'target_max_bid': item_df['amount'].max(),
                
                # Sequence length
                'total_bids': len(item_df),
                
                # Bid amount statistics
                'bid_amount_min': item_df['amount'].min(),
                'bid_amount_max': item_df['amount'].max(),
                'bid_amount_mean': item_df['amount'].mean(),
                'bid_amount_median': item_df['amount'].median(),
                'bid_amount_std': item_df['amount'].std(),
                'bid_amount_range': item_df['amount'].max() - item_df['amount'].min(),
                'bid_amount_q25': item_df['amount'].quantile(0.25),
                'bid_amount_q75': item_df['amount'].quantile(0.75),
                
                # First and last bid amounts
                'first_bid_amount': item_df['amount'].iloc[0],
                'last_bid_amount': item_df['amount'].iloc[-1],
                'second_bid_amount': item_df['amount'].iloc[1] if len(item_df) > 1 else item_df['amount'].iloc[0],
                'third_bid_amount': item_df['amount'].iloc[2] if len(item_df) > 2 else item_df['amount'].iloc[-1],
                
                # Bid increment statistics
                'bid_increment_mean': item_df['bid_increment'].mean() if 'bid_increment' in item_df else 0,
                'bid_increment_median': item_df['bid_increment'].median() if 'bid_increment' in item_df else 0,
                'bid_increment_max': item_df['bid_increment'].max() if 'bid_increment' in item_df else 0,
                'bid_increment_std': item_df['bid_increment'].std() if 'bid_increment' in item_df else 0,
                
                # Proxy bid statistics
                'proxy_bid_count': item_df['isproxy'].sum() if 'isproxy' in item_df else 0,
                'manual_bid_count': (~item_df['isproxy']).sum() if 'isproxy' in item_df else len(item_df),
                'proxy_bid_ratio': item_df['isproxy'].mean() if 'isproxy' in item_df else 0,
                
                # Temporal statistics
                'bidding_duration_hours': item_df['hours_since_first_bid'].max() if 'hours_since_first_bid' in item_df else 0,
                'bids_per_hour': len(item_df) / (item_df['hours_since_first_bid'].max() + 0.01) if 'hours_since_first_bid' in item_df else 0,
                
                # Early bid statistics (first 25% of bids)
                'early_bid_mean': item_df['amount'].iloc[:max(1, len(item_df)//4)].mean(),
                'early_bid_growth': (item_df['amount'].iloc[max(1, len(item_df)//4) - 1] - 
                                   item_df['amount'].iloc[0]) if len(item_df) > 1 else 0,
                
                # Late bid statistics (last 25% of bids)
                'late_bid_mean': item_df['amount'].iloc[-max(1, len(item_df)//4):].mean(),
                'late_bid_growth': (item_df['amount'].iloc[-1] - 
                                  item_df['amount'].iloc[-max(1, len(item_df)//4)]) if len(item_df) > 1 else 0,
                
                # Bid velocity (how fast bids are increasing)
                'bid_velocity': (item_df['amount'].iloc[-1] - item_df['amount'].iloc[0]) / max(1, len(item_df)),
                
                # Momentum features (recent trend)
                'momentum_last_5': (item_df['amount'].iloc[-1] - item_df['amount'].iloc[-min(5, len(item_df))]) if len(item_df) > 1 else 0,
                'momentum_last_10': (item_df['amount'].iloc[-1] - item_df['amount'].iloc[-min(10, len(item_df))]) if len(item_df) > 1 else 0,
            }
            
            item_features.append(features)
        
        # Create dataframe
        features_df = pd.DataFrame(item_features)
        
        print(f"\nCreated {len(features_df)} item records with {len(features_df.columns)-3} features")
        print(f"(excluding item_id, auction_id, target_max_bid)")
        
        return features_df
    
    def prepare_data(self, df):
        """Prepare features and target"""
        print("\n" + "="*80)
        print("PREPARING DATA FOR MODELING")
        print("="*80)
        
        # Separate features and target
        exclude_cols = ['item_id', 'auction_id', 'target_max_bid']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_names].copy()
        y = df['target_max_bid'].copy()
        item_ids = df['item_id'].copy()
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nFeatures ({len(self.feature_names)}):")
        for i, col in enumerate(self.feature_names, 1):
            print(f"  {i}. {col}")
        
        print(f"\nTarget statistics:")
        print(f"  Mean: ${y.mean():.2f}")
        print(f"  Median: ${y.median():.2f}")
        print(f"  Std: ${y.std():.2f}")
        print(f"  Min: ${y.min():.2f}")
        print(f"  Max: ${y.max():.2f}")
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y, item_ids
    
    def split_data(self, X, y, item_ids, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "="*80)
        print("SPLITTING DATA")
        print("="*80)
        
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, item_ids, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test, ids_train, ids_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
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
            verbose=50
        )
        
        print(f"\nTraining complete!")
    
    def evaluate_model(self, X, y, split_name='Test'):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print(f"{split_name.upper()} SET EVALUATION")
        print("="*80)
        
        # Predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        
        # Calculate percentage within ranges
        abs_errors = np.abs(y_pred - y)
        within_5_pct = np.mean(abs_errors / y <= 0.05) * 100
        within_10_pct = np.mean(abs_errors / y <= 0.10) * 100
        within_20_pct = np.mean(abs_errors / y <= 0.20) * 100
        
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
        
        return metrics, y_pred
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save to CSV
        importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        print(f"\nFeature importance plot saved to: {self.output_dir / 'feature_importance.png'}")
        print(f"Feature importance data saved to: {self.output_dir / 'feature_importance.csv'}")
    
    def plot_predictions(self, y_true, y_pred, split_name='Test'):
        """Plot prediction visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Scatter plot
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Winning Bid ($)', fontsize=11)
        ax.set_ylabel('Predicted Winning Bid ($)', fontsize=11)
        ax.set_title(f'{split_name} Set: Predicted vs Actual', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals
        ax = axes[0, 1]
        residuals = y_pred - y_true
        ax.scatter(y_true, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual Winning Bid ($)', fontsize=11)
        ax.set_ylabel('Residual ($)', fontsize=11)
        ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax = axes[1, 0]
        errors = np.abs(y_pred - y_true)
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
        pct_errors = np.abs((y_pred - y_true) / y_true) * 100
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
    
    def save_model_and_results(self, train_metrics, test_metrics):
        """Save model and results"""
        # Save model
        model_path = self.output_dir / 'xgboost_sequence_model.json'
        self.model.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'model_type': 'xgboost_sequence',
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save feature names
        features_path = self.output_dir / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"Feature names saved to: {features_path}")
    
    def run_pipeline(self):
        """Execute full pipeline"""
        print("\n" + "="*80)
        print("XGBOOST SEQUENTIAL BID PREDICTION PIPELINE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Load and engineer features
        df = self.load_and_engineer_features()
        
        # 2. Prepare data
        X, y, item_ids = self.prepare_data(df)
        
        # 3. Split data
        X_train, X_test, y_train, y_test, ids_train, ids_test = self.split_data(X, y, item_ids)
        
        # 4. Train model
        self.train_model(X_train, y_train, X_test, y_test)
        
        # 5. Evaluate
        train_metrics, train_preds = self.evaluate_model(X_train, y_train, 'Train')
        test_metrics, test_preds = self.evaluate_model(X_test, y_test, 'Test')
        
        # 6. Visualizations
        self.plot_feature_importance(top_n=20)
        self.plot_predictions(y_train, train_preds, 'Train')
        self.plot_predictions(y_test, test_preds, 'Test')
        
        # 7. Save results
        self.save_model_and_results(train_metrics, test_metrics)
        
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
    DATA_PATH = '/workspaces/maxsold/data/engineered_data/bid_history/bid_history_engineered_20251201.parquet'
    OUTPUT_DIR = 'model_pipeline/outputs_sequential_xgb'
    
    # Initialize and run pipeline
    pipeline = XGBoostSequencePipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
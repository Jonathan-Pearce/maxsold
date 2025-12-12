"""
XGBoost binary classification model pipeline for predicting if current_bid <= $10 on MaxSold auction items.
Target: current_bid_le_10_binary (1 if current_bid <= $10, 0 otherwise)
Features: All columns except current_bid, minimum_bid, bid_count, auction_id, id, current_bid_le_10_binary
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


class BidClassificationPipeline:
    """XGBoost pipeline for binary bid classification"""
    
    def __init__(self, data_path, output_dir='model_pipeline/outputs'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_col = 'current_bid_le_10_binary'
        self.exclude_cols = ['current_bid', 'id', 'auction_id', 'current_bid_le_10_binary',
                              'minimum_bid', 'bid_count', 'totalBids', 'average_bids_per_lot',
                              'bidding_extended', 'viewed']  # Exclude columns not to be used as features
        
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
        """Train XGBoost classification model"""
        print("\n" + "="*70)
        print("TRAINING XGBOOST CLASSIFIER")
        print("="*70)
        
        # Calculate class weights for imbalanced data
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1.0
        print(f"\nClass distribution in training set:")
        print(class_counts)
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # XGBoost parameters for binary classification
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'early_stopping_rounds': 10
        }
        
        print(f"\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create and train model
        self.model = xgb.XGBClassifier(**params)
        
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
        """Evaluate classification model performance"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Predictions (class labels)
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Prediction probabilities
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_pred_proba)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
        }
        
        print("\nTRAIN METRICS:")
        print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall:    {train_metrics['recall']:.4f}")
        print(f"  F1 Score:  {train_metrics['f1']:.4f}")
        print(f"  ROC AUC:   {train_metrics['roc_auc']:.4f}")
        
        print("\nTEST METRICS:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
        
        # Confusion matrices
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        
        print("\nTRAIN CONFUSION MATRIX:")
        print(train_cm)
        
        print("\nTEST CONFUSION MATRIX:")
        print(test_cm)
        
        # Classification reports
        print("\nTEST CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_test_pred, target_names=['> $10', '<= $10']))
        
        # Get best iteration if available
        best_iter = None
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            best_iter = int(self.model.best_iteration)
        
        # Save metrics
        metrics_path = self.output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'train': train_metrics,
                'test': test_metrics,
                'train_confusion_matrix': train_cm.tolist(),
                'test_confusion_matrix': test_cm.tolist(),
                'best_iteration': best_iter,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return train_metrics, test_metrics, y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba
    
    def plot_results(self, y_train, y_train_pred, y_test, y_test_pred, y_train_pred_proba, y_test_pred_proba):
        """Create visualization plots for classification"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Train Confusion Matrix
        train_cm = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title('Train: Confusion Matrix')
        axes[0, 0].set_xticklabels(['> $10', '<= $10'])
        axes[0, 0].set_yticklabels(['> $10', '<= $10'])
        
        # Plot 2: Test Confusion Matrix
        test_cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        axes[0, 1].set_title('Test: Confusion Matrix')
        axes[0, 1].set_xticklabels(['> $10', '<= $10'])
        axes[0, 1].set_yticklabels(['> $10', '<= $10'])
        
        # Plot 3: ROC Curve
        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_pred_proba)
        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_pred_proba)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        axes[1, 0].plot(train_fpr, train_tpr, label=f'Train (AUC = {train_auc:.3f})', linewidth=2)
        axes[1, 0].plot(test_fpr, test_tpr, label=f'Test (AUC = {test_auc:.3f})', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Precision-Recall Curve
        train_precision, train_recall, _ = precision_recall_curve(y_train, y_train_pred_proba)
        test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
        
        axes[1, 1].plot(train_recall, train_precision, label='Train', linewidth=2)
        axes[1, 1].plot(test_recall, test_precision, label='Test', linewidth=2)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'classification_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nClassification plots saved to: {plot_path}")
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
        plt.title(f'Top {top_n} Feature Importances (XGBoost Classifier)')
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
        
        # Save XGBoost model
        model_path = self.output_dir / 'xgboost_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to: {model_path}")
        
        # Also save in XGBoost's native format
        model_json_path = self.output_dir / 'xgboost_model.json'
        self.model.save_model(str(model_json_path))
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
        """Execute full classification pipeline"""
        print("\n" + "="*70)
        print("XGBOOST BINARY CLASSIFICATION PIPELINE")
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
        train_metrics, test_metrics, y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba = self.evaluate_model(
            X_train, y_train, X_test, y_test
        )
        
        # Visualize
        self.plot_results(y_train, y_train_pred, y_test, y_test_pred, y_train_pred_proba, y_test_pred_proba)
        
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
    
    pipeline = BidClassificationPipeline(
        data_path=data_path,
        output_dir='model_pipeline/outputs_classifier'
    )
    
    pipeline.run_pipeline()
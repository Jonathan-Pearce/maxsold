"""
Lasso Logistic Regression Binary Classification Pipeline for Bid Prediction

This script trains a Lasso (L1-regularized) logistic regression model to predict
whether an item's current_bid is <= $10.

Target variable: current_bid_le_10_binary
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report, auc
)


class LassoLogisticRegressionPipeline:
    """Pipeline for Lasso logistic regression binary classification"""
    
    def __init__(self, data_path, output_dir='model_pipeline/outputs_lasso'):
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
        if 'current_bid_le_10_binary' not in df.columns:
            print("\nCreating target variable 'current_bid_le_10_binary'...")
            if 'current_bid' in df.columns:
                df['current_bid_le_10_binary'] = (df['current_bid'] <= 10).astype(int)
                print(f"Target created from current_bid column")
            else:
                raise ValueError("Cannot create target: 'current_bid' column not found")
        
        # Check target distribution
        target_dist = df['current_bid_le_10_binary'].value_counts()
        print(f"\nTarget distribution:")
        for val, count in target_dist.items():
            pct = 100 * count / len(df)
            label = "<= $10" if val == 1 else "> $10"
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for modeling"""
        print("\n" + "="*70)
        print("PREPROCESSING FEATURES")
        print("="*70)
        
        # Create target
        if 'current_bid_le_10_binary' not in df.columns:
            raise ValueError("Target variable 'current_bid_le_10_binary' not found")
        
        y = df['current_bid_le_10_binary'].copy()
        
        # Drop target and related columns
        columns_to_drop = [
            'current_bid', 'id', 'auction_id', 'current_bid_le_10_binary',
                              'minimum_bid', 'bid_count', 'totalBids', 'average_bids_per_lot',
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
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]:,} samples")
        print(f"Test set:  {X_test.shape[0]:,} samples")
        
        # Check class distribution in train/test
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        print(f"\nTrain class distribution:")
        print(f"  Class 0 (> $10): {train_dist[0]:.1%}")
        print(f"  Class 1 (<= $10): {train_dist[1]:.1%}")
        
        print(f"\nTest class distribution:")
        print(f"  Class 0 (> $10): {test_dist[0]:.1%}")
        print(f"  Class 1 (<= $10): {test_dist[1]:.1%}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train Lasso logistic regression with hyperparameter tuning"""
        print("\n" + "="*70)
        print("TRAINING LASSO LOGISTIC REGRESSION")
        print("="*70)
        
        # Standardize features (important for L1 regularization)
        print("\nStandardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Features standardized (mean=0, std=1)")
        
        # Calculate class weights for imbalanced data
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1]
        
        print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            #'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength (inverse)
            'C': [0.001, 0.005],  # Regularization strength (inverse)
            'class_weight': ['balanced', None]
        }
        
        print(f"\nHyperparameter search space:")
        print(f"  C values: {param_grid['C']}")
        print(f"  class_weight: {param_grid['class_weight']}")
        
        # Create base model with L1 penalty (Lasso)
        base_model = LogisticRegression(
            penalty='l1',
            solver='saga',  # 'saga' supports L1 penalty
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            verbose=3
        )
        
        # Perform grid search with cross-validation
        print(f"\nPerforming 5-fold cross-validation...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=3
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\n{'='*70}")
        print(f"Best hyperparameters:")
        print(f"  C: {self.best_params['C']}")
        print(f"  class_weight: {self.best_params['class_weight']}")
        print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"{'='*70}")
        
        # Count non-zero coefficients (feature selection by Lasso)
        n_nonzero = np.sum(self.model.coef_ != 0)
        n_total = len(self.model.coef_[0])
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
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        y_train_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_test_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics for both train and test
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_pred_proba)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
        }
        
        # Print results
        print("\nTRAIN SET METRICS:")
        print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall:    {train_metrics['recall']:.4f}")
        print(f"  F1 Score:  {train_metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {train_metrics['roc_auc']:.4f}")
        
        print("\nTEST SET METRICS:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        
        # Classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT (TEST SET)")
        print("="*70)
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['> $10', '<= $10']))
        
        # Save metrics
        metrics_path = self.output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'train': train_metrics,
                'test': test_metrics,
                'best_params': self.best_params
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        return train_metrics, test_metrics, y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba
    
    def plot_results(self, y_train, y_train_pred, y_test, y_test_pred, 
                     y_train_pred_proba, y_test_pred_proba):
        """Create visualization plots"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Train confusion matrix
        cm_train = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['> $10', '<= $10'], 
                   yticklabels=['> $10', '<= $10'])
        axes[0].set_title('Confusion Matrix - Train Set', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Test confusion matrix
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   xticklabels=['> $10', '<= $10'], 
                   yticklabels=['> $10', '<= $10'])
        axes[1].set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        cm_path = self.output_dir / 'confusion_matrices.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrices saved to: {cm_path}")
        plt.close()
        
        # 2. ROC Curves
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Train ROC
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
        roc_auc_train = auc(fpr_train, tpr_train)
        ax.plot(fpr_train, tpr_train, lw=2, label=f'Train (AUC = {roc_auc_train:.4f})')
        
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc_test = auc(fpr_test, tpr_test)
        ax.plot(fpr_test, tpr_test, lw=2, label=f'Test (AUC = {roc_auc_test:.4f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Lasso Logistic Regression', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        roc_path = self.output_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to: {roc_path}")
        plt.close()
        
        # 3. Precision-Recall Curves
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Train PR curve
        precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)
        pr_auc_train = auc(recall_train, precision_train)
        ax.plot(recall_train, precision_train, lw=2, label=f'Train (AUC = {pr_auc_train:.4f})')
        
        # Test PR curve
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)
        pr_auc_test = auc(recall_test, precision_test)
        ax.plot(recall_test, precision_test, lw=2, label=f'Test (AUC = {pr_auc_test:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Lasso Logistic Regression', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        pr_path = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        print(f"Precision-Recall curve saved to: {pr_path}")
        plt.close()
    
    def plot_feature_importance(self, top_n=30):
        """Plot top features by absolute coefficient value"""
        print("\n" + "="*70)
        print(f"PLOTTING TOP {top_n} FEATURES")
        print("="*70)
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
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
        print(f"\nTop 10 positive coefficients (increase probability of <= $10):")
        positive_features = importance_df[importance_df['coefficient'] > 0].head(10)
        for idx, row in positive_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        print(f"\nTop 10 negative coefficients (decrease probability of <= $10):")
        negative_features = importance_df[importance_df['coefficient'] < 0].head(10)
        for idx, row in negative_features.iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
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
        
        model_path = self.output_dir / 'lasso_logistic_regression.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        print(f"\nFull pipeline saved to: {model_path}")
        
        # Save feature names
        features_path = self.output_dir / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"Feature names saved to: {features_path}")
    
    def run_pipeline(self):
        """Execute full classification pipeline"""
        print("\n" + "="*70)
        print("LASSO LOGISTIC REGRESSION CLASSIFICATION PIPELINE")
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
        self.plot_results(y_train, y_train_pred, y_test, y_test_pred, 
                         y_train_pred_proba, y_test_pred_proba)
        
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
    
    pipeline = LassoLogisticRegressionPipeline(
        data_path=data_path,
        output_dir='model_pipeline/outputs_lasso'
    )
    
    pipeline.run_pipeline()
"""
Machine Learning Pipeline for MaxSold Current Bid Prediction - Quick Version
Target: current_bid
Model: XGBoost Regression
Excluded Features: bid_count
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
sns.set_palette("husl")

# Get repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

print("="*80)
print("MAXSOLD ML PIPELINE - CURRENT BID PREDICTION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
DATA_PATH = REPO_ROOT / 'data' / 'final_data' / 'maxsold_final_dataset.parquet'
MODEL_DIR = REPO_ROOT / 'data' / 'models'
OUTPUT_DIR = REPO_ROOT / 'data' / 'models' / 'output'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA
print("="*80)
print("LOADING DATA")
print("="*80)
df = pd.read_parquet(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# 2. PREPROCESS
print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

# Remove bid_count
if 'bid_count' in df.columns:
    print("Removing 'bid_count' column")
    df = df.drop(columns=['bid_count'])

# Remove missing target
df = df.dropna(subset=['current_bid'])
print(f"Samples after removing missing target: {len(df):,}")

# Separate features and target
y = df['current_bid']
X = df.drop(columns=['current_bid'])

# Select numeric columns only for faster processing
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X = X[numeric_cols]

# Fill missing values
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"Final feature shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# 3. TRAIN/TEST SPLIT
print("\n" + "="*80)
print("TRAIN/TEST SPLIT")
print("="*80)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")

# 4. TRAIN MODEL
print("\n" + "="*80)
print("TRAINING XGBOOST MODEL")
print("="*80)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,  # Reduced for speed
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

print("Training model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=25
)
print("✓ Training complete!")

# 5. EVALUATE
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining Set Metrics:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE: {train_mae:.4f}")
print(f"  R²: {train_r2:.4f}")

print("\nTest Set Metrics:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE: {test_mae:.4f}")
print(f"  R²: {test_r2:.4f}")

# 6. FEATURE IMPORTANCE
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

# Save CSV
csv_path = OUTPUT_DIR / 'feature_importance.csv'
feature_importance.to_csv(csv_path, index=False)
print(f"✓ Saved to: {csv_path}")

# Plot top 20
print("Creating feature importance plot...")
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(20)
ax.barh(range(len(top_features)), top_features['importance'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plot_path = OUTPUT_DIR / 'feature_importance.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved to: {plot_path}")

# 7. PREDICTIONS PLOT
print("\nCreating predictions plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=5)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Current Bid', fontsize=12)
axes[0].set_ylabel('Predicted Current Bid', fontsize=12)
axes[0].set_title(f'Training Set (R²={train_r2:.4f})', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Current Bid', fontsize=12)
axes[1].set_ylabel('Predicted Current Bid', fontsize=12)
axes[1].set_title(f'Test Set (R²={test_r2:.4f})', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'predictions_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved to: {plot_path}")

# 8. RESIDUALS PLOT
print("\nCreating residual plots...")
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.3, s=5)
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
axes[0, 0].set_ylabel('Residuals', fontsize=12)
axes[0, 0].set_title('Training Set: Residual Plot', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.3, s=5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
axes[0, 1].set_ylabel('Residuals', fontsize=12)
axes[0, 1].set_title('Test Set: Residual Plot', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(train_residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Training Set: Residual Distribution', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(test_residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Test Set: Residual Distribution', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'residual_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved to: {plot_path}")

# 9. LEARNING CURVE
print("\nCreating learning curve...")
results = model.evals_result()
train_rmse_curve = [np.sqrt(x) for x in results['validation_0']['rmse']]
test_rmse_curve = [np.sqrt(x) for x in results['validation_1']['rmse']]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_rmse_curve, label='Training RMSE', linewidth=2)
ax.plot(test_rmse_curve, label='Test RMSE', linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('XGBoost Learning Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = OUTPUT_DIR / 'learning_curve.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved to: {plot_path}")

# 10. SAVE MODEL
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_path = MODEL_DIR / 'xgboost_model.pkl'
joblib.dump(model, model_path)
print(f"✓ Model saved to: {model_path}")

features_path = MODEL_DIR / 'feature_names.pkl'
joblib.dump(X.columns.tolist(), features_path)
print(f"✓ Feature names saved to: {features_path}")

# Save metrics summary
metrics_path = OUTPUT_DIR / 'metrics_summary.txt'
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
    f.write(f"  RMSE: {train_rmse:.4f}\n")
    f.write(f"  MAE: {train_mae:.4f}\n")
    f.write(f"  R²: {train_r2:.4f}\n")
    f.write("\nTEST SET METRICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"  RMSE: {test_rmse:.4f}\n")
    f.write(f"  MAE: {test_mae:.4f}\n")
    f.write(f"  R²: {test_r2:.4f}\n")
    f.write("\nDATASET INFO:\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Total samples: {len(df):,}\n")
    f.write(f"  Training samples: {len(X_train):,}\n")
    f.write(f"  Test samples: {len(X_test):,}\n")
    f.write(f"  Number of features: {X.shape[1]}\n")

print(f"✓ Metrics summary saved to: {metrics_path}")

print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nResults saved to:")
print(f"  Model: {MODEL_DIR}")
print(f"  Plots & Metrics: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"  - xgboost_model.pkl (trained model)")
print(f"  - feature_names.pkl (feature list)")
print(f"  - feature_importance.csv & .png")
print(f"  - predictions_comparison.png")
print(f"  - residual_analysis.png")
print(f"  - learning_curve.png")
print(f"  - metrics_summary.txt")

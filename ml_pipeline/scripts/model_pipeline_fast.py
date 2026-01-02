"""
Fast ML Pipeline - Samples subset of data for quick execution
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("FAST ML PIPELINE - SAMPLING 50K ROWS")
print("="*80)

# Get repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Setup
MODEL_DIR = REPO_ROOT / 'data' / 'models'
OUTPUT_DIR = REPO_ROOT / 'data' / 'models' / 'output'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load and sample data
print("\n1. Loading data...")
df = pd.read_parquet(REPO_ROOT / 'data' / 'final_data' / 'maxsold_final_dataset.parquet')
print(f"   Full dataset: {df.shape}")

# Sample for speed
df = df.sample(n=50000, random_state=42)
print(f"   Sampled: {df.shape}")

# Remove bid_count and missing targets
if 'bid_count' in df.columns:
    df = df.drop(columns=['bid_count'])
df = df.dropna(subset=['current_bid'])

y = df['current_bid']
X = df.drop(columns=['current_bid'])

# Keep only numeric features
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X = X[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Features: {X.shape[1]}")

# Split
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")

# Train
print("\n3. Training XGBoost (50 trees)...")
model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    verbosity=0
)
model.fit(X_train, y_train)
print("   ✓ Training complete")

# Evaluate
print("\n4. Evaluating...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n   Test RMSE: {test_rmse:.2f}")
print(f"   Test MAE:  {test_mae:.2f}")
print(f"   Test R²:   {test_r2:.4f}")

# Feature importance
print("\n5. Creating visualizations...")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

# Plot 1: Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))
top20 = importance_df.head(20)
ax.barh(range(len(top20)), top20['importance'])
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'])
ax.set_xlabel('Importance')
ax.set_title('Top 20 Features')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=200)
plt.close()
print("   ✓ feature_importance.png")

# Plot 2: Predictions
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_test_pred, alpha=0.3, s=5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title(f'Test Set Predictions (R²={test_r2:.4f})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predictions.png', dpi=200)
plt.close()
print("   ✓ predictions.png")

# Plot 3: Residuals
residuals = y_test - y_test_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test_pred, residuals, alpha=0.3, s=5)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot')
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=40, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'residuals.png', dpi=200)
plt.close()
print("   ✓ residuals.png")

# Save model
print("\n6. Saving model...")
joblib.dump(model, MODEL_DIR / 'xgboost_model.pkl')
joblib.dump(X.columns.tolist(), MODEL_DIR / 'feature_names.pkl')
print(f"   ✓ Saved to {MODEL_DIR}")

# Save summary
with open(OUTPUT_DIR / 'metrics_summary.txt', 'w') as f:
    f.write("MAXSOLD BID PREDICTION MODEL\n")
    f.write("="*60 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: XGBoost (50 estimators)\n")
    f.write(f"Target: current_bid\n")
    f.write(f"Excluded: bid_count\n\n")
    f.write(f"TEST METRICS:\n")
    f.write(f"  RMSE: {test_rmse:.4f}\n")
    f.write(f"  MAE:  {test_mae:.4f}\n")
    f.write(f"  R²:   {test_r2:.4f}\n\n")
    f.write(f"DATASET:\n")
    f.write(f"  Training: {len(X_train):,}\n")
    f.write(f"  Test: {len(X_test):,}\n")
    f.write(f"  Features: {X.shape[1]}\n")

print("\n" + "="*80)
print("✓ PIPELINE COMPLETE!")
print(f"  Model: {MODEL_DIR}/xgboost_model.pkl")
print(f"  Plots: {OUTPUT_DIR}/")
print("="*80)

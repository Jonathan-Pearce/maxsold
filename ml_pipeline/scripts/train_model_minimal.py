#!/usr/bin/env python3
"""
Minimal XGBoost Model - Train and Save
This version is optimized for execution within time constraints
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/final_data/maxsold_final_dataset.parquet'
MODEL_DIR = Path('data/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("XGBOOST MODEL TRAINING - MAXSOLD BID PREDICTION")
print("="*60)

# Step 1: Load data
print("\n[1/6] Loading data...")
df = pd.read_parquet(DATA_PATH)
original_shape = df.shape
df = df.sample(n=min(30000, len(df)), random_state=42)  # Sample for speed
print(f"      Loaded {df.shape[0]:,} rows (sampled from {original_shape[0]:,})")

# Step 2: Prepare features
print("\n[2/6] Preparing features...")
if 'bid_count' in df.columns:
    df = df.drop(columns=['bid_count'])
    print("      Excluded 'bid_count'")

df = df.dropna(subset=['current_bid'])
y = df['current_bid']
X = df.drop(columns=['current_bid'])
X = X.select_dtypes(include=['int64', 'float64']).fillna(0)
print(f"      Features: {X.shape[1]}, Target: current_bid")

# Step 3: Split data
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")

# Step 4: Train model
print("\n[4/6] Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model.fit(X_train, y_train)
print("      Training complete!")

# Step 5: Evaluate
print("\n[5/6] Evaluating model...")
y_test_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"\n      Test Set Performance:")
print(f"      - RMSE: ${rmse:.2f}")
print(f"      - MAE:  ${mae:.2f}")
print(f"      - R²:   {r2:.4f}")

# Step 6: Save model
print("\n[6/6] Saving model...")
joblib.dump(model, MODEL_DIR / 'xgboost_model.pkl')
joblib.dump(X.columns.tolist(), MODEL_DIR / 'feature_names.pkl')

# Save feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
(MODEL_DIR / 'output').mkdir(exist_ok=True)
importance_df.to_csv(MODEL_DIR / 'output' / 'feature_importance.csv', index=False)

print(f"      Saved to: {MODEL_DIR}/")
print(f"      - xgboost_model.pkl")
print(f"      - feature_names.pkl")
print(f"      - output/feature_importance.csv")

# Display top features
print(f"\n      Top 10 Most Important Features:")
for i, row in importance_df.head(10).iterrows():
    print(f"      {row.name+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

print("\n" + "="*60)
print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY")
print("="*60)
print(f"\nModel performance (Test Set):")
print(f"  - RMSE: ${rmse:.2f} (average prediction error)")
print(f"  - R²: {r2:.4f} (variance explained)")
print(f"\nModel saved and ready for use!")

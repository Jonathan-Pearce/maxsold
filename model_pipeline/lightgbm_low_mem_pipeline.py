import pandas as pd
import numpy as np
from pathlib import Path
import json, pickle, os, gc
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Config
DATA_PATH = '/workspaces/maxsold/data/final_data/item_details/items_merged_20251201.parquet'
OUTPUT_DIR = Path('model_pipeline/outputs_lowmem')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'current_bid'
EXCLUDE = {'minimum_bid', 'bid_count'}  # excluded from X (plus target)
DROP_EMBEDDINGS = True
SAMPLE_FRAC = 0.2         # set to 1.0 to use full dataset (might OOM)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# LightGBM low-memory params
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 50,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,            # VERY IMPORTANT: reduce CPU threads to lower memory
    'verbose': -1
}

def small_memory_pipeline():
    print('Loading dataset:', DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    print('Original shape:', df.shape)
    total_mem_mb = df.memory_usage(deep=True).sum()/1024**2
    print(f'Approx memory usage: {total_mem_mb:.1f} MB')

    if TARGET not in df.columns:
        raise RuntimeError(f"Target column '{TARGET}' not in dataset")

    # Drop embeddings to save memory
    if DROP_EMBEDDINGS:
        emb_cols = [c for c in df.columns if 'title_emb_' in c or 'combined_emb_' in c]
        if emb_cols:
            print(f"Dropping {len(emb_cols)} embedding columns to save memory")
            df = df.drop(columns=emb_cols, errors='ignore')

    # Remove explicitly excluded columns from features (keep them in df if needed)
    exclude_and_target = set(EXCLUDE) | {TARGET}
    # If exclude names not present, ignore
    print('Columns to exclude from features:', exclude_and_target & set(df.columns))

    # Optional downsample
    if 0 < SAMPLE_FRAC < 1.0:
        n_before = len(df)
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
        print(f"Downsampled from {n_before:,} -> {len(df):,} rows (frac={SAMPLE_FRAC})")

    # Drop rows with null target
    df = df[df[TARGET].notnull()].copy()

    # Build feature matrix
    X = df[[c for c in df.columns if c not in exclude_and_target]].copy()
    y = df[TARGET].copy()
    print('Feature matrix shape before processing:', X.shape)

    # Datetime handling
    dt_cols = X.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
    for col in dt_cols:
        X[f'{col}_year'] = X[col].dt.year
        X[f'{col}_month'] = X[col].dt.month
        X[f'{col}_day'] = X[col].dt.day
        X[f'{col}_hour'] = X[col].dt.hour
        X[f'{col}_dow'] = X[col].dt.dayofweek
        X.drop(columns=[col], inplace=True)

    # Categorical encoding (label encode small set)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    if cat_cols:
        print(f'Label-encoding {len(cat_cols)} categorical columns (sample: {cat_cols[:5]})')
        for col in cat_cols:
            vc = X[col].value_counts()
            if len(vc) > 1000:
                top = set(vc.head(1000).index)
                X[col] = X[col].where(X[col].isin(top), other='__OTHER__')
            le = LabelEncoder()
            X[col] = X[col].fillna('__MISSING__').astype(str)
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    # Numeric cleanup
    X = X.replace([np.inf, -np.inf], np.nan)
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().any():
            med = X[col].median()
            X[col] = X[col].fillna(med)

    print('Final feature matrix shape:', X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

    # Build LightGBM dataset
    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False)

    # Train
    print('Training LightGBM with params:', {k: LGB_PARAMS[k] for k in ('n_estimators','learning_rate','num_leaves','n_jobs')})
    model = lgb.train(
        LGB_PARAMS,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train','valid']
        #early_stopping_rounds=10,
        #verbose_eval=10
    )

    # Evaluate
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'EVAL RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}')

    # Save model and encoders
    model_path = OUTPUT_DIR / 'lightgbm_lowmem_model.txt'
    model.save_model(str(model_path))
    with open(OUTPUT_DIR / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    meta = {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'trained_at': datetime.utcnow().isoformat(),
        'rows_used': len(X)
    }
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print('Model and metrics saved to', OUTPUT_DIR)
    return model, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Lower OMP/BLAS threads to reduce memory/CPU contention
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    model, X_train, X_test, y_train, y_test = small_memory_pipeline()
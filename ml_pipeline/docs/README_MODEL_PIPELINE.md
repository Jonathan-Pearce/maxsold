# MaxSold Machine Learning Model Pipeline

## Overview
This pipeline trains an XGBoost regression model to predict `current_bid` values using the MaxSold dataset from Kaggle.

## Key Features
- **Target Variable**: `current_bid`
- **Excluded Feature**: `bid_count` (as requested)
- **Model**: XGBoost Regression
- **Train/Test Split**: 80/20
- **Evaluation Metrics**: RMSE, MAE, R², MAPE
- **Visualizations**: Feature importance, predictions, residuals, learning curves

## Files Created

### Main Pipeline Scripts
1. **`model_pipeline.py`** - Full-featured pipeline with all preprocessing
   - Handles categorical, numeric, and datetime features
   - Label encoding for categorical variables
   - Comprehensive visualizations
   - ~200 estimators (slower but more accurate)

2. **`model_pipeline_quick.py`** - Optimized version
   - Uses only numeric features
   - ~100 estimators
   - Faster execution (~2-3 minutes)

3. **`model_pipeline_fast.py`** - Rapid prototyping version
   - Samples 50K rows from dataset
   - 50 estimators
   - Very fast (~30-60 seconds)

### Support Scripts
- **`verify_model_setup.py`** - Verifies all dependencies and data
- **`run_model.sh`** - Bash script to run pipeline
- **`run_model_background.sh`** - Run pipeline in background

## Quick Start

### Option 1: Run the Full Pipeline
```bash
python model_pipeline.py
```

### Option 2: Run Quick Version (Recommended for first run)
```bash
python model_pipeline_quick.py
```

### Option 3: Run Fast Version (Testing/Development)
```bash
python model_pipeline_fast.py
```

### Option 4: Background Execution
```bash
bash run_model_background.sh
# Monitor: tail -f model_pipeline.log
```

## Output Structure

```
data/
├── models/
│   ├── xgboost_model.pkl          # Trained XGBoost model
│   ├── feature_names.pkl           # List of feature names
│   ├── label_encoders.pkl          # Encoders for categorical variables
│   └── output/
│       ├── feature_importance.csv  # Feature importance rankings
│       ├── feature_importance.png  # Top 20 features visualization
│       ├── predictions_comparison.png  # Actual vs Predicted plots
│       ├── residual_analysis.png   # Residual diagnostics
│       ├── learning_curve.png      # Training progress
│       ├── error_distribution.png  # Error analysis
│       └── metrics_summary.txt     # Comprehensive metrics report
```

## Model Architecture

### XGBoost Parameters
```python
{
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,  # (varies by script version)
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}
```

## Evaluation Metrics

The pipeline calculates and displays:

### Regression Metrics
- **RMSE (Root Mean Squared Error)**: Measures average prediction error
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² (R-squared)**: Proportion of variance explained (0-1 scale)
- **MAPE (Mean Absolute Percentage Error)**: Error as percentage

### Visualizations
1. **Feature Importance**: Bar chart of top contributing features
2. **Predictions Plot**: Scatter plot of actual vs predicted values
3. **Residual Plot**: Identifies patterns in prediction errors
4. **Residual Distribution**: Histogram showing error distribution
5. **Learning Curve**: Model convergence during training
6. **Error Distribution**: Percentage error analysis

## Using the Trained Model

### Load and Predict
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('data/models/xgboost_model.pkl')
feature_names = joblib.load('data/models/feature_names.pkl')

# Load encoders (if using full pipeline)
# label_encoders = joblib.load('data/models/label_encoders.pkl')

# Make predictions
new_data = pd.DataFrame(...)  # Your data
predictions = model.predict(new_data[feature_names])
```

## Data Preprocessing

### Full Pipeline (`model_pipeline.py`)
- Removes `bid_count` column
- Handles missing values
- Extracts datetime features (year, month, day, dayofweek, hour)
- Label encodes categorical variables (with cardinality reduction)
- Fills numeric missing values with median
- Handles infinite values

### Quick/Fast Pipelines
- Uses only numeric features
- Simpler preprocessing for speed
- Recommended for initial exploration

## Dataset Information

- **Source**: Kaggle - `pearcej/maxsold-final-dataset`
- **Format**: Parquet
- **Location**: `data/final_data/maxsold_final_dataset.parquet`
- **Size**: ~272K rows, 143 columns
- **Target Range**: $0 - $16,100
- **Target Mean**: $27.28
- **Target Median**: $7.00

## Requirements

All dependencies are in `requirements.txt`:
```
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
pyarrow>=14.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Issue: Script runs too slowly
- **Solution**: Use `model_pipeline_fast.py` which samples 50K rows

### Issue: Memory errors
- **Solution**: Reduce `n_estimators` or sample size further

### Issue: Missing packages
- **Solution**: Run `pip install -r requirements.txt`

### Issue: Data not found
- **Solution**: Run `kaggle datasets download -d pearcej/maxsold-final-dataset -p data/final_data/ --unzip`

## Performance Expectations

### Full Pipeline (model_pipeline.py)
- Runtime: 5-10 minutes
- Memory: ~2-4 GB
- Accuracy: Highest (all features)

### Quick Pipeline (model_pipeline_quick.py)
- Runtime: 2-3 minutes
- Memory: ~1-2 GB
- Accuracy: High (numeric features only)

### Fast Pipeline (model_pipeline_fast.py)
- Runtime: 30-60 seconds
- Memory: < 1 GB
- Accuracy: Good (sampled data)

## Next Steps

1. **Run Initial Test**: `python model_pipeline_fast.py`
2. **Review Outputs**: Check `data/models/output/` directory
3. **Analyze Metrics**: Read `metrics_summary.txt`
4. **Examine Plots**: View PNG files for insights
5. **Full Training**: Run `python model_pipeline_quick.py` for complete results
6. **Tune Model**: Adjust hyperparameters in scripts as needed
7. **Production**: Use `model_pipeline.py` for final model with all features

## Model Interpretation

### Feature Importance
- Check `feature_importance.csv` for complete rankings
- Top features indicate strongest predictors of current_bid
- Use this to understand bidding behavior

### Prediction Quality
- R² close to 1.0 indicates excellent fit
- RMSE shows average dollar error
- Residual plots should show random scatter (no patterns)

### Model Tuning
To improve performance, adjust in script:
- `max_depth`: Control model complexity (3-10)
- `learning_rate`: Learning speed (0.01-0.3)
- `n_estimators`: Number of trees (50-500)
- `subsample`: Row sampling (0.5-1.0)
- `colsample_bytree`: Feature sampling (0.5-1.0)

## Contact & Support

For issues or questions:
1. Check generated `metrics_summary.txt` for diagnostics
2. Review error logs if using background execution
3. Verify data integrity with `verify_model_setup.py`

---

**Last Updated**: December 28, 2025
**Model Type**: XGBoost Regression
**Task**: Current Bid Prediction

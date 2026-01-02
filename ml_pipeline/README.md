# MaxSold Machine Learning Pipeline

Complete XGBoost regression pipeline for predicting current bid values.

## 📁 Directory Structure

```
ml_pipeline/
├── scripts/              # Main pipeline scripts
│   ├── model_pipeline.py           # Full pipeline with all features
│   ├── model_pipeline_quick.py     # Quick pipeline (numeric features)
│   ├── model_pipeline_fast.py      # Fast pipeline (sampled data)
│   └── train_model_minimal.py      # Minimal pipeline (fastest)
│
├── utils/                # Utility scripts
│   ├── verify_model_setup.py       # Setup verification
│   └── launch_pipeline.py          # Pipeline launcher
│
├── bash/                 # Shell scripts
│   ├── run_model.sh                # Run pipeline
│   └── run_model_background.sh     # Run in background
│
└── docs/                 # Documentation
    ├── MODEL_PIPELINE_GUIDE.md     # Complete guide
    ├── README_MODEL_PIPELINE.md    # Detailed reference
    └── ML_PIPELINE_README.py       # Quick reference
```

## 🚀 Quick Start

### From Repository Root

```bash
# Verify setup
python ml_pipeline/utils/verify_model_setup.py

# Train minimal model (fastest - 30-60 seconds)
python ml_pipeline/scripts/train_model_minimal.py

# Train with visualizations (1-2 minutes)
python ml_pipeline/scripts/model_pipeline_fast.py

# Full dataset training (2-3 minutes)
python ml_pipeline/scripts/model_pipeline_quick.py

# Complete pipeline with all features (5-10 minutes)
python ml_pipeline/scripts/model_pipeline.py
```

### Using Bash Scripts

```bash
# Run from repository root
bash ml_pipeline/bash/run_model.sh

# Or run in background
bash ml_pipeline/bash/run_model_background.sh
tail -f model_pipeline.log  # Monitor progress
```

## 📊 Pipeline Scripts

### 1. train_model_minimal.py ⭐ RECOMMENDED
**Fastest execution (30-60 seconds)**
- Samples 30K rows
- 50 estimators
- Saves model + feature importance
- Perfect for quick testing

### 2. model_pipeline_fast.py
**Quick with visualizations (1-2 minutes)**
- Samples 50K rows
- All diagnostic plots
- Metrics summary

### 3. model_pipeline_quick.py
**Full dataset, numeric features (2-3 minutes)**
- Complete 272K dataset
- Production-quality model
- Comprehensive diagnostics

### 4. model_pipeline.py
**Complete pipeline (5-10 minutes)**
- All feature types (numeric, categorical, datetime)
- Best accuracy
- Full preprocessing pipeline

## 🎯 Model Details

- **Model Type**: XGBoost Regression
- **Target Variable**: `current_bid`
- **Excluded Feature**: `bid_count` (as requested)
- **Train/Test Split**: 80/20
- **Metrics**: RMSE, MAE, R², MAPE

## 📈 Outputs

All outputs are saved to `../../data/models/` (relative to scripts):

```
data/models/
├── xgboost_model.pkl              # Trained model
├── feature_names.pkl               # Feature list
├── label_encoders.pkl              # Encoders (full pipeline only)
└── output/
    ├── feature_importance.csv      # Feature rankings
    ├── feature_importance.png      # Top 20 features chart
    ├── predictions_comparison.png  # Actual vs predicted
    ├── residual_analysis.png       # Residual diagnostics
    ├── learning_curve.png          # Training progress
    ├── error_distribution.png      # Error analysis
    └── metrics_summary.txt         # Complete metrics
```

## 📚 Documentation

- **[MODEL_PIPELINE_GUIDE.md](docs/MODEL_PIPELINE_GUIDE.md)** - Complete implementation guide
- **[README_MODEL_PIPELINE.md](docs/README_MODEL_PIPELINE.md)** - Detailed reference
- **[ML_PIPELINE_README.py](docs/ML_PIPELINE_README.py)** - Quick reference display

## 🔧 Usage Examples

### Train and Save Model

```bash
cd /workspaces/maxsold
python ml_pipeline/scripts/train_model_minimal.py
```

### Load and Use Model

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('data/models/xgboost_model.pkl')
features = joblib.load('data/models/feature_names.pkl')

# Make predictions
new_data = pd.DataFrame(...)  # Your data
predictions = model.predict(new_data[features])
```

### Verify Setup

```bash
python ml_pipeline/utils/verify_model_setup.py
```

## 🛠 Requirements

All dependencies in main `requirements.txt`:
- pandas >= 2.0.0
- numpy >= 1.24.0
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0
- pyarrow >= 14.0.0

## 📝 Notes

- All scripts use relative paths from their location
- Scripts expect to be run from repository root or their own directory
- Data files remain in `data/` directory at repository root
- Model outputs saved to `data/models/` and `data/models/output/`

## 🐛 Troubleshooting

**Issue**: Module not found
```bash
pip install -r requirements.txt
```

**Issue**: Data not found
```bash
kaggle datasets download -d pearcej/maxsold-final-dataset -p data/final_data/ --unzip
```

**Issue**: Script too slow
```bash
python ml_pipeline/scripts/train_model_minimal.py  # Use fastest version
```

---

**Created**: December 28, 2025  
**Last Updated**: January 2, 2026  
**Version**: 2.0 (Reorganized Structure)

# MaxSold Machine Learning Pipeline

Complete machine learning pipelines for MaxSold auction prediction.

## 📦 Available Models

### 1. XGBoost Current Bid Predictor
Traditional ML model for predicting current bid values from item features.

**Use Case**: Predict item prices based on features like title, category, auction timing, etc.

### 2. Bid Sequence Predictor (Deep Learning) ⭐ NEW
LSTM/GRU-based model for predicting final auction prices from partial bid sequences.

**Use Case**: Predict final winning bid amount given only the first X bids (production scenario).

## 📁 Directory Structure

```
ml_pipeline/
├── scripts/                          # Main pipeline scripts
│   ├── model_pipeline.py             # Full XGBoost pipeline
│   ├── model_pipeline_quick.py       # Quick XGBoost pipeline
│   ├── model_pipeline_fast.py        # Fast XGBoost pipeline (sampled)
│   ├── train_model_minimal.py        # Minimal XGBoost pipeline
│   ├── train_bid_sequence_model.py   # Bid sequence model trainer
│   ├── demo_bid_sequence_model.py    # Bid sequence demo
│   └── example_bid_prediction.py     # Production usage examples
│
├── bid_sequence_model/               # Bid sequence model package
│   ├── data_loader.py                # Data loading & preprocessing
│   ├── model.py                      # LSTM/GRU architecture
│   ├── trainer.py                    # Training pipeline
│   ├── __init__.py                   # Package initialization
│   └── README.md                     # Detailed documentation
│
├── utils/                            # Utility scripts
│   ├── verify_model_setup.py         # Setup verification
│   └── launch_pipeline.py            # Pipeline launcher
│
├── bash/                             # Shell scripts
│   ├── run_model.sh                  # Run pipeline
│   └── run_model_background.sh       # Run in background
│
└── docs/                             # Documentation
    ├── MODEL_PIPELINE_GUIDE.md       # Complete guide
    ├── README_MODEL_PIPELINE.md      # Detailed reference
    └── ML_PIPELINE_README.py         # Quick reference
```

## 🚀 Quick Start

### XGBoost Current Bid Predictor

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

### Bid Sequence Predictor (Deep Learning)

```bash
# Run demo with synthetic data (no Kaggle data needed)
python ml_pipeline/scripts/demo_bid_sequence_model.py

# Train on real data (requires bid history from Kaggle)
python ml_pipeline/scripts/train_bid_sequence_model.py \
    --bid_history data/bid_history.parquet \
    --item_metadata data/item_metadata.parquet \
    --epochs 50

# See production usage examples
python ml_pipeline/scripts/example_bid_prediction.py
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
- torch >= 2.0.0 (for bid sequence model - PyTorch)

## 📖 Detailed Documentation

### Bid Sequence Model
See **[bid_sequence_model/README.md](bid_sequence_model/README.md)** for complete documentation including:
- Model architecture details
- Data requirements and format
- Training configuration options
- Production deployment examples
- Handling partial bid sequences
- Performance metrics and tuning

### XGBoost Model
- **[MODEL_PIPELINE_GUIDE.md](docs/MODEL_PIPELINE_GUIDE.md)** - Complete implementation guide
- **[README_MODEL_PIPELINE.md](docs/README_MODEL_PIPELINE.md)** - Detailed reference
- **[ML_PIPELINE_README.py](docs/ML_PIPELINE_README.py)** - Quick reference display

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

**Issue**: Bid sequence model - no Kaggle data
```bash
# Use the demo with synthetic data to test the model
python ml_pipeline/scripts/demo_bid_sequence_model.py
```

---

**Created**: December 28, 2025  
**Last Updated**: January 2, 2026  
**Version**: 2.0 (Reorganized Structure)

# Cookiecutter Data Science Structure - Quick Visual Reference

## Before and After Comparison

### BEFORE (Old Structure)
```
maxsold/
├── scrapers/              # Data collection scripts
│   ├── 01_extract_auction_search.py
│   ├── 02_extract_auction_details.py
│   ├── 03_extract_items_details.py
│   ├── 04_extract_bid_history.py
│   ├── 05_extract_item_enriched_details.py
│   ├── 06_save_auction_images.py
│   └── monthly_scraping_pipeline.py
├── backfilling/           # Historical data scripts
│   └── extract_auction_location_data.py
├── feature_engineering/   # Feature creation
│   ├── auction_features.py
│   ├── item_details_text_features_small.py
│   ├── image_features.py
│   └── final_dataset_builder.py
├── ml_pipeline/          # Machine learning
│   ├── bid_sequence_model/
│   ├── scripts/
│   └── utils/
├── data/                 # Mixed data storage
│   ├── engineered/
│   ├── final/
│   ├── images/
│   └── models/
├── docs/                 # Documentation
└── utils/               # Utilities
```

### AFTER (Cookiecutter Data Science Structure)
```
maxsold/
├── src/                      # ⭐ All source code organized here
│   ├── data/                # 📥 Data collection (formerly scrapers/ + backfilling/)
│   │   ├── 01_extract_auction_search.py
│   │   ├── 02_extract_auction_details.py
│   │   ├── 03_extract_items_details.py
│   │   ├── 04_extract_bid_history.py
│   │   ├── 05_extract_item_enriched_details.py
│   │   ├── 06_save_auction_images.py
│   │   ├── monthly_scraping_pipeline.py
│   │   └── extract_auction_location_data.py
│   ├── features/            # 🔧 Feature engineering (formerly feature_engineering/)
│   │   ├── auction_features.py
│   │   ├── auction_details_features.py
│   │   ├── item_details_text_features_small.py
│   │   ├── image_features.py
│   │   ├── dataset_merger.py
│   │   └── final_dataset_builder.py
│   ├── models/              # 🤖 ML pipeline (formerly ml_pipeline/)
│   │   ├── bid_sequence_model/
│   │   ├── scripts/
│   │   └── utils/
│   └── visualization/       # 📊 Visualization scripts
│
├── data/                     # 💾 Data storage (organized by stage)
│   ├── raw/                 # Original, immutable data (formerly images/)
│   ├── interim/             # Intermediate data (formerly engineered/)
│   ├── processed/           # Final datasets (formerly final/)
│   └── external/            # Third-party data
│
├── models/                   # 🎯 Trained models (formerly data/models/)
│   ├── item_features/
│   └── output/
│
├── notebooks/                # 📓 Jupyter notebooks (NEW)
├── references/               # 📚 Reference materials (NEW)
├── reports/                  # 📄 Generated reports (NEW)
│   └── figures/             # Charts and figures
├── docs/                     # 📖 Documentation
└── utils/                    # 🛠️ Utilities
```

## Migration Map

| Old Location | → | New Location | Purpose |
|-------------|---|--------------|---------|
| `scrapers/` | → | `src/data/` | Data collection scripts |
| `backfilling/` | → | `src/data/` | Historical data extraction |
| `feature_engineering/` | → | `src/features/` | Feature engineering |
| `ml_pipeline/` | → | `src/models/` | Model training & prediction |
| `data/images/` | → | `data/raw/` | Raw immutable data |
| `data/engineered/` | → | `data/interim/` | Intermediate data |
| `data/final/` | → | `data/processed/` | Final datasets |
| `data/models/` | → | `models/` | Trained model files |
| - | → | `notebooks/` | Jupyter notebooks (NEW) |
| - | → | `references/` | Reference materials (NEW) |
| - | → | `reports/` | Generated reports (NEW) |

## Key Benefits

### 1. **Standardization** 🎯
- Industry-standard structure recognized by data scientists worldwide
- Follows best practices from hundreds of data science projects

### 2. **Clear Separation of Concerns** 🏗️
- **Source code** (`src/`) separated from **data** (`data/`) and **models** (`models/`)
- Each directory has a single, clear purpose

### 3. **Data Pipeline Clarity** 📊
```
raw → interim → processed
(immutable) → (transformed) → (model-ready)
```

### 4. **Reproducibility** 🔄
- Clear data lineage from raw to processed
- All transformations in code, not manual edits

### 5. **Collaboration** 🤝
- Team members can quickly understand project structure
- Reduces onboarding time for new developers

## Usage Examples

### Old Way (Before)
```bash
python scrapers/01_extract_auction_search.py
python feature_engineering/auction_features.py
python ml_pipeline/scripts/train_model_minimal.py
```

### New Way (After)
```bash
python src/data/01_extract_auction_search.py
python src/features/auction_features.py
python src/models/scripts/train_model_minimal.py
```

## Data Flow

```
┌─────────────┐
│   Raw Data  │  data/raw/
│ (scrapers)  │  - Original auction data
└──────┬──────┘  - Images
       │         - API responses
       ↓
┌─────────────┐
│  Interim    │  data/interim/
│  (features) │  - Extracted features
└──────┬──────┘  - Cleaned data
       │         - Feature matrices
       ↓
┌─────────────┐
│ Processed   │  data/processed/
│  (models)   │  - Final training data
└──────┬──────┘  - Test datasets
       │         - Model-ready CSVs
       ↓
┌─────────────┐
│   Models    │  models/
│  (output)   │  - Trained models (.pkl)
└─────────────┘  - Predictions
                 - Evaluations
```

## Next Steps

1. ✅ **Structure Created** - All directories and README files in place
2. ✅ **Files Copied** - Scripts copied to new locations
3. ✅ **Documentation** - Comprehensive docs created
4. ⏳ **Import Updates** - Update Python imports (future work)
5. ⏳ **Testing** - Verify scripts work with new structure
6. ⏳ **Cleanup** - Remove old directories after verification

## Learn More

- 📖 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed reorganization guide
- 📖 [src/README.md](src/README.md) - Source code overview
- 📖 [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - Official documentation

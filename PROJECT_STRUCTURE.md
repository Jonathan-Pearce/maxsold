# Project Organization

This project follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) structure, which provides a standardized and logical organization for data science projects.

## Directory Structure

```
├── README.md          <- The top-level README for developers using this project
├── requirements.txt   <- The requirements file for reproducing the environment
│
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed (previously engineered/)
│   ├── processed      <- The final, canonical data sets for modeling (previously final/)
│   └── raw            <- The original, immutable data dump (includes images/)
│
├── docs               <- Documentation files
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   ├── item_features  <- Item feature models
│   └── output         <- Model output files
│
├── notebooks          <- Jupyter notebooks for exploration and analysis
│
├── references         <- Data dictionaries, manuals, and other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data (previously scrapers/ and backfilling/)
│   │   ├── 01_extract_auction_search.py
│   │   ├── 02_extract_auction_details.py
│   │   ├── 03_extract_items_details.py
│   │   ├── 04_extract_bid_history.py
│   │   ├── 05_extract_item_enriched_details.py
│   │   ├── 06_save_auction_images.py
│   │   ├── monthly_scraping_pipeline.py
│   │   └── extract_auction_location_data.py
│   │
│   ├── features       <- Scripts to turn raw data into features (previously feature_engineering/)
│   │   ├── auction_features.py
│   │   ├── auction_details_features.py
│   │   ├── item_details_text_features_small.py
│   │   ├── image_features.py
│   │   ├── dataset_merger.py
│   │   └── final_dataset_builder.py
│   │
│   ├── models         <- Scripts to train models and make predictions (previously ml_pipeline/)
│   │   ├── bid_sequence_model/
│   │   ├── scripts/
│   │   └── utils/
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│
└── Docker files       <- Docker configuration files
    ├── Dockerfile
    ├── docker-compose.yml
    ├── docker-run.sh
    └── docker-run.bat
```

## Migration Map

This section shows how the old structure maps to the new structure:

### Old → New Structure Mapping

| Old Location | New Location | Purpose |
|-------------|--------------|---------|
| `scrapers/` | `src/data/` | Data collection scripts |
| `backfilling/` | `src/data/` | Historical data extraction |
| `feature_engineering/` | `src/features/` | Feature engineering scripts |
| `ml_pipeline/` | `src/models/` | Model training and prediction |
| `data/engineered/` | `data/interim/` | Intermediate processed data |
| `data/final/` | `data/processed/` | Final modeling datasets |
| `data/images/` | `data/raw/` | Raw image data |
| `data/models/` | `models/` | Trained model files |
| `utils/` | Keep as is | Utility functions |
| `docs/` | Keep as is | Documentation |

### Why Cookiecutter Data Science?

The Cookiecutter Data Science structure provides several benefits:

1. **Standardization**: A well-known structure that data scientists can quickly understand
2. **Separation of Concerns**: Clear distinction between raw data, processed data, source code, and outputs
3. **Reproducibility**: Organized structure makes it easier to reproduce results
4. **Collaboration**: Standard layout helps team members navigate the project
5. **Best Practices**: Built on lessons learned from hundreds of data science projects

### Key Principles

1. **Data is immutable**: Raw data should never be modified. All transformations should be done in code.
2. **Notebooks are for exploration**: Analysis and exploration in notebooks; production code goes in `src/`
3. **Build from the environment up**: Use virtual environments and requirements files
4. **Keep secrets and configuration out of version control**: Use .env files for sensitive information
5. **Be conservative in changing the default folder structure**: The structure is designed to work for most projects

## Using the New Structure

### Running Scripts

Scripts now use the `src/` package structure. You may need to update Python paths:

```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:${PWD}"
python src/data/01_extract_auction_search.py
```

Or with Docker:
```bash
./docker-run.sh shell
python src/data/01_extract_auction_search.py
```

### Importing Modules

Use absolute imports from the `src` package:

```python
# Instead of relative imports
from src.data import monthly_scraping_pipeline
from src.features import auction_features
from src.models.scripts import train_model_minimal
```

## Backward Compatibility

The old directories (`scrapers/`, `feature_engineering/`, `ml_pipeline/`) are maintained during the transition period for backward compatibility. Once all scripts and documentation are updated to use the new structure, these directories can be removed.

**Important Note for GitHub Actions:**
The repository has GitHub Actions workflows that reference old paths. See [docs/GITHUB_ACTIONS_MIGRATION.md](docs/GITHUB_ACTIONS_MIGRATION.md) for guidance on updating these workflows.

## Next Steps

1. ✅ Create new directory structure
2. ✅ Copy files to new locations
3. ✅ Update documentation
4. ⏳ Update import paths in Python files
5. ⏳ Update GitHub Actions workflows (see [docs/GITHUB_ACTIONS_MIGRATION.md](docs/GITHUB_ACTIONS_MIGRATION.md))
6. ⏳ Update Docker configuration if needed
7. ⏳ Test all scripts with new structure
8. ⏳ Remove old directories after verification

## References

- [Cookiecutter Data Science Project Template](https://drivendata.github.io/cookiecutter-data-science/)
- [Good Enough Practices in Scientific Computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)

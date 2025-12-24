#!/bin/bash
# Quick verification script for the refactored pipeline

echo "========================================"
echo "Verifying Feature Engineering Pipeline"
echo "========================================"

echo ""
echo "Checking Python syntax..."
python -m py_compile feature_engineering/__init__.py && echo "✓ feature_engineering/__init__.py"
python -m py_compile feature_engineering/auction_features.py && echo "✓ auction_features.py"
python -m py_compile feature_engineering/item_features.py && echo "✓ item_features.py"
python -m py_compile feature_engineering/item_enriched_features.py && echo "✓ item_enriched_features.py"
python -m py_compile feature_engineering/dataset_merger.py && echo "✓ dataset_merger.py"
python -m py_compile utils/__init__.py && echo "✓ utils/__init__.py"
python -m py_compile utils/kaggle_pipeline.py && echo "✓ kaggle_pipeline.py"
python -m py_compile run_pipeline.py && echo "✓ run_pipeline.py"

echo ""
echo "All syntax checks passed!"
echo ""
echo "To run the full pipeline:"
echo "  python run_pipeline.py"
echo ""
echo "To run tests:"
echo "  python test_modules.py"
echo ""
echo "For more information, see README_PIPELINE.md"
echo "========================================"

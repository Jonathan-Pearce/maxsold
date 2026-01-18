"""
MaxSold Feature Engineering Package

This package provides modular, reusable feature engineering transformations
for auction, item, and item-enriched datasets.

Modules:
- auction_features: Transform auction details data
- item_features: Transform item details data with text embeddings
- item_enriched_features: Transform enriched item data
- image_features: Extract deep learning features from item images
- dataset_merger: Merge all feature-engineered datasets
"""

from .auction_features import AuctionFeatureEngineer
from .item_features import ItemFeatureEngineer  
from .item_enriched_features import ItemEnrichedFeatureEngineer
from .image_features import ImageFeatureExtractor, aggregate_features_by_item
from .dataset_merger import DatasetMerger

__all__ = [
    'AuctionFeatureEngineer',
    'ItemFeatureEngineer',
    'ItemEnrichedFeatureEngineer',
    'ImageFeatureExtractor',
    'aggregate_features_by_item',
    'DatasetMerger'
]

__version__ = '1.0.0'

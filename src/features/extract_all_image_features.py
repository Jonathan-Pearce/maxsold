"""
Example: Extract Image Features from All Auction Images

This script demonstrates how to use the ImageFeatureExtractor to process
all images in the data/images directory and save the results.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_engineering import ImageFeatureExtractor, aggregate_features_by_item


def main():
    """
    Extract features from all auction images and save results.
    """
    print("="*80)
    print("EXTRACTING IMAGE FEATURES FROM ALL AUCTION IMAGES")
    print("="*80)
    
    # Setup paths
    image_dir = project_root / 'data' / 'images'
    output_dir = project_root / 'data' / 'image_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor with MobileNetV2
    print("\nInitializing MobileNetV2 feature extractor...")
    extractor = ImageFeatureExtractor(model_name='mobilenet_v2')
    
    # Extract features from all images
    print("\nProcessing all images (this may take a while)...")
    df_features = extractor.extract_features_from_directory(
        image_dir=image_dir,
        pattern='*.webp'  # Process all .webp images
    )
    
    print(f"\n✓ Successfully extracted features from {len(df_features)} images")
    print(f"✓ Feature vector dimension: {extractor.feature_dim}")
    
    # Save raw image-level features
    raw_output_path = output_dir / 'image_features_raw.parquet'
    extractor.save_features(df_features, raw_output_path)
    
    # Aggregate features by item (multiple images per item)
    print("\nAggregating features by item...")
    df_aggregated = aggregate_features_by_item(df_features, method='mean')
    
    print(f"✓ Aggregated {len(df_features)} images into {len(df_aggregated)} items")
    
    # Save item-level aggregated features
    agg_output_path = output_dir / 'image_features_by_item.parquet'
    extractor.save_features(df_aggregated, agg_output_path)
    
    # Save model metadata for reproducibility
    extractor.save_model_info(output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nOutput files created:")
    print(f"  1. Raw features (per image):     {raw_output_path}")
    print(f"     - Shape: {df_features.shape}")
    print(f"     - Columns: image_path, image_name, img_feature_0...img_feature_{extractor.feature_dim-1}")
    print(f"\n  2. Aggregated features (per item): {agg_output_path}")
    print(f"     - Shape: {df_aggregated.shape}")
    print(f"     - Columns: item_id, img_feature_0...img_feature_{extractor.feature_dim-1}")
    print(f"\n  3. Model metadata:                {output_dir / 'image_features_metadata.pkl'}")
    print(f"\nThese features can now be used for training regression models!")
    print("="*80)


if __name__ == '__main__':
    main()

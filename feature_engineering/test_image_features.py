"""
Test script for Image Feature Extraction

This script tests the ImageFeatureExtractor on actual auction images.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_engineering import ImageFeatureExtractor, aggregate_features_by_item


def main():
    print("="*80)
    print("IMAGE FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Setup paths
    image_dir = project_root / 'data' / 'images'
    output_dir = project_root / 'data' / 'image_features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if images exist
    if not image_dir.exists():
        print(f"✗ Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Test with MobileNetV2 (default)
    print("\n--- Test 1: MobileNetV2 Feature Extraction ---")
    try:
        extractor = ImageFeatureExtractor(model_name='mobilenet_v2')
        
        # Process a subset of images for testing (first 50)
        df_features = extractor.extract_features_from_directory(
            image_dir=image_dir,
            pattern='*.webp',
            max_images=50
        )
        
        print(f"\n✓ Extracted features from {len(df_features)} images")
        print(f"✓ Feature dimension: {extractor.feature_dim}")
        print(f"✓ DataFrame shape: {df_features.shape}")
        print(f"\nSample features:")
        print(df_features[['image_name', 'img_feature_0', 'img_feature_1']].head())
        
        # Save raw features
        raw_features_path = output_dir / 'image_features_raw_sample.parquet'
        extractor.save_features(df_features, raw_features_path)
        
        # Test aggregation by item
        print("\n--- Test 2: Aggregate Features by Item ---")
        df_aggregated = aggregate_features_by_item(df_features, method='mean')
        
        print(f"✓ Aggregated shape: {df_aggregated.shape}")
        print(f"\nSample aggregated features:")
        print(df_aggregated[['item_id', 'img_feature_0', 'img_feature_1']].head())
        
        # Save aggregated features
        agg_features_path = output_dir / 'image_features_by_item_sample.parquet'
        extractor.save_features(df_aggregated, agg_features_path)
        
        # Save model info
        extractor.save_model_info(output_dir)
        
        print("\n" + "="*80)
        print("TEST PASSED")
        print("="*80)
        print(f"\nOutput files:")
        print(f"  - {raw_features_path}")
        print(f"  - {agg_features_path}")
        print(f"  - {output_dir / 'image_features_metadata.pkl'}")
        
        print("\n✓ Image feature extraction pipeline is working correctly!")
        print("✓ Ready to process all images in the dataset")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

# Image Feature Extraction Pipeline

This module provides deep learning-based feature extraction from auction item images using pre-trained CNN models.

## Overview

The image feature extraction pipeline uses a pre-trained **MobileNetV2** model (trained on ImageNet) to extract rich visual features from auction item images. The final classification layer is removed, and the output of the last convolutional layer is used as a feature vector for downstream machine learning tasks.

## Features

- **Pre-trained CNN Models**: MobileNetV2 (1280-dim features) or ShuffleNetV2 (1024-dim features)
- **Efficient Processing**: Batch processing with progress tracking
- **Image Format Support**: WebP, JPEG, PNG
- **Aggregation**: Multiple images per item can be aggregated (mean, max, or first)
- **Output Formats**: Parquet or CSV
- **Reproducibility**: Model metadata saved for deployment

## Installation

Install the required dependencies:

```bash
pip install torch>=2.6.0 torchvision>=0.20.0
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Extract Features from All Images

```python
from feature_engineering import ImageFeatureExtractor, aggregate_features_by_item

# Initialize extractor
extractor = ImageFeatureExtractor(model_name='mobilenet_v2')

# Extract features from a directory
df_features = extractor.extract_features_from_directory(
    image_dir='data/images',
    pattern='*.webp'
)

# Save raw features
extractor.save_features(df_features, 'output/image_features.parquet')

# Aggregate by item (if multiple images per item)
df_aggregated = aggregate_features_by_item(df_features, method='mean')
extractor.save_features(df_aggregated, 'output/features_by_item.parquet')
```

### Extract Features from a Single Image

```python
from feature_engineering import ImageFeatureExtractor

extractor = ImageFeatureExtractor(model_name='mobilenet_v2')
features = extractor.extract_features_from_image('path/to/image.webp')

# features is a numpy array of shape (1280,)
print(f"Feature vector shape: {features.shape}")
```

## Command-Line Usage

### Test on Sample Images

```bash
cd feature_engineering
python test_image_features.py
```

This will:
- Process 50 sample images
- Extract features using MobileNetV2
- Save raw and aggregated features
- Display statistics and sample outputs

### Process All Images

```bash
cd feature_engineering
python extract_all_image_features.py
```

This will:
- Process all .webp images in `data/images/`
- Generate two output files:
  - `data/image_features/image_features_raw.parquet` - Features per image
  - `data/image_features/image_features_by_item.parquet` - Features per item (aggregated)

## Output Format

### Raw Features (per image)

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Relative path to image |
| `image_name` | string | Image filename |
| `img_feature_0` to `img_feature_1279` | float | Feature values |

### Aggregated Features (per item)

| Column | Type | Description |
|--------|------|-------------|
| `item_id` | string | Item ID extracted from filename |
| `img_feature_0` to `img_feature_1279` | float | Aggregated feature values |

## Model Details

### MobileNetV2 (Default)

- **Architecture**: MobileNetV2 (ImageNet pre-trained)
- **Feature Dimension**: 1280
- **Input Size**: 224x224
- **Advantages**: Lightweight, fast, good for production
- **Use Case**: General-purpose feature extraction

### ShuffleNetV2

- **Architecture**: ShuffleNetV2 1.0x (ImageNet pre-trained)
- **Feature Dimension**: 1024
- **Input Size**: 224x224
- **Advantages**: Even more efficient than MobileNetV2
- **Use Case**: Resource-constrained environments

## Image Preprocessing

All images undergo the following preprocessing steps:

1. **Resize**: 256x256 pixels
2. **Center Crop**: 224x224 pixels
3. **Normalization**: ImageNet mean and std
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## API Reference

### `ImageFeatureExtractor`

Main class for extracting image features.

#### Constructor

```python
ImageFeatureExtractor(model_name='mobilenet_v2', device=None)
```

**Parameters:**
- `model_name` (str): CNN model to use ('mobilenet_v2' or 'shufflenet_v2_x1_0')
- `device` (str): Device to run model on ('cuda', 'cpu', or None for auto-detect)

#### Methods

##### `extract_features_from_image(image_path)`

Extract features from a single image.

**Parameters:**
- `image_path` (str or Path): Path to image file

**Returns:**
- `numpy.ndarray`: Feature vector (flattened)

##### `extract_features_from_directory(image_dir, pattern='*.webp', max_images=None)`

Extract features from all images in a directory.

**Parameters:**
- `image_dir` (str or Path): Directory containing images
- `pattern` (str): File pattern to match (default: '*.webp')
- `max_images` (int): Maximum number of images to process (None = all)

**Returns:**
- `pd.DataFrame`: DataFrame with image paths and features

##### `save_features(df, output_path)`

Save extracted features to disk.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with features
- `output_path` (str or Path): Path to save file (.parquet or .csv)

##### `save_model_info(output_dir)`

Save model configuration for reproducibility.

**Parameters:**
- `output_dir` (str or Path): Directory to save metadata

### `aggregate_features_by_item(df_features, method='mean')`

Aggregate image features when multiple images exist per item.

**Parameters:**
- `df_features` (pd.DataFrame): DataFrame with image features
- `method` (str): Aggregation method ('mean', 'max', or 'first')

**Returns:**
- `pd.DataFrame`: Aggregated features by item ID

## Usage in ML Pipeline

The extracted features can be used for various downstream tasks:

1. **Price Prediction**: Train regression models to predict item prices
2. **Category Classification**: Classify items into categories
3. **Similarity Search**: Find similar items based on visual features
4. **Quality Assessment**: Assess item quality from images

### Example: Using Features for Regression

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load extracted features
df_features = pd.read_parquet('data/image_features/image_features_by_item.parquet')

# Prepare feature matrix
feature_cols = [col for col in df_features.columns if col.startswith('img_feature_')]
X = df_features[feature_cols].values

# Load target variable (e.g., price)
# y = ...

# Train model
model = RandomForestRegressor()
model.fit(X, y)
```

## Performance

- **Speed**: ~50 images/second on CPU (varies by hardware)
- **Memory**: ~500MB for model weights
- **GPU Acceleration**: Automatically used if CUDA is available

## Troubleshooting

### Out of Memory

If you encounter memory issues:

1. Process images in batches using `max_images` parameter
2. Use a smaller model (ShuffleNetV2)
3. Ensure sufficient RAM (recommended: 8GB+)

### CUDA Errors

If GPU is not working:

```python
# Force CPU mode
extractor = ImageFeatureExtractor(model_name='mobilenet_v2', device='cpu')
```

## Integration with Existing Pipeline

The image features can be merged with other features in the pipeline:

```python
from feature_engineering import DatasetMerger

# Load different feature sets
df_item_text = pd.read_parquet('data/engineered/item/item_engineered.parquet')
df_image = pd.read_parquet('data/image_features/image_features_by_item.parquet')

# Merge on item ID
df_combined = df_item_text.merge(df_image, left_on='id', right_on='item_id', how='left')
```

## Future Enhancements

Potential improvements to consider:

- Support for additional models (ResNet, EfficientNet)
- Fine-tuning on auction-specific images
- Multi-scale feature extraction
- Attention-based feature selection
- Image quality assessment

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [ShuffleNetV2 Paper](https://arxiv.org/abs/1807.11164)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)

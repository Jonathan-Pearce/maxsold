"""
Image Feature Engineering

This module provides a class for extracting deep learning features from auction item images
using pre-trained CNN models (MobileNetV2) on ImageNet.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle


class ImageFeatureExtractor:
    """
    Feature extraction for auction item images using pre-trained CNN.
    
    Uses MobileNetV2 pre-trained on ImageNet, removing the final classification
    layer to extract feature vectors for downstream regression tasks.
    """
    
    def __init__(self, model_name='mobilenet_v2', device=None):
        """
        Initialize the image feature extractor.
        
        Parameters:
        model_name (str): CNN model to use ('mobilenet_v2' or 'shufflenet_v2_x1_0')
        device (str): Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model and preprocessing
        self.model = None
        self.feature_dim = None
        self.transform = None
        self._setup_model()
        
    def _setup_model(self):
        """
        Load pre-trained model and remove final classification layer.
        """
        print(f"\nInitializing {self.model_name} on {self.device}...")
        
        if self.model_name == 'mobilenet_v2':
            # Load pre-trained MobileNetV2
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            base_model = models.mobilenet_v2(weights=weights)
            
            # Remove classifier, keep only features
            self.model = base_model.features
            self.feature_dim = 1280  # MobileNetV2 output channels
            
        elif self.model_name == 'shufflenet_v2_x1_0':
            # Load pre-trained ShuffleNetV2
            weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
            base_model = models.shufflenet_v2_x1_0(weights=weights)
            
            # Remove classifier, keep conv layers
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 1024  # ShuffleNetV2 output channels
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded: {self.model_name}")
        print(f"✓ Feature dimension: {self.feature_dim}")
        print(f"✓ Device: {self.device}")
    
    def extract_features_from_image(self, image_path):
        """
        Extract features from a single image.
        
        Parameters:
        image_path (str or Path): Path to image file
        
        Returns:
        numpy.ndarray: Feature vector (flattened)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                
                # Global average pooling if needed
                if len(features.shape) == 4:  # [batch, channels, height, width]
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                
                # Flatten
                features = features.view(features.size(0), -1)
                
            # Convert to numpy
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_features_from_directory(self, image_dir, pattern='*.webp', max_images=None):
        """
        Extract features from all images in a directory.
        
        Parameters:
        image_dir (str or Path): Directory containing images
        pattern (str): File pattern to match (default: '*.webp')
        max_images (int): Maximum number of images to process (None = all)
        
        Returns:
        pd.DataFrame: DataFrame with image paths and features
        """
        image_dir = Path(image_dir)
        
        # Find all image files recursively
        if '**' not in pattern:
            pattern = f'**/{pattern}'
        image_paths = sorted(image_dir.glob(pattern))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"\nProcessing {len(image_paths)} images from {image_dir}...")
        
        results = []
        for img_path in tqdm(image_paths, desc="Extracting features"):
            features = self.extract_features_from_image(img_path)
            
            if features is not None:
                # Extract metadata from path
                relative_path = img_path.relative_to(image_dir)
                
                result = {
                    'image_path': str(relative_path),
                    'image_name': img_path.name,
                    'features': features
                }
                results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Expand features into separate columns efficiently
        if len(df) > 0:
            feature_matrix = np.vstack(df['features'].values)
            
            # Create feature columns as a separate DataFrame and concatenate
            feature_cols = {f'img_feature_{i}': feature_matrix[:, i] 
                          for i in range(feature_matrix.shape[1])}
            df_features = pd.DataFrame(feature_cols)
            
            # Drop the packed features column and concatenate with feature columns
            df = pd.concat([df.drop('features', axis=1), df_features], axis=1)
        
        print(f"✓ Extracted features from {len(df)} images")
        print(f"✓ Feature dimensions: {self.feature_dim}")
        
        return df
    
    def save_features(self, df, output_path):
        """
        Save extracted features to disk.
        
        Parameters:
        df (pd.DataFrame): DataFrame with features
        output_path (str or Path): Path to save file (.parquet or .csv)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        print(f"✓ Features saved to: {output_path}")
        print(f"✓ Shape: {df.shape}")
    
    def save_model_info(self, output_dir):
        """
        Save model configuration for reproducibility.
        
        Parameters:
        output_dir (str or Path): Directory to save metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'torch_version': torch.__version__
        }
        
        metadata_path = output_dir / 'image_features_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Model info saved to: {metadata_path}")


def extract_item_id_from_filename(filename):
    """
    Extract item ID from image filename.
    
    Example: '7433850_0.webp' -> '7433850'
    
    Parameters:
    filename (str): Image filename
    
    Returns:
    str: Item ID
    """
    # Remove extension and suffix (e.g., _0, _1, _2)
    base_name = Path(filename).stem
    item_id = base_name.split('_')[0]
    return item_id


def aggregate_features_by_item(df_features, method='mean'):
    """
    Aggregate image features when multiple images exist per item.
    
    Parameters:
    df_features (pd.DataFrame): DataFrame with image features
    method (str): Aggregation method ('mean', 'max', or 'first')
    
    Returns:
    pd.DataFrame: Aggregated features by item ID
    """
    # Extract item IDs from image names
    df_features['item_id'] = df_features['image_name'].apply(extract_item_id_from_filename)
    
    # Get feature columns
    feature_cols = [col for col in df_features.columns if col.startswith('img_feature_')]
    
    if method == 'mean':
        df_agg = df_features.groupby('item_id')[feature_cols].mean().reset_index()
    elif method == 'max':
        df_agg = df_features.groupby('item_id')[feature_cols].max().reset_index()
    elif method == 'first':
        df_agg = df_features.groupby('item_id')[feature_cols].first().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")
    
    print(f"\n✓ Aggregated {len(df_features)} images into {len(df_agg)} items (method: {method})")
    
    return df_agg

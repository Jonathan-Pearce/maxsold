"""
Item Feature Engineering

This module provides a reusable class for transforming item details data
with text embeddings using TF-IDF + LSA.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import re
import pickle
from pathlib import Path


class ItemFeatureEngineer:
    """
    Feature engineering for item details data.
    
    Features created:
    - Text embeddings (64-dimensional) for title + description combined
    - current_bid_le_10_binary: Binary feature for bids <= $10
    - log_current_bid: Log-transformed current bid
    """
    
    def __init__(self, n_components=64, max_features=5000):
        """
        Initialize the feature engineer.
        
        Parameters:
        n_components (int): Number of embedding dimensions (default: 64)
        max_features (int): Maximum number of features for TF-IDF (default: 5000)
        """
        self.n_components = n_components
        self.max_features = max_features
        self.fitted = False
        
        # Models to be fitted
        self.tfidf_combined = None
        self.svd_combined = None
    
    def fit(self, df):
        """
        Fit the feature engineer (learn TF-IDF vocabulary and SVD components).
        
        Parameters:
        df (pd.DataFrame): Training data with 'title' and 'description' columns
        
        Returns:
        self
        """
        print(f"\nFitting ItemFeatureEngineer...")
        print(f"  - n_components: {self.n_components}")
        print(f"  - max_features: {self.max_features}")
        
        # Prepare combined text
        titles = df['title'].fillna('').astype(str).apply(self._clean_text).tolist()
        descriptions = df['description'].fillna('').astype(str).apply(self._clean_text).tolist()
        combined_texts = [f"{t} {d}" for t, d in zip(titles, descriptions)]
        
        # Fit TF-IDF
        print(f"  - Vectorizing {len(combined_texts)} texts...")
        self.tfidf_combined = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        combined_tfidf_matrix = self.tfidf_combined.fit_transform(combined_texts)
        print(f"  - TF-IDF matrix shape: {combined_tfidf_matrix.shape}")
        
        # Fit SVD
        print(f"  - Applying LSA to reduce to {self.n_components} dimensions...")
        self.svd_combined = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd_combined.fit(combined_tfidf_matrix)
        print(f"  - Explained variance: {self.svd_combined.explained_variance_ratio_.sum():.4f}")
        
        self.fitted = True
        print(f"✓ Fitting complete")
        return self
    
    def transform(self, df):
        """
        Transform item data with feature engineering.
        
        Parameters:
        df (pd.DataFrame): Input dataframe with item data
        
        Returns:
        pd.DataFrame: Dataframe with engineered features
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Call fit() first.")
        
        df_engineered = df.copy()
        
        # Create embeddings
        print(f"\nTransforming {len(df)} items...")
        
        # Prepare combined text
        titles = df_engineered['title'].fillna('').astype(str).apply(self._clean_text).tolist()
        descriptions = df_engineered['description'].fillna('').astype(str).apply(self._clean_text).tolist()
        combined_texts = [f"{t} {d}" for t, d in zip(titles, descriptions)]
        
        # Transform with TF-IDF
        print(f"  - Creating TF-IDF vectors...")
        combined_tfidf_matrix = self.tfidf_combined.transform(combined_texts)
        
        # Transform with SVD
        print(f"  - Applying SVD...")
        combined_embeddings = self.svd_combined.transform(combined_tfidf_matrix)
        
        # Normalize embeddings
        combined_embeddings = normalize(combined_embeddings, norm='l2')
        
        # Add embeddings as columns
        for i in range(self.n_components):
            df_engineered[f'combined_emb_{i}'] = combined_embeddings[:, i]
        
        print(f"  - Created {self.n_components} embedding features")
        
        # Bid features
        df_engineered['current_bid_le_10_binary'] = np.where(
            df_engineered['current_bid'] <= 10, 1, 0
        )
        df_engineered['log_current_bid'] = np.log1p(df_engineered['current_bid'])
        
        print(f"✓ Transformation complete")
        return df_engineered
    
    def fit_transform(self, df):
        """
        Fit and transform in one step.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        
        Returns:
        pd.DataFrame: Transformed dataframe
        """
        self.fit(df)
        return self.transform(df)
    
    def get_model_columns(self):
        """
        Get the list of columns to keep for modeling.
        
        Returns:
        list: Column names for modeling
        """
        columns = [
            'id', 'auction_id', 'viewed', 'current_bid', 'bid_count', 
            'bidding_extended', 'current_bid_le_10_binary', 'log_current_bid'
        ]
        
        # Add embedding columns
        columns.extend([f'combined_emb_{i}' for i in range(self.n_components)])
        
        return columns
    
    def save_models(self, output_dir):
        """
        Save fitted models to disk for later use.
        
        Parameters:
        output_dir (str or Path): Directory to save models
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before saving")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save TF-IDF vectorizer
        tfidf_path = output_dir / 'combined_tfidf_vectorizer.pkl'
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf_combined, f)
        
        # Save SVD model
        svd_path = output_dir / 'combined_svd_model.pkl'
        with open(svd_path, 'wb') as f:
            pickle.dump(self.svd_combined, f)
        
        # Save metadata
        metadata = {
            'n_components': self.n_components,
            'max_features': self.max_features,
            'embedding_dimension': self.n_components
        }
        metadata_path = output_dir / 'embeddings_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✓ Models saved to: {output_dir}")
        print(f"  - {tfidf_path.name}")
        print(f"  - {svd_path.name}")
        print(f"  - {metadata_path.name}")
    
    def load_models(self, model_dir):
        """
        Load fitted models from disk.
        
        Parameters:
        model_dir (str or Path): Directory containing saved models
        """
        model_dir = Path(model_dir)
        
        # Load TF-IDF vectorizer
        tfidf_path = model_dir / 'combined_tfidf_vectorizer.pkl'
        with open(tfidf_path, 'rb') as f:
            self.tfidf_combined = pickle.load(f)
        
        # Load SVD model
        svd_path = model_dir / 'combined_svd_model.pkl'
        with open(svd_path, 'rb') as f:
            self.svd_combined = pickle.load(f)
        
        # Load metadata
        metadata_path = model_dir / 'embeddings_metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.n_components = metadata['n_components']
        self.max_features = metadata['max_features']
        self.fitted = True
        
        print(f"\n✓ Models loaded from: {model_dir}")
        print(f"  - n_components: {self.n_components}")
        print(f"  - max_features: {self.max_features}")
    
    @staticmethod
    def _clean_text(text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ''
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        return text.lower().strip()

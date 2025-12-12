import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import os

# Uses TF-IDF + TruncatedSVD (LSA) to create compact 64-dimensional embeddings
# Memory efficient alternative to sentence transformers
# Generates embeddings for title, description, and combined text
# Saves embeddings as both DataFrame columns and separate .npy files
# Calculates cosine similarity between title and description embeddings

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ''
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

def create_text_embeddings_small(input_path, output_dir=None, n_components=64, max_features=5000):
    """
    Convert title and description columns into compact 64-dimensional embeddings using TF-IDF + LSA.
    
    Parameters:
    input_path (str): Path to the parquet file
    output_dir (str): Directory to save embeddings (default: same as input)
    n_components (int): Number of embedding dimensions (default: 64)
    max_features (int): Maximum number of features for TF-IDF (default: 5000)
    
    Returns:
    pd.DataFrame: Dataframe with embedding features added
    """
    # Load the dataset
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    df_engineered = df.copy()
    
    """ # ========== TITLE EMBEDDINGS ==========
    if 'title' in df_engineered.columns:
        print("\n" + "="*50)
        print(f"Creating TITLE embeddings ({n_components} dimensions)...")
        print("="*50)
        
        # Clean titles
        titles = df_engineered['title'].fillna('').astype(str).apply(clean_text).tolist()
        
        # Create TF-IDF vectors
        print(f"Vectorizing {len(titles)} titles...")
        tfidf_title = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        title_tfidf_matrix = tfidf_title.fit_transform(titles)
        print(f"TF-IDF matrix shape: {title_tfidf_matrix.shape}")
        
        # Apply LSA (Latent Semantic Analysis) for dimensionality reduction
        print(f"Applying LSA to reduce to {n_components} dimensions...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        title_embeddings = svd.fit_transform(title_tfidf_matrix)
        print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
        
        # Normalize embeddings
        title_embeddings = normalize(title_embeddings, norm='l2')
        
        # Add embeddings as columns
        embedding_dim = title_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'title_emb_{i}'] = title_embeddings[:, i]
        
        print(f"Created {embedding_dim} title embedding features")
        
        # Save embeddings and models
        title_emb_path = os.path.join(output_dir, 'title_embeddings_small.npy')
        np.save(title_emb_path, title_embeddings)
        
        tfidf_path = os.path.join(output_dir, 'title_tfidf_vectorizer_small.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_title, f)
        
        svd_path = os.path.join(output_dir, 'title_svd_model_small.pkl')
        with open(svd_path, 'wb') as f:
            pickle.dump(svd, f)
        
        print(f"Saved title embeddings to: {title_emb_path}")
    
    # ========== DESCRIPTION EMBEDDINGS ==========
    if 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print(f"Creating DESCRIPTION embeddings ({n_components} dimensions)...")
        print("="*50)
        
        # Clean descriptions
        descriptions = df_engineered['description'].fillna('').astype(str).apply(clean_text).tolist()
        
        # Create TF-IDF vectors
        print(f"Vectorizing {len(descriptions)} descriptions...")
        tfidf_desc = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        desc_tfidf_matrix = tfidf_desc.fit_transform(descriptions)
        print(f"TF-IDF matrix shape: {desc_tfidf_matrix.shape}")
        
        # Apply LSA
        print(f"Applying LSA to reduce to {n_components} dimensions...")
        svd_desc = TruncatedSVD(n_components=n_components, random_state=42)
        description_embeddings = svd_desc.fit_transform(desc_tfidf_matrix)
        print(f"Explained variance ratio: {svd_desc.explained_variance_ratio_.sum():.4f}")
        
        # Normalize embeddings
        description_embeddings = normalize(description_embeddings, norm='l2')
        
        # Add embeddings as columns
        embedding_dim = description_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'description_emb_{i}'] = description_embeddings[:, i]
        
        print(f"Created {embedding_dim} description embedding features")
        
        # Save embeddings and models
        desc_emb_path = os.path.join(output_dir, 'description_embeddings_small.npy')
        np.save(desc_emb_path, description_embeddings)
        
        tfidf_path = os.path.join(output_dir, 'description_tfidf_vectorizer_small.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_desc, f)
        
        svd_path = os.path.join(output_dir, 'description_svd_model_small.pkl')
        with open(svd_path, 'wb') as f:
            pickle.dump(svd_desc, f)
        
        print(f"Saved description embeddings to: {desc_emb_path}") """
    
    # ========== COMBINED TEXT EMBEDDINGS ==========
    if 'title' in df_engineered.columns and 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print(f"Creating COMBINED text embeddings ({n_components} dimensions)...")
        print("="*50)

        # Clean titles
        titles = df_engineered['title'].fillna('').astype(str).apply(clean_text).tolist()
        
        # Clean descriptions
        descriptions = df_engineered['description'].fillna('').astype(str).apply(clean_text).tolist()

        # Combine title and description
        combined_texts = [f"{t} {d}" for t, d in zip(titles, descriptions)]
        
        # Create TF-IDF vectors
        print(f"Vectorizing {len(combined_texts)} combined texts...")
        tfidf_combined = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        combined_tfidf_matrix = tfidf_combined.fit_transform(combined_texts)
        print(f"TF-IDF matrix shape: {combined_tfidf_matrix.shape}")
        
        # Apply LSA
        print(f"Applying LSA to reduce to {n_components} dimensions...")
        svd_combined = TruncatedSVD(n_components=n_components, random_state=42)
        combined_embeddings = svd_combined.fit_transform(combined_tfidf_matrix)
        print(f"Explained variance ratio: {svd_combined.explained_variance_ratio_.sum():.4f}")
        
        # Normalize embeddings
        combined_embeddings = normalize(combined_embeddings, norm='l2')
        
        # Add embeddings as columns
        embedding_dim = combined_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'combined_emb_{i}'] = combined_embeddings[:, i]
        
        print(f"Created {embedding_dim} combined embedding features")
        
        # Save embeddings and models
        combined_emb_path = os.path.join(output_dir, 'combined_embeddings_small.npy')
        np.save(combined_emb_path, combined_embeddings)
        
        tfidf_path = os.path.join(output_dir, 'combined_tfidf_vectorizer_small.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_combined, f)
        
        svd_path = os.path.join(output_dir, 'combined_svd_model_small.pkl')
        with open(svd_path, 'wb') as f:
            pickle.dump(svd_combined, f)
        
        print(f"Saved combined embeddings to: {combined_emb_path}")
        
        # Calculate cosine similarity between title and description
        #print("\nCalculating title-description similarity...")
        #title_desc_similarity = (title_embeddings * description_embeddings).sum(axis=1)
        #df_engineered['title_desc_similarity'] = title_desc_similarity
        #print("Added title-description cosine similarity feature")
    
    print("\n" + "="*50)
    print(f"Final dataset shape: {df_engineered.shape}")
    print(f"Total new features created: {df_engineered.shape[1] - df.shape[1]}")
    print("="*50)
    
    # Save metadata
    metadata = {
        'method': 'TF-IDF + LSA',
        'n_components': n_components,
        'max_features': max_features,
        'embedding_dimension': n_components,
        'num_samples': len(df_engineered),
        'columns': df_engineered.columns.tolist()
    }
    
    metadata_path = os.path.join(output_dir, 'embeddings_metadata_small.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\nSaved metadata to: {metadata_path}")
    
    return df_engineered

def bid_features(df):

    #create binary current_bid feature <= 10 or > 10    
    df['current_bid_le_10_binary'] = np.where(df['current_bid'] <= 10 , 1, 0)

    return df


# Main execution
if __name__ == "__main__":
    input_file = '/workspaces/maxsold/data/raw_data/items_details/items_details_20251201.parquet'
    output_directory = '/workspaces/maxsold/data/engineered_data/items_details/embeddings/'
    
    # Create compact 64-dimensional embeddings using TF-IDF + LSA
    df_with_embeddings = create_text_embeddings_small(
        input_path=input_file,
        output_dir=output_directory,
        n_components=64,      # Compact embedding size
        max_features=5000     # Maximum vocabulary size
    )

    # Add bid features
    df_with_embeddings = bid_features(df_with_embeddings)

    # Keep only relevant columns for modeling (including all embedding columns)
    columns_to_keep = [
        'id', 'auction_id', 'viewed', 'current_bid', 'bid_count', 'bidding_extended', 'current_bid_le_10_binary'
    ] + [col for col in df_with_embeddings.columns if '_emb_' in col
    ]
    df_with_embeddings_model = df_with_embeddings[columns_to_keep]
    
    # Save the full dataset with embeddings
    output_file = '/workspaces/maxsold/data/engineered_data/items_details/items_details_20251201_with_embeddings_small.parquet'
    df_with_embeddings_model.to_parquet(output_file, index=False)
    print(f"\nFull dataset with compact embeddings saved to: {output_file}")
    
    # Display summary
    print("\n" + "="*50)
    print("EMBEDDING SUMMARY")
    print("="*50)
    
    embedding_cols = [col for col in df_with_embeddings_model.columns if '_emb_' in col]
    print(f"\nTotal embedding features: {len(embedding_cols)}")
    print(f"Total embedding columns: title={64}, description={64}, combined={64}")
    
    #subset data to columns not kept for modeling and print first few rows
    columns_not_kept = [col for col in df_with_embeddings.columns if col not in columns_to_keep]
    print("\nColumns not kept for modeling:")
    print(columns_not_kept)
    print("\nFirst few rows of columns not kept for modeling:")
    print(df_with_embeddings[columns_not_kept].head())
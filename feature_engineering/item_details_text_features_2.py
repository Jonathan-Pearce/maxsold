import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import re
import pickle
import os
from tqdm import tqdm

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

def create_tfidf_embeddings(input_path, output_dir=None, n_components=128, max_features=5000):
    """
    Convert title and description columns into TF-IDF based embeddings.
    
    Parameters:
    input_path (str): Path to the parquet file
    output_dir (str): Directory to save embeddings (default: same as input)
    n_components (int): Number of dimensions for LSA reduction (set to None to skip)
    max_features (int): Maximum number of features for TF-IDF
    
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
    
    # ========== TITLE EMBEDDINGS ==========
    if 'title' in df_engineered.columns:
        print("\n" + "="*50)
        print("Creating TITLE TF-IDF embeddings...")
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
        
        # Optional: Apply LSA (Latent Semantic Analysis) for dimensionality reduction
        if n_components and n_components < title_tfidf_matrix.shape[1]:
            print(f"Applying LSA to reduce to {n_components} dimensions...")
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            title_embeddings = svd.fit_transform(title_tfidf_matrix)
            print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
            
            # Save SVD model
            svd_path = os.path.join(output_dir, 'title_svd_model.pkl')
            with open(svd_path, 'wb') as f:
                pickle.dump(svd, f)
        else:
            title_embeddings = title_tfidf_matrix.toarray()
        
        # Normalize embeddings
        title_embeddings = normalize(title_embeddings, norm='l2')
        
        # Add embeddings as columns
        embedding_dim = title_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'title_emb_{i}'] = title_embeddings[:, i]
        
        print(f"Created {embedding_dim} title embedding features")
        
        # Save embeddings and vectorizer
        title_emb_path = os.path.join(output_dir, 'title_embeddings.npy')
        np.save(title_emb_path, title_embeddings)
        
        tfidf_path = os.path.join(output_dir, 'title_tfidf_vectorizer.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_title, f)
        
        print(f"Saved title embeddings to: {title_emb_path}")
    
    # ========== DESCRIPTION EMBEDDINGS ==========
    if 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print("Creating DESCRIPTION TF-IDF embeddings...")
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
        
        # Optional: Apply LSA
        if n_components and n_components < desc_tfidf_matrix.shape[1]:
            print(f"Applying LSA to reduce to {n_components} dimensions...")
            svd_desc = TruncatedSVD(n_components=n_components, random_state=42)
            description_embeddings = svd_desc.fit_transform(desc_tfidf_matrix)
            print(f"Explained variance ratio: {svd_desc.explained_variance_ratio_.sum():.4f}")
            
            # Save SVD model
            svd_path = os.path.join(output_dir, 'description_svd_model.pkl')
            with open(svd_path, 'wb') as f:
                pickle.dump(svd_desc, f)
        else:
            description_embeddings = desc_tfidf_matrix.toarray()
        
        # Normalize embeddings
        description_embeddings = normalize(description_embeddings, norm='l2')
        
        # Add embeddings as columns
        embedding_dim = description_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'description_emb_{i}'] = description_embeddings[:, i]
        
        print(f"Created {embedding_dim} description embedding features")
        
        # Save embeddings and vectorizer
        desc_emb_path = os.path.join(output_dir, 'description_embeddings.npy')
        np.save(desc_emb_path, description_embeddings)
        
        tfidf_path = os.path.join(output_dir, 'description_tfidf_vectorizer.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_desc, f)
        
        print(f"Saved description embeddings to: {desc_emb_path}")
    
    # ========== COMBINED TEXT EMBEDDINGS ==========
    if 'title' in df_engineered.columns and 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print("Creating COMBINED text embeddings...")
        print("="*50)
        
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
        
        # Optional: Apply LSA
        if n_components and n_components < combined_tfidf_matrix.shape[1]:
            print(f"Applying LSA to reduce to {n_components} dimensions...")
            svd_combined = TruncatedSVD(n_components=n_components, random_state=42)
            combined_embeddings = svd_combined.fit_transform(combined_tfidf_matrix)
            print(f"Explained variance ratio: {svd_combined.explained_variance_ratio_.sum():.4f}")
            
            # Save SVD model
            svd_path = os.path.join(output_dir, 'combined_svd_model.pkl')
            with open(svd_path, 'wb') as f:
                pickle.dump(svd_combined, f)
        else:
            combined_embeddings = combined_tfidf_matrix.toarray()
        
        # Normalize embeddings
        combined_embeddings = normalize(combined_embeddings, norm='l2')
        
        # Add embeddings as columns
        embedding_dim = combined_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'combined_emb_{i}'] = combined_embeddings[:, i]
        
        print(f"Created {embedding_dim} combined embedding features")
        
        # Save embeddings and vectorizer
        combined_emb_path = os.path.join(output_dir, 'combined_embeddings.npy')
        np.save(combined_emb_path, combined_embeddings)
        
        tfidf_path = os.path.join(output_dir, 'combined_tfidf_vectorizer.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_combined, f)
        
        print(f"Saved combined embeddings to: {combined_emb_path}")
        
        # Calculate cosine similarity between title and description
        print("\nCalculating title-description similarity...")
        title_desc_similarity = (title_embeddings * description_embeddings).sum(axis=1)
        df_engineered['title_desc_similarity'] = title_desc_similarity
        print("Added title-description cosine similarity feature")
    
    print("\n" + "="*50)
    print(f"Final dataset shape: {df_engineered.shape}")
    print(f"Total new features created: {df_engineered.shape[1] - df.shape[1]}")
    print("="*50)
    
    # Save metadata
    metadata = {
        'method': 'TF-IDF + LSA',
        'n_components': n_components,
        'max_features': max_features,
        'embedding_dimension': embedding_dim,
        'num_samples': len(df_engineered),
        'columns': df_engineered.columns.tolist()
    }
    
    metadata_path = os.path.join(output_dir, 'embeddings_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\nSaved metadata to: {metadata_path}")
    
    return df_engineered


# Main execution
if __name__ == "__main__":
    input_file = '/workspaces/maxsold/data/items_details/items_details_20251201.parquet'
    output_directory = '/workspaces/maxsold/data/items_details/embeddings/'
    
    # Create embeddings using TF-IDF + LSA
    df_with_embeddings = create_tfidf_embeddings(
        input_path=input_file,
        output_dir=output_directory,
        n_components=128,  # Reduced dimensions (set to None to keep all features)
        max_features=5000  # Maximum vocabulary size
    )
    
    # Save the full dataset with embeddings
    output_file = '/workspaces/maxsold/data/items_details/items_details_20251201_with_embeddings.parquet'
    df_with_embeddings.to_parquet(output_file, index=False)
    print(f"\nFull dataset with embeddings saved to: {output_file}")
    
    # Display summary
    print("\n" + "="*50)
    print("EMBEDDING SUMMARY")
    print("="*50)
    
    embedding_cols = [col for col in df_with_embeddings.columns if '_emb_' in col]
    print(f"\nTotal embedding features: {len(embedding_cols)}")
    
    if 'title_desc_similarity' in df_with_embeddings.columns:
        print(f"\nTitle-Description Similarity Statistics:")
        print(df_with_embeddings['title_desc_similarity'].describe())
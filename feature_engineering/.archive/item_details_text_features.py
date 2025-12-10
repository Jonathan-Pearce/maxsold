import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import pickle
import os

#Uses Sentence Transformers to create semantic embeddings
#Generates embeddings for title, description, and combined text
#Saves embeddings as both DataFrame columns and separate .npy files
#Calculates cosine similarity between title and description embeddings
#Supports GPU acceleration if available
#Processes data in batches for memory efficiency
#Provides multiple model options (default: all-MiniLM-L6-v2 for 384-dim embeddings)

def create_text_embeddings(input_path, output_dir=None, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Convert title and description columns into word embeddings using sentence transformers.
    
    Parameters:
    input_path (str): Path to the parquet file
    output_dir (str): Directory to save embeddings (default: same as input)
    model_name (str): Sentence transformer model to use
    batch_size (int): Batch size for encoding
    
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
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load the sentence transformer model
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    df_engineered = df.copy()
    
    # ========== TITLE EMBEDDINGS ==========
    if 'title' in df_engineered.columns:
        print("\n" + "="*50)
        print("Creating TITLE embeddings...")
        print("="*50)
        
        # Clean and prepare titles
        titles = df_engineered['title'].fillna('').astype(str).tolist()
        
        # Generate embeddings in batches
        print(f"Encoding {len(titles)} titles...")
        title_embeddings = model.encode(
            titles,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings as columns
        embedding_dim = title_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'title_emb_{i}'] = title_embeddings[:, i]
        
        print(f"Created {embedding_dim} title embedding features")
        
        # Save embeddings separately
        title_emb_path = os.path.join(output_dir, 'title_embeddings.npy')
        np.save(title_emb_path, title_embeddings)
        print(f"Saved title embeddings to: {title_emb_path}")
    
    # ========== DESCRIPTION EMBEDDINGS ==========
    if 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print("Creating DESCRIPTION embeddings...")
        print("="*50)
        
        # Clean HTML and prepare descriptions
        import re
        descriptions = df_engineered['description'].fillna('').astype(str).apply(
            lambda x: re.sub(r'<[^>]+>', '', x)
        ).tolist()
        
        # Truncate very long descriptions (optional, to avoid memory issues)
        max_length = 512  # characters
        descriptions = [desc[:max_length] if len(desc) > max_length else desc for desc in descriptions]
        
        # Generate embeddings in batches
        print(f"Encoding {len(descriptions)} descriptions...")
        description_embeddings = model.encode(
            descriptions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings as columns
        embedding_dim = description_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'description_emb_{i}'] = description_embeddings[:, i]
        
        print(f"Created {embedding_dim} description embedding features")
        
        # Save embeddings separately
        desc_emb_path = os.path.join(output_dir, 'description_embeddings.npy')
        np.save(desc_emb_path, description_embeddings)
        print(f"Saved description embeddings to: {desc_emb_path}")
    
    # ========== COMBINED TEXT EMBEDDINGS ==========
    if 'title' in df_engineered.columns and 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print("Creating COMBINED text embeddings...")
        print("="*50)
        
        # Combine title and description
        combined_texts = [
            f"{title} {desc}" for title, desc in zip(titles, descriptions)
        ]
        
        # Generate embeddings
        print(f"Encoding {len(combined_texts)} combined texts...")
        combined_embeddings = model.encode(
            combined_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings as columns
        embedding_dim = combined_embeddings.shape[1]
        for i in range(embedding_dim):
            df_engineered[f'combined_emb_{i}'] = combined_embeddings[:, i]
        
        print(f"Created {embedding_dim} combined embedding features")
        
        # Save embeddings separately
        combined_emb_path = os.path.join(output_dir, 'combined_embeddings.npy')
        np.save(combined_emb_path, combined_embeddings)
        print(f"Saved combined embeddings to: {combined_emb_path}")
        
        # Calculate cosine similarity between title and description
        from sklearn.metrics.pairwise import cosine_similarity
        print("\nCalculating title-description similarity...")
        title_desc_similarity = []
        for t_emb, d_emb in zip(title_embeddings, description_embeddings):
            sim = cosine_similarity(t_emb.reshape(1, -1), d_emb.reshape(1, -1))[0, 0]
            title_desc_similarity.append(sim)
        
        df_engineered['title_desc_similarity'] = title_desc_similarity
        print("Added title-description cosine similarity feature")
    
    print("\n" + "="*50)
    print(f"Final dataset shape: {df_engineered.shape}")
    print(f"Total new features created: {df_engineered.shape[1] - df.shape[1]}")
    print("="*50)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'embedding_dimension': model.get_sentence_embedding_dimension(),
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
    
    # Create embeddings
    df_with_embeddings = create_text_embeddings(
        input_path=input_file,
        output_dir=output_directory,
        model_name='all-MiniLM-L6-v2',  # Fast and efficient (384 dimensions)
        # Alternative models:
        # 'all-mpnet-base-v2'  # Better quality but slower (768 dimensions)
        # 'paraphrase-MiniLM-L6-v2'  # Optimized for paraphrase detection
        batch_size=32
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
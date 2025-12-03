import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def nlp_feature_engineering(input_path):
    """
    Perform NLP feature engineering on title and description columns.
    
    Parameters:
    input_path (str): Path to the parquet file
    
    Returns:
    pd.DataFrame: Dataframe with NLP engineered features
    """
    # Load the dataset
    df = pd.read_parquet(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Create a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # ========== TITLE FEATURES ==========
    if 'title' in df_engineered.columns:
        print("\n" + "="*50)
        print("Processing TITLE column...")
        print("="*50)
        
        # Basic text statistics
        df_engineered['title_length'] = df_engineered['title'].fillna('').astype(str).str.len()
        df_engineered['title_word_count'] = df_engineered['title'].fillna('').astype(str).str.split().str.len()
        df_engineered['title_avg_word_length'] = df_engineered['title'].fillna('').astype(str).apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Character-based features
        df_engineered['title_uppercase_count'] = df_engineered['title'].fillna('').astype(str).str.count(r'[A-Z]')
        df_engineered['title_digit_count'] = df_engineered['title'].fillna('').astype(str).str.count(r'\d')
        df_engineered['title_special_char_count'] = df_engineered['title'].fillna('').astype(str).str.count(r'[^a-zA-Z0-9\s]')
        df_engineered['title_exclamation_count'] = df_engineered['title'].fillna('').astype(str).str.count(r'!')
        
        # Boolean features
        df_engineered['title_has_numbers'] = (df_engineered['title_digit_count'] > 0).astype(int)
        df_engineered['title_is_all_caps'] = df_engineered['title'].fillna('').astype(str).apply(
            lambda x: 1 if x.isupper() and len(x) > 0 else 0
        )
        
        # Brand/quality indicators (common keywords)
        brand_keywords = ['vintage', 'antique', 'new', 'used', 'rare', 'collectible', 'original', 'authentic']
        for keyword in brand_keywords:
            df_engineered[f'title_has_{keyword}'] = df_engineered['title'].fillna('').astype(str).str.lower().str.contains(keyword).astype(int)
        
        print(f"Created {len([col for col in df_engineered.columns if col.startswith('title_')])} title features")
    
    # ========== DESCRIPTION FEATURES ==========
    if 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print("Processing DESCRIPTION column...")
        print("="*50)
        
        # Clean HTML tags if present
        df_engineered['description_cleaned'] = df_engineered['description'].fillna('').astype(str).apply(
            lambda x: re.sub(r'<[^>]+>', '', x)
        )
        
        # Basic text statistics
        df_engineered['description_length'] = df_engineered['description_cleaned'].str.len()
        df_engineered['description_word_count'] = df_engineered['description_cleaned'].str.split().str.len()
        df_engineered['description_avg_word_length'] = df_engineered['description_cleaned'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        df_engineered['description_sentence_count'] = df_engineered['description_cleaned'].str.count(r'[.!?]+')
        
        # Character-based features
        df_engineered['description_uppercase_count'] = df_engineered['description_cleaned'].str.count(r'[A-Z]')
        df_engineered['description_digit_count'] = df_engineered['description_cleaned'].str.count(r'\d')
        df_engineered['description_special_char_count'] = df_engineered['description_cleaned'].str.count(r'[^a-zA-Z0-9\s]')
        
        # URL and contact features
        df_engineered['description_has_url'] = df_engineered['description_cleaned'].str.contains(
            r'http[s]?://|www\.', case=False, regex=True
        ).astype(int)
        df_engineered['description_has_email'] = df_engineered['description_cleaned'].str.contains(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', case=False, regex=True
        ).astype(int)
        df_engineered['description_has_phone'] = df_engineered['description_cleaned'].str.contains(
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', regex=True
        ).astype(int)
        
        # Condition keywords
        condition_keywords = ['excellent', 'good', 'fair', 'poor', 'mint', 'damaged', 'broken', 'working', 'functional']
        for keyword in condition_keywords:
            df_engineered[f'description_has_{keyword}'] = df_engineered['description_cleaned'].str.lower().str.contains(keyword).astype(int)
        
        # Measurement indicators
        df_engineered['description_has_measurements'] = df_engineered['description_cleaned'].str.contains(
            r'\d+\s*(inch|cm|mm|ft|meter|metre|")', case=False, regex=True
        ).astype(int)
        
        # Completeness indicators
        df_engineered['description_is_empty'] = (df_engineered['description_length'] == 0).astype(int)
        df_engineered['description_is_short'] = (df_engineered['description_word_count'] < 10).astype(int)
        
        print(f"Created {len([col for col in df_engineered.columns if col.startswith('description_')])} description features")
    
    # ========== COMBINED FEATURES ==========
    if 'title' in df_engineered.columns and 'description' in df_engineered.columns:
        print("\n" + "="*50)
        print("Creating COMBINED features...")
        print("="*50)
        
        # Ratio features
        df_engineered['title_desc_length_ratio'] = df_engineered['title_length'] / (df_engineered['description_length'] + 1)
        df_engineered['title_desc_word_ratio'] = df_engineered['title_word_count'] / (df_engineered['description_word_count'] + 1)
        
        # Overlap features
        df_engineered['title_desc_overlap'] = df_engineered.apply(
            lambda row: len(set(str(row['title']).lower().split()) & set(str(row['description_cleaned']).lower().split())),
            axis=1
        )
        
        print(f"Created {len([col for col in df_engineered.columns if col.startswith('title_desc_')])} combined features")
    
    print("\n" + "="*50)
    print(f"Final dataset shape: {df_engineered.shape}")
    print(f"Total new features created: {df_engineered.shape[1] - df.shape[1]}")
    print("="*50)
    
    return df_engineered


# Main execution
if __name__ == "__main__":
    input_file = '/workspaces/maxsold/data/items_details/items_details_20251201.parquet'
    
    # Perform NLP feature engineering
    df_processed = nlp_feature_engineering(input_file)
    
    # Save the processed data
    output_file = '/workspaces/maxsold/data/items_details/items_details_20251201_nlp_features.parquet'
    df_processed.to_parquet(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    # Display feature summary
    nlp_features = [col for col in df_processed.columns 
                    if col.startswith(('title_', 'description_'))]
    
    print(f"\n{'='*50}")
    print("NLP FEATURE SUMMARY")
    print(f"{'='*50}")
    print(f"\nTotal NLP features created: {len(nlp_features)}\n")
    
    # Show statistics for some key features
    print("Sample statistics:")
    stats_cols = ['title_length', 'title_word_count', 'description_length', 'description_word_count']
    available_stats = [col for col in stats_cols if col in df_processed.columns]
    if available_stats:
        print(df_processed[available_stats].describe())
    
    # Show sample of boolean features
    print("\nSample of boolean features (first 5 rows):")
    boolean_features = [col for col in nlp_features if df_processed[col].dtype in ['int64', 'int32'] 
                        and df_processed[col].nunique() <= 2][:10]
    if boolean_features:
        print(df_processed[boolean_features].head())
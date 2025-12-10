from pathlib import Path
import pandas as pd
import numpy as np
import re
from datetime import datetime

#This code creates the following features:

#auction_length_hours: Duration of auction in hours
#postal_code: Extracted Canadian postal code from removal_info
#postal_code_fsa: First 3 characters of postal code (Forward Sortation Area)
#intro_cleaned: HTML-stripped and cleaned intro text
#intro_length: Character length of cleaned intro
#pickup_day_of_week: 0=Monday, 6=Sunday
#pickup_day_name: Day name (Monday, Tuesday, etc.)
#pickup_is_weekend: 1 if Saturday/Sunday, 0 otherwise
#has_partner_url: 1 if partner_url exists, 0 otherwise


def feature_engineering(df):
    """
    Perform feature engineering on the auction dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with auction data
    
    Returns:
    pd.DataFrame: Dataframe with engineered features
    """
    # Create a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # 1. Feature: Length of auction in hours (ends - starts)
    df_engineered['starts'] = pd.to_datetime(df_engineered['starts'])
    df_engineered['ends'] = pd.to_datetime(df_engineered['ends'])
    df_engineered['auction_length_hours'] = (
        df_engineered['ends'] - df_engineered['starts']
    ).dt.total_seconds() / 3600
    
    # 2. Extract postal code from removal_info
    def extract_postal_code(text):
        if pd.isna(text):
            return None
        # Canadian postal code pattern: A1A 1A1 or A1A1A1
        match = re.search(r'[A-Z]\d[A-Z]\s?\d[A-Z]\d', str(text), re.IGNORECASE)
        if match:
            return match.group(0).upper().replace(' ', '')
        return None
    
    df_engineered['postal_code'] = df_engineered['removal_info'].apply(extract_postal_code)
    
    # 2b. Extract Forward Sortation Area (first 3 characters of postal code)
    df_engineered['postal_code_fsa'] = df_engineered['postal_code'].apply(
        lambda x: x[:3] if pd.notna(x) and len(str(x)) >= 3 else None
    )
    
    # 3. Clean and extract text from intro
    def clean_intro(text):
        if pd.isna(text):
            return ''
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()
    
    df_engineered['intro_cleaned'] = df_engineered['intro'].apply(clean_intro)
    df_engineered['intro_length'] = df_engineered['intro_cleaned'].str.len()
    
    # 4. Extract day of week from pickup_time
    df_engineered['pickup_time'] = pd.to_datetime(df_engineered['pickup_time'], errors='coerce')
    df_engineered['pickup_day_of_week'] = df_engineered['pickup_time'].dt.dayofweek
    df_engineered['pickup_day_name'] = df_engineered['pickup_time'].dt.day_name()
    df_engineered['pickup_is_weekend'] = df_engineered['pickup_day_of_week'].isin([5, 6]).astype(int)
    
    # 5. Boolean variable for partner_url presence
    #df_engineered['has_partner_url'] = df_engineered['partner_url'].notna().astype(int)
    df_engineered['has_partner_url'] = np.where(df_engineered['partner_url']!="", 1, 0) 
    
    return df_engineered


if __name__ == "__main__":
    # Load raw data
    data_path = Path('/workspaces/maxsold/data/raw_data/auction_details/auction_details_20251201.parquet')
    df = pd.read_parquet(data_path)
    print("Original data shape:", df.shape)
    print("\nFirst few rows of original data:")
    print(df.head())
    
    # Process features
    df_processed = feature_engineering(df)
    print("\n" + "="*60)
    print("Processed data shape:", df_processed.shape)
    print("\nFirst few rows of processed data:")
    print(df_processed.head())
    
    # Save to engineered data directory
    output_dir = Path('/workspaces/maxsold/data/engineered_data/auction_details')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'auction_details_20251201_engineered.parquet'
    df_processed.to_parquet(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved processed data to: {output_path}")
    print(f"Output file size: {output_path.stat().st_size / 1024:.2f} KB")
    
    # Display new feature columns
    new_cols = ['auction_length_hours', 'postal_code', 'postal_code_fsa', 'intro_cleaned', 'intro_length',
                'pickup_day_of_week', 'pickup_day_name', 'pickup_is_weekend', 'has_partner_url']
    available_new_cols = [col for col in new_cols if col in df_processed.columns]
    print(f"\nNew features created: {available_new_cols}")
    print("\nSample of new features:")
    print(df_processed[available_new_cols].head())
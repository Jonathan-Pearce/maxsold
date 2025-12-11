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
    
    # 2b. Extract Postal District (first 1 characters of postal code)
    df_engineered['postal_code_pd'] = df_engineered['postal_code'].apply(
        lambda x: x[:1] if pd.notna(x) and len(str(x)) >= 1 else None
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
    #df_engineered['pickup_time_hour'] = df_engineered['pickup_time'].dt.hour
    
    # 5. Boolean variable for partner_url presence
    #df_engineered['has_partner_url'] = df_engineered['partner_url'].notna().astype(int)
    df_engineered['has_partner_url'] = np.where(df_engineered['partner_url']!="", 1, 0) 

    # 6. One-hot encoding for pickup_day_name
    pickup_day_dummies = pd.get_dummies(df_engineered['pickup_day_name'], prefix='pickup_day')
    df_engineered = pd.concat([df_engineered, pickup_day_dummies], axis=1)

    # 7. One-hot encoding for postal_code_pd
    postal_code_pd_dummies = pd.get_dummies(
        df_engineered['postal_code_pd'], 
        prefix='postal_code_pd',
        drop_first=False
    )
    df_engineered = pd.concat([df_engineered, postal_code_pd_dummies], axis=1)

    # 8. extract pickup_time_hour from removal_info (text before first instance of 'AM' or 'PM')
    def extract_pickup_hour(text):
        if pd.isna(text):
            return None
        match = re.search(r'(\d{1,2}):\d{2}\s?(AM|PM)', str(text), re.IGNORECASE)
        if match:
            hour = int(match.group(1))
            period = match.group(2).upper()
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
            return hour
        return None
    
    df_engineered['pickup_time_hour'] = df_engineered['removal_info'].apply(extract_pickup_hour)
    
    # 9. create binary feature for auctions with pickup time during work hours (9 AM to 5 PM, Monday to Friday)
    df_engineered['pickup_during_work_hours'] = ((df_engineered['pickup_time_hour'] >= 9) & (df_engineered['pickup_time_hour'] <= 17) & (~df_engineered['pickup_day_of_week'].isin([5, 6]))).astype(int)

    # 10. create feature for seller managed, partner managed auctions from title
    df_engineered['is_seller_managed'] = df_engineered['title'].str.contains('SELLER MANAGED', case=False, na=False).astype(int)
    #df_engineered['is_partner_managed'] = df_engineered['title'].str.contains('PARTNER MANAGED', case=False, na=False).astype(int)

    # 11. create feature for condo, storage unit auction from title
    df_engineered['is_condo_auction'] = df_engineered['title'].str.contains('(CONDO)', case=False, na=False).astype(int)
    df_engineered['is_storage_unit_auction'] = df_engineered['title'].str.contains('(STORAGE)', case=False, na=False).astype(int)

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

    # Keep only relevant columns for modeling
    columns_to_keep = [
        'auction_id', 'auction_length_hours', 'catalog_lots',
        'intro_length',
        'pickup_is_weekend', 'has_partner_url', 'pickup_day_Friday', 'pickup_day_Monday',
        'pickup_day_Saturday', 'pickup_day_Sunday', 'pickup_day_Thursday', 'pickup_day_Tuesday',
        'pickup_day_Wednesday', 'postal_code_pd_K', 'postal_code_pd_L', 'postal_code_pd_M',
        'postal_code_pd_N', 'postal_code_pd_P', 'pickup_time_hour', 'pickup_during_work_hours',
        'is_seller_managed', 'is_partner_managed', 'is_condo_auction', 'is_storage_unit_auction'
    ]
    df_processed_model = df_processed[columns_to_keep]

    # Save to engineered data directory
    output_dir = Path('/workspaces/maxsold/data/engineered_data/auction_details')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'auction_details_20251201_engineered.parquet'
    df_processed_model.to_parquet(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved processed data to: {output_path}")
    print(f"Output file size: {output_path.stat().st_size / 1024:.2f} KB")
    
    print("\nColumns in processed data:")
    print(df_processed_model.columns.tolist())


    #subset data to columns not kept for modeling and print first few rows
    columns_not_kept = [col for col in df_processed.columns if col not in columns_to_keep]
    print("\nColumns not kept for modeling:")
    print(columns_not_kept)
    print("\nFirst few rows of columns not kept for modeling:")
    print(df_processed[columns_not_kept].head())
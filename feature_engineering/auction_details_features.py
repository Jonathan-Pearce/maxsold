from pathlib import Path
import pandas as pd
import numpy as np
import re
from datetime import datetime

#This code creates the following features:

#auction_length_hours: Duration of auction in hours
#postal_code: Extracted Canadian postal code from removal_info
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
    df_engineered['has_partner_url'] = df_engineered['partner_url'].notna().astype(int)
    
    return df_engineered


# Example usage:
# df = pd.read_csv('your_auction_data.csv')
# df_processed = feature_engineering(df)
# print(df_processed[['auction_length_hours', 'postal_code', 'pickup_day_name', 'has_partner_url']].head())

data_path = Path('/workspaces/maxsold/data/auction_details/auction_details_20251201.parquet')
df = pd.read_parquet(data_path)
df_processed = feature_engineering(df)
print(df_processed.head())
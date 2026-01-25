"""
Auction Feature Engineering

This module provides a reusable class for transforming auction details data.
Can be used for batch processing or real-time inference.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


class AuctionFeatureEngineer:
    """
    Feature engineering for auction details data.
    
    Features created:
    - auction_length_hours: Duration of auction in hours
    - postal_code: Extracted Canadian postal code from removal_info
    - postal_code_pd: Postal district (first character)
    - intro_cleaned: HTML-stripped and cleaned intro text
    - intro_length: Character length of cleaned intro
    - pickup_day_of_week: 0=Monday, 6=Sunday
    - pickup_day_name: Day name (Monday, Tuesday, etc.)
    - pickup_is_weekend: 1 if Saturday/Sunday, 0 otherwise
    - has_partner_url: 1 if partner_url exists, 0 otherwise
    - pickup_time_hour: Extracted hour from removal_info
    - pickup_during_work_hours: Binary for work hours pickup
    - is_seller_managed: Extracted from title
    - is_condo_auction: Extracted from title
    - is_storage_unit_auction: Extracted from title
    - One-hot encoded pickup days
    - One-hot encoded postal districts
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.fitted = False
        self.postal_districts = None
        self.pickup_days = None
    
    def fit(self, df):
        """
        Fit the feature engineer (learn categories for one-hot encoding).
        
        Parameters:
        df (pd.DataFrame): Training data
        
        Returns:
        self
        """
        df_temp = df.copy()
        
        # Learn postal districts
        df_temp['postal_code'] = df_temp['removal_info'].apply(self._extract_postal_code)
        df_temp['postal_code_pd'] = df_temp['postal_code'].apply(
            lambda x: x[:1] if pd.notna(x) and len(str(x)) >= 1 else None
        )
        self.postal_districts = sorted(df_temp['postal_code_pd'].dropna().unique().tolist())
        
        # Learn pickup days
        df_temp['pickup_time'] = pd.to_datetime(df_temp['pickup_time'], errors='coerce')
        df_temp['pickup_day_name'] = df_temp['pickup_time'].dt.day_name()
        self.pickup_days = sorted(df_temp['pickup_day_name'].dropna().unique().tolist())
        
        self.fitted = True
        return self
    
    def transform(self, df):
        """
        Transform auction data with feature engineering.
        
        Parameters:
        df (pd.DataFrame): Input dataframe with auction data
        
        Returns:
        pd.DataFrame: Dataframe with engineered features
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform. Call fit() first.")
        
        df_engineered = df.copy()
        
        # 1. Auction length in hours
        df_engineered['starts'] = pd.to_datetime(df_engineered['starts'])
        df_engineered['ends'] = pd.to_datetime(df_engineered['ends'])
        df_engineered['auction_length_hours'] = (
            df_engineered['ends'] - df_engineered['starts']
        ).dt.total_seconds() / 3600
        
        # 2. Extract postal code
        df_engineered['postal_code'] = df_engineered['removal_info'].apply(self._extract_postal_code)
        df_engineered['postal_code_pd'] = df_engineered['postal_code'].apply(
            lambda x: x[:1] if pd.notna(x) and len(str(x)) >= 1 else None
        )
        
        # 3. Clean intro text
        df_engineered['intro_cleaned'] = df_engineered['intro'].apply(self._clean_intro)
        df_engineered['intro_length'] = df_engineered['intro_cleaned'].str.len()
        
        # 4. Pickup time features
        df_engineered['pickup_time'] = pd.to_datetime(df_engineered['pickup_time'], errors='coerce')
        df_engineered['pickup_day_of_week'] = df_engineered['pickup_time'].dt.dayofweek
        df_engineered['pickup_day_name'] = df_engineered['pickup_time'].dt.day_name()
        df_engineered['pickup_is_weekend'] = df_engineered['pickup_day_of_week'].isin([5, 6]).astype(int)
        
        # 5. Partner URL presence
        df_engineered['has_partner_url'] = np.where(df_engineered['partner_url'] != "", 1, 0)
        
        # 6. One-hot encoding for pickup_day_name
        for day in self.pickup_days:
            col_name = f'pickup_day_{day}'
            df_engineered[col_name] = (df_engineered['pickup_day_name'] == day).astype(int)
        
        # 7. One-hot encoding for postal_code_pd
        for district in self.postal_districts:
            col_name = f'postal_code_pd_{district}'
            df_engineered[col_name] = (df_engineered['postal_code_pd'] == district).astype(int)
        
        # 8. Extract pickup hour
        df_engineered['pickup_time_hour'] = df_engineered['removal_info'].apply(self._extract_pickup_hour)
        
        # 9. Pickup during work hours
        df_engineered['pickup_during_work_hours'] = (
            (df_engineered['pickup_time_hour'] >= 9) & 
            (df_engineered['pickup_time_hour'] <= 17) & 
            (~df_engineered['pickup_day_of_week'].isin([5, 6]))
        ).astype(int)
        
        # 10. Seller managed
        df_engineered['is_seller_managed'] = df_engineered['title'].str.contains(
            'SELLER MANAGED', case=False, na=False
        ).astype(int)
        
        # 11. Condo and storage unit auctions
        df_engineered['is_condo_auction'] = df_engineered['title'].str.contains(
            '(CONDO)', case=False, na=False
        ).astype(int)
        df_engineered['is_storage_unit_auction'] = df_engineered['title'].str.contains(
            '(STORAGE)', case=False, na=False
        ).astype(int)
        
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
            'auction_id', 'auction_length_hours', 'catalog_lots',
            'intro_length', 'pickup_is_weekend', 'has_partner_url',
            'pickup_time_hour', 'pickup_during_work_hours',
            'is_seller_managed', 'is_condo_auction', 'is_storage_unit_auction'
        ]
        
        # Add one-hot encoded columns
        if self.pickup_days:
            columns.extend([f'pickup_day_{day}' for day in self.pickup_days])
        if self.postal_districts:
            columns.extend([f'postal_code_pd_{district}' for district in self.postal_districts])
        
        return columns
    
    @staticmethod
    def _extract_postal_code(text):
        """Extract Canadian postal code from text"""
        if pd.isna(text):
            return None
        match = re.search(r'[A-Z]\d[A-Z]\s?\d[A-Z]\d', str(text), re.IGNORECASE)
        if match:
            return match.group(0).upper().replace(' ', '')
        return None
    
    @staticmethod
    def _clean_intro(text):
        """Clean and extract text from intro"""
        if pd.isna(text):
            return ''
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def _extract_pickup_hour(text):
        """Extract pickup hour from removal_info"""
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

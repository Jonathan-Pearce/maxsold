"""
Dataset Merger

This module provides a reusable class for merging auction, item, and enriched item datasets.
"""

import pandas as pd
import gc
from pathlib import Path


class DatasetMerger:
    """
    Merge auction and item datasets into a final combined dataset.
    
    Strategy:
    1. Merge auction data (if multiple sources)
    2. Merge items with auction data (on auction_id)
    3. Merge enriched item features (on item_id)
    """
    
    def __init__(self):
        """Initialize the dataset merger"""
        pass
    
    def merge(self, df_auction, df_items, df_enriched=None):
        """
        Merge datasets.
        
        Parameters:
        df_auction (pd.DataFrame): Auction features data
        df_items (pd.DataFrame): Item features data
        df_enriched (pd.DataFrame, optional): Enriched item features data
        
        Returns:
        pd.DataFrame: Merged dataset
        """
        print("=" * 70)
        print("MERGING DATASETS")
        print("=" * 70)
        
        # Standardize column names
        df_auction = self._standardize_columns(df_auction, 'auction')
        df_items = self._standardize_columns(df_items, 'items')
        
        # Merge items with auction data
        print(f"\n1. Merging items with auction data...")
        print(f"   Items shape: {df_items.shape}")
        print(f"   Auction shape: {df_auction.shape}")
        
        # Check for overlapping columns
        items_cols = set(df_items.columns)
        auction_cols = set(df_auction.columns)
        overlap = (items_cols & auction_cols) - {'auction_id'}
        
        if overlap:
            print(f"   Overlapping columns: {overlap}")
            print(f"   Adding '_auction' suffix to {len(overlap)} columns")
            rename_dict = {col: f"{col}_auction" for col in overlap}
            df_auction = df_auction.rename(columns=rename_dict)
        
        # Merge on auction_id
        df_merged = df_items.merge(
            df_auction,
            on='auction_id',
            how='left',
            suffixes=('', '_auction')
        )
        print(f"   Merged shape: {df_merged.shape}")
        
        # Free memory
        del df_items, df_auction
        gc.collect()
        
        # Merge enriched features if provided
        if df_enriched is not None:
            print(f"\n2. Merging enriched item features...")
            print(f"   Enriched shape: {df_enriched.shape}")
            
            df_enriched = self._standardize_columns(df_enriched, 'enriched')
            
            # Check for overlapping columns
            merged_cols = set(df_merged.columns)
            enriched_cols = set(df_enriched.columns)
            overlap = (merged_cols & enriched_cols) - {'item_id'}
            
            if overlap:
                print(f"   Overlapping columns: {overlap}")
                print(f"   Adding '_enriched' suffix to {len(overlap)} columns")
                rename_dict = {col: f"{col}_enriched" for col in overlap}
                df_enriched = df_enriched.rename(columns=rename_dict)
            
            # Merge on item_id
            if 'item_id' not in df_merged.columns:
                print(f"   WARNING: 'item_id' not found in merged data")
                print(f"   Skipping enriched features merge")
            else:
                df_final = df_merged.merge(
                    df_enriched,
                    on='item_id',
                    how='left',
                    suffixes=('', '_enriched')
                )
                print(f"   Final shape: {df_final.shape}")
                
                # Check match quality
                if len(df_enriched.columns) > 1:
                    sample_col = [c for c in df_enriched.columns if c != 'item_id'][0]
                    matched_count = df_final[sample_col].notna().sum()
                    match_pct = (matched_count / len(df_final)) * 100
                    print(f"   Items with enriched features: {matched_count:,} ({match_pct:.1f}%)")
                
                # Free memory
                del df_enriched
                gc.collect()
        else:
            print(f"\n2. No enriched features provided, skipping")
            df_final = df_merged
        
        # Summary
        print("\n" + "=" * 70)
        print("MERGE COMPLETE")
        print("=" * 70)
        print(f"Final dataset: {df_final.shape[0]:,} rows × {df_final.shape[1]:,} columns")
        
        return df_final
    
    def _standardize_columns(self, df, dataset_type):
        """
        Standardize column names for merging.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        dataset_type (str): Type of dataset ('auction', 'items', 'enriched')
        
        Returns:
        pd.DataFrame: Dataframe with standardized column names
        """
        df = df.copy()
        
        # Standardize auction_id
        auction_id_cols = ['auction_id', 'amAuctionId', 'auctionId']
        for col in auction_id_cols:
            if col in df.columns and col != 'auction_id':
                df = df.rename(columns={col: 'auction_id'})
                break
        
        # Standardize item_id
        item_id_cols = ['item_id', 'amLotId', 'lotId', 'id']
        for col in item_id_cols:
            if col in df.columns and col != 'item_id':
                df = df.rename(columns={col: 'item_id'})
                break
        
        # Convert IDs to string
        if 'auction_id' in df.columns:
            df['auction_id'] = df['auction_id'].astype(str)
        if 'item_id' in df.columns:
            df['item_id'] = df['item_id'].astype(str)
        
        return df

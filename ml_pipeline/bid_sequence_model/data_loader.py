"""
Data Loader for Bid Sequence Model

This module handles loading and preprocessing bid history data for sequence modeling.
Handles the reversed bid numbering (winning bid = 1) and merges with item metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class BidSequenceDataLoader:
    """
    Load and preprocess bid history data for sequence prediction.
    
    The raw data has reversed bid numbers (winning bid = 1, first bid = max).
    This loader corrects the ordering and creates sequences suitable for LSTM/GRU models.
    """
    
    def __init__(self, bid_history_path: Optional[str] = None, 
                 item_metadata_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        bid_history_path : str, optional
            Path to bid history parquet/csv file
        item_metadata_path : str, optional
            Path to item metadata parquet/csv file
        """
        self.bid_history_path = bid_history_path
        self.item_metadata_path = item_metadata_path
        self.bid_data = None
        self.item_metadata = None
        
    def load_bid_history(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load bid history data from file.
        
        Parameters:
        -----------
        path : str, optional
            Path to bid history file. Uses self.bid_history_path if not provided.
            
        Returns:
        --------
        pd.DataFrame
            Loaded bid history data
        """
        path = path or self.bid_history_path
        if path is None:
            raise ValueError("No bid history path provided")
            
        path = Path(path)
        print(f"Loading bid history from: {path}")
        
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        print(f"Loaded {len(df):,} bid records")
        self.bid_data = df
        return df
    
    def load_item_metadata(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load item metadata (auction start/end times, etc.).
        
        Parameters:
        -----------
        path : str, optional
            Path to item metadata file
            
        Returns:
        --------
        pd.DataFrame
            Loaded item metadata
        """
        path = path or self.item_metadata_path
        if path is None:
            print("No item metadata path provided, skipping")
            return None
            
        path = Path(path)
        print(f"Loading item metadata from: {path}")
        
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        print(f"Loaded metadata for {len(df):,} items")
        self.item_metadata = df
        return df
    
    def reverse_bid_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse bid numbering so first bid = 1, last/winning bid = max.
        
        The raw data has winning bid = 1, first bid = total_bids (reversed).
        This function corrects the ordering.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Bid history dataframe with columns: auction_id, item_id, bid_number,
            time_of_bid, amount, isproxy
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with corrected bid numbering
        """
        print("\nReversing bid numbering (first bid = 1)...")
        
        # Sort by item and amount to ensure correct temporal ordering
        # Lower amounts typically come first (except in weird edge cases)
        df = df.sort_values(['auction_id', 'item_id', 'amount', 'time_of_bid']).copy()
        
        # Calculate total bids per item
        df['total_bids'] = df.groupby(['auction_id', 'item_id'])['item_id'].transform('count')
        
        # Create corrected bid number (1 = first bid)
        df['bid_number_corrected'] = df.groupby(['auction_id', 'item_id']).cumcount() + 1
        
        # Store original bid number if not already stored
        if 'bid_number_original' not in df.columns and 'bid_number' in df.columns:
            df['bid_number_original'] = df['bid_number']
            
        # Replace bid_number with corrected version
        df['bid_number'] = df['bid_number_corrected']
        df = df.drop(columns=['bid_number_corrected'])
        
        print(f"Reversed {len(df):,} bids across {df['item_id'].nunique():,} items")
        return df
    
    def merge_item_metadata(self, bid_df: pd.DataFrame, 
                           metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge item metadata with bid history.
        
        Parameters:
        -----------
        bid_df : pd.DataFrame
            Bid history data
        metadata_df : pd.DataFrame
            Item metadata with auction/item timing info
            
        Returns:
        --------
        pd.DataFrame
            Merged dataframe
        """
        print("\nMerging item metadata...")
        
        # Determine merge keys
        merge_keys = []
        if 'auction_id' in bid_df.columns and 'auction_id' in metadata_df.columns:
            merge_keys.append('auction_id')
        if 'item_id' in bid_df.columns and 'item_id' in metadata_df.columns:
            merge_keys.append('item_id')
            
        if not merge_keys:
            print("Warning: No common keys found for merging")
            return bid_df
            
        print(f"Merging on: {merge_keys}")
        merged = bid_df.merge(metadata_df, on=merge_keys, how='left')
        print(f"Merged result: {len(merged):,} rows")
        
        return merged
    
    def create_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features useful for sequence modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Bid history data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional features
        """
        print("\nCreating sequence features...")
        
        # Ensure time_of_bid is datetime
        if 'time_of_bid' in df.columns:
            df['time_of_bid'] = pd.to_datetime(df['time_of_bid'])
            
            # Time since first bid (in hours)
            df['first_bid_time'] = df.groupby(['auction_id', 'item_id'])['time_of_bid'].transform('min')
            df['hours_since_first'] = (df['time_of_bid'] - df['first_bid_time']).dt.total_seconds() / 3600
            
        # Bid increment from previous bid
        df = df.sort_values(['auction_id', 'item_id', 'bid_number'])
        df['bid_increment'] = df.groupby(['auction_id', 'item_id'])['amount'].diff().fillna(0)
        
        # Position in sequence (as percentage)
        if 'total_bids' in df.columns:
            df['bid_position_pct'] = df['bid_number'] / df['total_bids']
            
        # Proxy bid count up to this point
        if 'isproxy' in df.columns:
            df['proxy_count_so_far'] = df.groupby(['auction_id', 'item_id'])['isproxy'].cumsum()
            df['proxy_ratio_so_far'] = df['proxy_count_so_far'] / df['bid_number']
            
        print(f"Created sequence features")
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         max_sequence_length: int = 50,
                         features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare bid sequences for LSTM/GRU model training.
        
        Each sequence represents bids for one item, padded to max_sequence_length.
        Target is the final (winning) bid amount.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Bid history data with features
        max_sequence_length : int
            Maximum sequence length (shorter sequences are padded)
        features : List[str], optional
            List of feature columns to use. If None, uses default features.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            X: sequences (n_items, max_seq_len, n_features)
            y: final prices (n_items,)
            item_ids: item identifiers (n_items,)
        """
        print(f"\nPreparing sequences (max_length={max_sequence_length})...")
        
        # Default features
        if features is None:
            features = ['amount', 'bid_increment', 'hours_since_first', 
                       'isproxy', 'bid_position_pct', 'proxy_ratio_so_far']
            # Only use features that exist
            features = [f for f in features if f in df.columns]
            
        print(f"Using features: {features}")
        
        # Convert isproxy to int if present
        if 'isproxy' in features:
            df['isproxy'] = df['isproxy'].astype(int)
            
        # Group by item
        grouped = df.groupby(['auction_id', 'item_id'])
        
        sequences = []
        targets = []
        item_identifiers = []
        
        for (auction_id, item_id), group in grouped:
            # Sort by bid number to ensure correct order
            group = group.sort_values('bid_number')
            
            # Get feature values
            feature_values = group[features].values
            
            # Get final (winning) bid amount
            final_price = group['amount'].iloc[-1]
            
            # Pad or truncate sequence
            seq_len = min(len(feature_values), max_sequence_length)
            
            if len(feature_values) < max_sequence_length:
                # Pad with zeros
                padding = np.zeros((max_sequence_length - len(feature_values), len(features)))
                padded_seq = np.vstack([feature_values, padding])
            else:
                # Take last max_sequence_length bids
                padded_seq = feature_values[-max_sequence_length:]
                
            sequences.append(padded_seq)
            targets.append(final_price)
            item_identifiers.append(f"{auction_id}_{item_id}")
            
        X = np.array(sequences)
        y = np.array(targets)
        item_ids = np.array(item_identifiers)
        
        print(f"Created {len(X):,} sequences")
        print(f"  Sequence shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y, item_ids
    
    def create_partial_sequences(self, df: pd.DataFrame,
                                 partial_length: int = 10,
                                 max_sequence_length: int = 50,
                                 features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create partial sequences (first N bids only) for testing production scenario.
        
        This simulates the production use case where only the first X bids are known.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Bid history data
        partial_length : int
            Number of initial bids to include
        max_sequence_length : int
            Maximum sequence length for padding
        features : List[str], optional
            Features to use
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            X: partial sequences, y: final prices, item_ids: identifiers
        """
        print(f"\nCreating partial sequences (first {partial_length} bids)...")
        
        # Filter to only first N bids
        df_partial = df[df['bid_number'] <= partial_length].copy()
        
        # Recalculate position-based features for partial data
        df_partial['bid_position_pct'] = df_partial['bid_number'] / partial_length
        
        # Get final prices from full dataset
        final_prices = df.groupby(['auction_id', 'item_id'])['amount'].max()
        
        # Prepare sequences from partial data
        X, _, item_ids = self.prepare_sequences(
            df_partial, 
            max_sequence_length=min(partial_length, max_sequence_length),
            features=features
        )
        
        # Get corresponding final prices
        y = []
        for item_id in item_ids:
            auction_id, item_id_only = item_id.split('_')
            final_price = final_prices.get((int(auction_id), int(item_id_only)), 0)
            y.append(final_price)
        y = np.array(y)
        
        print(f"Created {len(X):,} partial sequences")
        return X, y, item_ids
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics about the bid data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Bid history data
            
        Returns:
        --------
        Dict
            Summary statistics
        """
        summary = {
            'total_bids': len(df),
            'unique_items': df['item_id'].nunique() if 'item_id' in df.columns else 0,
            'unique_auctions': df['auction_id'].nunique() if 'auction_id' in df.columns else 0,
            'avg_bids_per_item': len(df) / df['item_id'].nunique() if 'item_id' in df.columns else 0,
            'min_bid_amount': df['amount'].min() if 'amount' in df.columns else 0,
            'max_bid_amount': df['amount'].max() if 'amount' in df.columns else 0,
            'avg_bid_amount': df['amount'].mean() if 'amount' in df.columns else 0,
            'proxy_bid_pct': (df['isproxy'].sum() / len(df) * 100) if 'isproxy' in df.columns else 0,
        }
        return summary

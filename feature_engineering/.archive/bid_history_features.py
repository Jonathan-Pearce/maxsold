"""
Bid History Feature Engineering

This script processes raw bid history data and creates engineered features for modeling.

Features created:
- bid_number_corrected: First bid = 1, last bid = total_bids (reversed from original)
- total_bids_for_item: Total number of bids per item
- hours_since_first_bid: Hours elapsed since first bid on the item
- proxy_bid_count: Number of proxy bids per item
- manual_bid_count: Number of manual bids per item
- proxy_bid_ratio: Ratio of proxy bids to total bids
- bid_amount_min: Minimum bid amount for item
- bid_amount_max: Maximum bid amount for item (final bid)
- bid_amount_mean: Average bid amount for item
- bid_amount_std: Standard deviation of bid amounts
- bid_range: Difference between max and min bid
- first_bid_amount: Amount of first bid
- bid_increment_mean: Average increment between consecutive bids
- bid_increment_max: Maximum increment between consecutive bids
- bidding_duration_hours: Hours between first and last bid
- bids_per_hour: Average bids per hour during bidding period
- hour_of_day: Hour when bid was placed (0-23)
- day_of_week: Day of week when bid was placed (0=Monday, 6=Sunday)
- is_weekend: Whether bid was placed on weekend
- is_first_bid: Whether this is the first bid on the item
- is_last_bid: Whether this is the last bid on the item
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """Load bid history data from parquet file"""
    print(f"Loading data from: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    return df


def reverse_bid_numbers(df):
    """
    Reverse bid numbers so first bid = 1, last bid = total_bids
    Original has last bid = 1, first bid = total_bids
    """
    print("\n" + "="*80)
    print("REVERSING BID NUMBERS")
    print("="*80)
    
    # Sort by item and time to ensure correct ordering
    df_sorted = df.sort_values(['item_id', 'amount', 'time_of_bid']).copy()
    
    # Calculate total bids per item
    df_sorted['total_bids_for_item'] = df_sorted.groupby('item_id')['item_id'].transform('count')
    
    # Create old bid number (column)
    #df_sorted['bid_number_old'] = df_sorted['bid_number']

    # Create corrected bid number (1 = first bid by time)
    df_sorted['bid_number'] = df_sorted.groupby('item_id').cumcount() + 1
    
    return df_sorted


def create_time_features(df):
    """Create time-based features"""
    print("\n" + "="*80)
    print("CREATING TIME FEATURES")
    print("="*80)
    
    # Ensure time_of_bid is datetime
    df['time_of_bid'] = pd.to_datetime(df['time_of_bid'])
    
    # Hours since first bid for each item
    df['first_bid_time'] = df.groupby('item_id')['time_of_bid'].transform('min')
    df['last_bid_time'] = df.groupby('item_id')['time_of_bid'].transform('max')
    df['hours_since_first_bid'] = (df['time_of_bid'] - df['first_bid_time']).dt.total_seconds() / 3600
    
    # Temporal features
    #df['hour_of_day'] = df['time_of_bid'].dt.hour
    #df['day_of_week'] = df['time_of_bid'].dt.dayofweek
    #df['day_name'] = df['time_of_bid'].dt.day_name()
    #df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    #df['month'] = df['time_of_bid'].dt.month
    #df['year'] = df['time_of_bid'].dt.year
    
    #print(f"Created time features:")
    #print(f"  - hours_since_first_bid: range {df['hours_since_first_bid'].min():.2f} to {df['hours_since_first_bid'].max():.2f}")
    #print(f"  - bidding_duration_hours: mean {df['bidding_duration_hours'].mean():.2f} hours")
    #print(f"  - hour_of_day: range {df['hour_of_day'].min()} to {df['hour_of_day'].max()}")
    #print(f"  - day_of_week: 0=Monday, 6=Sunday")
    #print(f"  - is_weekend: {df['is_weekend'].sum():,} weekend bids ({100*df['is_weekend'].mean():.1f}%)")
    
    return df


def create_proxy_features(df):
    """Create features related to proxy bidding"""
    print("\n" + "="*80)
    print("CREATING PROXY BID FEATURES")
    print("="*80)
    
    # Count proxy bids per item
    df['proxy_bid_count'] = df.groupby('item_id')['isproxy'].transform('sum')
    
    # Count manual bids per item
    df['manual_bid_count'] = df['total_bids_for_item'] - df['proxy_bid_count']
    
    # Ratio of proxy bids
    df['proxy_bid_ratio'] = df['proxy_bid_count'] / df['total_bids_for_item']
    
    # Is this bid a proxy bid (binary)
    #df['is_proxy_bid'] = df['isproxy'].astype(int)
    
    print(f"Proxy bid features:")
    print(f"  - Total proxy bids: {df['isproxy'].sum():,} ({100*df['isproxy'].mean():.1f}%)")
    print(f"  - proxy_bid_count per item: mean {df['proxy_bid_count'].mean():.2f}")
    print(f"  - manual_bid_count per item: mean {df['manual_bid_count'].mean():.2f}")
    print(f"  - proxy_bid_ratio: mean {df['proxy_bid_ratio'].mean():.3f}")
    
    return df


def create_increment_features(df):
    """Create features related to bid increments"""
    print("\n" + "="*80)
    print("CREATING BID INCREMENT FEATURES")
    print("="*80)
    
    # Calculate increment from previous bid
    df = df.sort_values(['item_id', 'bid_number'])
    df['bid_increment'] = df.groupby('item_id')['amount'].diff()
    
    # Fill first bid increment with 0 or the bid amount
    df['bid_increment'] = df['bid_increment'].fillna(0)
    
    return df


def create_sequence_features(df):
    """Create features related to bid sequence and position"""
    print("\n" + "="*80)
    print("CREATING SEQUENCE FEATURES")
    print("="*80)
    
    # Position in sequence (as percentage)
    df['bid_position_pct'] = df['bid_number'] / df['total_bids_for_item']
    
    # Is first or last bid
    #df['is_first_bid'] = (df['bid_number'] == 1).astype(int)
    #df['is_last_bid'] = (df['bid_number'] == df['total_bids_for_item']).astype(int)
    
    # Bids remaining after this bid
    #df['bids_remaining'] = df['total_bids_for_item'] - df['bid_number']
    
    # Bidding velocity (bids per hour)
    #df['bids_per_hour'] = df['total_bids_for_item'] / (df['bidding_duration_hours'] + 0.001)  # Add small value to avoid division by 0
    
    #print(f"Sequence features:")
    #print(f"  - bid_position_pct: range {df['bid_position_pct'].min():.3f} to {df['bid_position_pct'].max():.3f}")
    #print(f"  - is_first_bid: {df['is_first_bid'].sum():,} first bids")
    #print(f"  - is_last_bid: {df['is_last_bid'].sum():,} last bids")
    #print(f"  - bids_per_hour: mean {df['bids_per_hour'].mean():.2f}")
    
    return df

def feature_engineering(df):
    """
    Main feature engineering pipeline
    
    Parameters:
    df (pd.DataFrame): Raw bid history dataframe
    
    Returns:
    pd.DataFrame: Dataframe with engineered features
    """
    print("\n" + "="*80)
    print("BID HISTORY FEATURE ENGINEERING PIPELINE")
    print("="*80)
    print(f"Input shape: {df.shape}")
    
    # Apply all feature engineering steps
    df = reverse_bid_numbers(df)
    df = create_time_features(df)
    df = create_proxy_features(df)
    df = create_increment_features(df)
    df = create_sequence_features(df)
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Output shape: {df.shape}")
    print(f"New features created: {df.shape[1] - len(['auction_id', 'item_id', 'bid_number', 'time_of_bid', 'amount', 'isproxy'])}")
    
    return df


def main():
    """Main execution function"""
    # Define paths
    input_path = Path("/workspaces/maxsold/data/raw_data/bid_history/bid_history_20251201.parquet")
    output_path = Path("/workspaces/maxsold/data/engineered_data/bid_history/bid_history_engineered_20251201.parquet")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_path)
    
    # Apply feature engineering
    df_engineered = feature_engineering(df)
    
    # Display sample of results
    print("\n" + "="*80)
    print("SAMPLE OF ENGINEERED DATA")
    print("="*80)
    print("\nFirst 10 rows with key features:")
    key_cols = ['item_id', 'bid_number', 'total_bids_for_item', 
                'hours_since_first_bid', 'proxy_bid_count', 'amount', 'bid_increment']
    print(df_engineered[key_cols].head(10).to_string(index=False))
    
    # Save engineered data
    print(f"\nSaving engineered data to: {output_path}")
    df_engineered.to_parquet(output_path, index=False)
    print(f"Saved successfully! Shape: {df_engineered.shape}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("FEATURE SUMMARY STATISTICS")
    print("="*80)
    
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
    print("\nNumeric feature statistics:")
    print(df_engineered[numeric_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
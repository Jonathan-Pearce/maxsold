"""
Bid Data Collapsing Script - Group consecutive bids within time threshold

This script collapses consecutive bids that occur within 1 minute of each other
into single "bid burst" rows, creating a compressed representation of bidding activity.

Output columns:
- item_id: Item identifier
- auction_id: Auction identifier
- burst_group: Sequential group number (1, 2, 3, ...)
- num_bids: Number of bids in this burst
- num_proxy_bids: Number of proxy bids in this burst
- num_manual_bids: Number of manual bids in this burst
- first_bid_amount: Amount of first bid in burst
- last_bid_amount: Amount of last bid in burst
- amount_range: Difference between last and first bid amounts
- first_bid_time: Time of first bid in burst
- last_bid_time: Time of last bid in burst
- burst_duration_minutes: Duration of burst in minutes
- first_bid_number: Bid number of first bid in burst
- last_bid_number: Bid number of last bid in burst
- hours_since_first_bid_start: Hours since first item bid (start of burst)
- hours_since_first_bid_end: Hours since first item bid (end of burst)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def collapse_bids_by_time(item_bids, time_threshold_minutes=1):
    """
    Group consecutive bids that occur within time_threshold_minutes of each other.
    Creates comprehensive statistics for each burst group.
    
    Parameters:
    -----------
    item_bids : pd.DataFrame
        Bids for a single item, sorted by time
    time_threshold_minutes : float
        Time threshold in minutes for grouping consecutive bids
    
    Returns:
    --------
    pd.DataFrame
        Collapsed bid groups with statistics
    """
    if len(item_bids) == 0:
        return pd.DataFrame()
    
    if len(item_bids) == 1:
        # Single bid - create one group
        bid = item_bids.iloc[0]
        return pd.DataFrame([{
            'item_id': bid['item_id'],
            'auction_id': bid['auction_id'],
            'burst_group': 1,
            'num_bids': 1,
            'num_proxy_bids': int(bid['isproxy']),
            'num_manual_bids': 1 - int(bid['isproxy']),
            'first_bid_amount': bid['amount'],
            'last_bid_amount': bid['amount'],
            'amount_range': 0.0,
            'first_bid_time': bid['time_of_bid'],
            'last_bid_time': bid['time_of_bid'],
            'burst_duration_minutes': 0.0,
            'first_bid_number': bid['bid_number'],
            'last_bid_number': bid['bid_number'],
            'hours_since_first_bid_start': bid['hours_since_first_bid'],
            'hours_since_first_bid_end': bid['hours_since_first_bid']
        }])
    
    # Sort by time
    item_bids = item_bids.sort_values('hours_since_first_bid').copy()
    
    # Convert threshold to hours
    threshold_hours = time_threshold_minutes / 60.0
    
    # Calculate time differences between consecutive bids
    item_bids['time_diff'] = item_bids['hours_since_first_bid'].diff()
    
    # Create burst groups (new group when time gap exceeds threshold)
    item_bids['burst_group'] = (item_bids['time_diff'] > threshold_hours).cumsum() + 1
    
    # Aggregate by burst group
    collapsed_groups = []
    
    for burst_id, burst_bids in item_bids.groupby('burst_group'):
        burst_bids = burst_bids.sort_values('hours_since_first_bid')
        
        first_bid = burst_bids.iloc[0]
        last_bid = burst_bids.iloc[-1]
        
        group_stats = {
            'item_id': first_bid['item_id'],
            'auction_id': first_bid['auction_id'],
            'burst_group': int(burst_id),
            'num_bids': len(burst_bids),
            'num_proxy_bids': int(burst_bids['isproxy'].sum()),
            'num_manual_bids': int((1 - burst_bids['isproxy']).sum()),
            'first_bid_amount': first_bid['amount'],
            'last_bid_amount': last_bid['amount'],
            'amount_range': last_bid['amount'] - first_bid['amount'],
            'first_bid_time': first_bid['time_of_bid'],
            'last_bid_time': last_bid['time_of_bid'],
            'burst_duration_minutes': (last_bid['hours_since_first_bid'] - first_bid['hours_since_first_bid']) * 60,
            'first_bid_number': int(first_bid['bid_number']),
            'last_bid_number': int(last_bid['bid_number']),
            'hours_since_first_bid_start': first_bid['hours_since_first_bid'],
            'hours_since_first_bid_end': last_bid['hours_since_first_bid']
        }
        
        collapsed_groups.append(group_stats)
    
    return pd.DataFrame(collapsed_groups)


def main():
    """Main execution function"""
    print("="*80)
    print("BID DATA COLLAPSING SCRIPT")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define paths
    input_path = Path("/workspaces/maxsold/data/engineered_data/bid_history/bid_history_engineered_20251201.parquet")
    output_path = Path("/workspaces/maxsold/data/engineered_data/bid_history/bid_history_collapsed_1min_20251201.parquet")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    TIME_THRESHOLD_MINUTES = 2
    MAX_ITEMS = 100  # Set to None to process all items, or specify number to limit
    
    print(f"\nConfiguration:")
    print(f"  Time threshold: {TIME_THRESHOLD_MINUTES} minute(s)")
    print(f"  Max items to process: {MAX_ITEMS if MAX_ITEMS else 'ALL'}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"Unique items: {df['item_id'].nunique():,}")
    print(f"Unique auctions: {df['auction_id'].nunique():,}")
    
    # Ensure time_of_bid is datetime
    if df['time_of_bid'].dtype != 'datetime64[ns]':
        print("Converting time_of_bid to datetime...")
        df['time_of_bid'] = pd.to_datetime(df['time_of_bid'])
    
    # Get items to process
    all_items = df['item_id'].unique()
    
    if MAX_ITEMS and MAX_ITEMS < len(all_items):
        items = all_items[:MAX_ITEMS]
        df = df[df['item_id'].isin(items)].copy()  # Filter to selected items only
        print(f"\n*** LIMITING TO FIRST {MAX_ITEMS} ITEMS ***")
        print(f"  Total bids for selected items: {len(df):,}")
    else:
        items = all_items
    
    # Process each item
    print(f"\nCollapsing bids by item...")
    print(f"Grouping bids within {TIME_THRESHOLD_MINUTES} minute(s) of each other")
    print(f"Processing {len(items):,} items...")
    
    all_collapsed = []
    
    # Progress tracking
    total_items = len(items)
    progress_interval = max(1, total_items // 20)  # Report every 5%
    
    for idx, item_id in enumerate(items, 1):
        item_bids = df[df['item_id'] == item_id].copy()
        collapsed = collapse_bids_by_time(item_bids, TIME_THRESHOLD_MINUTES)
        
        if not collapsed.empty:
            all_collapsed.append(collapsed)
        
        # Progress reporting
        if idx % progress_interval == 0 or idx == total_items:
            pct = 100 * idx / total_items
            print(f"  Progress: {idx:,}/{total_items:,} items ({pct:.1f}%)")
    
    # Combine all collapsed data
    print(f"\nCombining collapsed data...")
    df_collapsed = pd.concat(all_collapsed, ignore_index=True)
    
    # Summary statistics
    print("\n" + "="*80)
    print("COLLAPSING SUMMARY")
    print("="*80)
    
    if MAX_ITEMS and MAX_ITEMS < len(all_items):
        print(f"\n*** PROCESSED {len(items):,} of {len(all_items):,} TOTAL ITEMS ***")
    
    print(f"\nOriginal data (for processed items):")
    print(f"  Total bids: {len(df):,}")
    print(f"  Unique items: {df['item_id'].nunique():,}")
    print(f"  Average bids per item: {len(df) / df['item_id'].nunique():.2f}")
    
    print(f"\nCollapsed data:")
    print(f"  Total burst groups: {len(df_collapsed):,}")
    print(f"  Unique items: {df_collapsed['item_id'].nunique():,}")
    print(f"  Average bursts per item: {len(df_collapsed) / df_collapsed['item_id'].nunique():.2f}")
    
    compression_ratio = len(df) / len(df_collapsed)
    reduction_pct = 100 * (1 - len(df_collapsed) / len(df))
    
    print(f"\nCompression:")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Data reduction: {reduction_pct:.1f}%")
    print(f"  Reduced from {len(df):,} to {len(df_collapsed):,} rows")
    
    # Burst statistics
    print(f"\nBurst statistics:")
    print(f"  Bids per burst:")
    print(f"    Mean: {df_collapsed['num_bids'].mean():.2f}")
    print(f"    Median: {df_collapsed['num_bids'].median():.0f}")
    print(f"    Max: {df_collapsed['num_bids'].max()}")
    
    print(f"\n  Burst duration (minutes):")
    non_zero_duration = df_collapsed[df_collapsed['burst_duration_minutes'] > 0]
    if len(non_zero_duration) > 0:
        print(f"    Mean: {non_zero_duration['burst_duration_minutes'].mean():.2f}")
        print(f"    Median: {non_zero_duration['burst_duration_minutes'].median():.2f}")
        print(f"    Max: {non_zero_duration['burst_duration_minutes'].max():.2f}")
    else:
        print(f"    No multi-bid bursts found")
    
    print(f"\n  Proxy bid ratio:")
    df_collapsed['proxy_ratio'] = df_collapsed['num_proxy_bids'] / df_collapsed['num_bids']
    print(f"    Mean: {df_collapsed['proxy_ratio'].mean():.3f}")
    print(f"    Median: {df_collapsed['proxy_ratio'].median():.3f}")
    
    print(f"\n  Amount range (last - first):")
    print(f"    Mean: ${df_collapsed['amount_range'].mean():.2f}")
    print(f"    Median: ${df_collapsed['amount_range'].median():.2f}")
    print(f"    Max: ${df_collapsed['amount_range'].max():.2f}")
    
    # Distribution of burst sizes
    print(f"\n  Burst size distribution:")
    print(f"    1 bid: {(df_collapsed['num_bids'] == 1).sum():,} bursts ({100*(df_collapsed['num_bids'] == 1).sum()/len(df_collapsed):.1f}%)")
    print(f"    2 bids: {(df_collapsed['num_bids'] == 2).sum():,} bursts ({100*(df_collapsed['num_bids'] == 2).sum()/len(df_collapsed):.1f}%)")
    print(f"    3-5 bids: {((df_collapsed['num_bids'] >= 3) & (df_collapsed['num_bids'] <= 5)).sum():,} bursts")
    print(f"    6-10 bids: {((df_collapsed['num_bids'] >= 6) & (df_collapsed['num_bids'] <= 10)).sum():,} bursts")
    print(f"    11+ bids: {(df_collapsed['num_bids'] >= 11).sum():,} bursts")
    
    # Sample output
    print("\n" + "="*80)
    print("SAMPLE OUTPUT")
    print("="*80)
    print("\nFirst 10 rows:")
    sample_cols = ['item_id', 'burst_group', 'num_bids', 'num_proxy_bids', 
                   'first_bid_amount', 'last_bid_amount', 'amount_range',
                   'burst_duration_minutes', 'first_bid_number', 'last_bid_number']
    print(df_collapsed[sample_cols].head(10).to_string(index=False))
    
    # Save collapsed data
    print(f"\n" + "="*80)
    print("SAVING COLLAPSED DATA")
    print("="*80)
    
    # Update output filename if processing limited items
    if MAX_ITEMS and MAX_ITEMS < len(all_items):
        output_path = output_path.parent / f'bid_history_collapsed_{TIME_THRESHOLD_MINUTES}min_{MAX_ITEMS}items_20251201.parquet'
    
    print(f"Saving to: {output_path}")
    df_collapsed.to_parquet(output_path, index=False)
    print(f"Saved successfully!")
    print(f"Output shape: {df_collapsed.shape}")
    
    # Save metadata/summary
    metadata_path = output_path.parent / output_path.name.replace('.parquet', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("BID DATA COLLAPSING METADATA\n")
        f.write("="*80 + "\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Time threshold: {TIME_THRESHOLD_MINUTES} minute(s)\n")
        f.write(f"Items processed: {len(items):,} of {len(all_items):,}\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output file: {output_path}\n\n")
        f.write(f"Original rows (for processed items): {len(df):,}\n")
        f.write(f"Collapsed rows: {len(df_collapsed):,}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Data reduction: {reduction_pct:.1f}%\n\n")
        f.write("COLUMNS:\n")
        for col in df_collapsed.columns:
            f.write(f"  - {col}\n")
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    print("\n" + "="*80)
    print("SCRIPT COMPLETE!")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
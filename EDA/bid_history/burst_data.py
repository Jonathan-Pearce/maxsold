"""
Bid Progression Analysis

This script analyzes the collapsed bid history data and creates visualizations
of bid progressions over time for a sample of items.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_collapsed_bid_data(file_path):
    """Load the collapsed bid history data"""
    print(f"Loading data from: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} collapsed bid bursts")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData shape: {df.shape}")
    print(f"Unique items: {df['item_id'].nunique()}")
    return df

def prepare_bid_progressions(df):
    """Prepare data for bid progression analysis"""
    # Convert time columns to datetime if not already
    if df['first_bid_time'].dtype != 'datetime64[ns]':
        df['first_bid_time'] = pd.to_datetime(df['first_bid_time'])
    if df['last_bid_time'].dtype != 'datetime64[ns]':
        df['last_bid_time'] = pd.to_datetime(df['last_bid_time'])
    
    # For each item, calculate time since the item's first bid (in hours)
    df['item_first_bid'] = df.groupby('item_id')['first_bid_time'].transform('min')
    df['hours_since_item_start'] = (df['first_bid_time'] - df['item_first_bid']).dt.total_seconds() / 3600
    
    return df

def select_sample_items(df, n_samples=10, strategy='diverse'):
    """
    Select a sample of items for visualization
    
    strategy options:
    - 'random': Random sample
    - 'diverse': Items with diverse bid counts
    - 'high_activity': Items with most bids
    - 'price_range': Items across different price ranges
    """
    # Calculate total bids per item
    item_stats = df.groupby('item_id').agg({
        'num_bids': 'sum',
        'last_bid_amount': 'max',
        'burst_group': 'count'
    }).rename(columns={'burst_group': 'num_bursts'})
    
    if strategy == 'random':
        selected = item_stats.sample(n=min(n_samples, len(item_stats)), random_state=42)
    elif strategy == 'diverse':
        # Select items across different total bid counts
        item_stats_sorted = item_stats.sort_values('num_bids')
        indices = np.linspace(0, len(item_stats_sorted)-1, n_samples, dtype=int)
        selected = item_stats_sorted.iloc[indices]
    elif strategy == 'high_activity':
        selected = item_stats.nlargest(n_samples, 'num_bids')
    elif strategy == 'price_range':
        # Select items across different price ranges
        item_stats_sorted = item_stats.sort_values('last_bid_amount')
        indices = np.linspace(0, len(item_stats_sorted)-1, n_samples, dtype=int)
        selected = item_stats_sorted.iloc[indices]
    else:
        selected = item_stats.sample(n=min(n_samples, len(item_stats)), random_state=42)
    
    print(f"\nSelected {len(selected)} items using '{strategy}' strategy:")
    print(selected)
    
    return selected.index.tolist()

def plot_bid_progressions(df, sample_items, output_path=None):
    """
    Plot bid progressions for selected items
    X-axis: first_bid_time (hours since item's first bid)
    Y-axis: last_bid_amount (final amount in each burst)
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Filter to sample items
    df_sample = df[df['item_id'].isin(sample_items)].copy()
    
    # Plot 1: Individual item progressions
    ax1 = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(sample_items)))
    
    for idx, item_id in enumerate(sample_items):
        item_data = df_sample[df_sample['item_id'] == item_id].sort_values('hours_since_item_start')
        
        # Plot line connecting bursts
        ax1.plot(item_data['hours_since_item_start'], 
                item_data['last_bid_amount'],
                marker='o', 
                linewidth=2,
                markersize=6,
                alpha=0.7,
                color=colors[idx],
                label=f'Item {item_id}')
        
        # Add markers for burst size (num_bids)
        for _, row in item_data.iterrows():
            size = row['num_bids'] * 5  # Scale marker size by number of bids in burst
            ax1.scatter(row['hours_since_item_start'], 
                       row['last_bid_amount'],
                       s=size,
                       alpha=0.3,
                       color=colors[idx])
    
    ax1.set_xlabel('Hours Since First Bid', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bid Amount ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Bid Progression Over Time by Item\n(Marker size indicates number of bids in burst)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Aggregate view - all bursts
    ax2 = axes[1]
    
    # Create scatter plot with color based on item
    for idx, item_id in enumerate(sample_items):
        item_data = df_sample[df_sample['item_id'] == item_id].sort_values('hours_since_item_start')
        
        # Scatter plot with size based on num_bids
        sizes = item_data['num_bids'] * 20
        ax2.scatter(item_data['hours_since_item_start'], 
                   item_data['last_bid_amount'],
                   s=sizes,
                   alpha=0.6,
                   color=colors[idx],
                   label=f'Item {item_id}',
                   edgecolors='black',
                   linewidth=0.5)
    
    ax2.set_xlabel('Hours Since First Bid', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bid Amount ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Bid Amount Distribution Over Time\n(Bubble size = number of bids in burst)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()
    
    return fig

def plot_additional_analysis(df, sample_items, output_path=None):
    """Create additional analysis plots"""
    df_sample = df[df['item_id'].isin(sample_items)].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Burst duration over time
    ax1 = axes[0, 0]
    for item_id in sample_items:
        item_data = df_sample[df_sample['item_id'] == item_id].sort_values('hours_since_item_start')
        ax1.scatter(item_data['hours_since_item_start'], 
                   item_data['burst_duration_minutes'],
                   alpha=0.6,
                   s=50,
                   label=f'Item {item_id}')
    ax1.set_xlabel('Hours Since First Bid', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Burst Duration (minutes)', fontsize=11, fontweight='bold')
    ax1.set_title('Bidding Burst Duration Over Time', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of bids per burst over time
    ax2 = axes[0, 1]
    for item_id in sample_items:
        item_data = df_sample[df_sample['item_id'] == item_id].sort_values('hours_since_item_start')
        ax2.scatter(item_data['hours_since_item_start'], 
                   item_data['num_bids'],
                   alpha=0.6,
                   s=50,
                   label=f'Item {item_id}')
    ax2.set_xlabel('Hours Since First Bid', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Bids in Burst', fontsize=11, fontweight='bold')
    ax2.set_title('Bid Intensity Over Time', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Proxy vs Manual bids over time
    ax3 = axes[1, 0]
    df_sample['proxy_pct'] = df_sample['num_proxy_bids'] / df_sample['num_bids'] * 100
    for item_id in sample_items:
        item_data = df_sample[df_sample['item_id'] == item_id].sort_values('hours_since_item_start')
        ax3.scatter(item_data['hours_since_item_start'], 
                   item_data['proxy_pct'],
                   alpha=0.6,
                   s=50,
                   label=f'Item {item_id}')
    ax3.set_xlabel('Hours Since First Bid', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Proxy Bid %', fontsize=11, fontweight='bold')
    ax3.set_title('Proxy Bid Percentage Over Time', fontsize=12, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)
    
    # Plot 4: Bid amount range per burst
    ax4 = axes[1, 1]
    for item_id in sample_items:
        item_data = df_sample[df_sample['item_id'] == item_id].sort_values('hours_since_item_start')
        ax4.scatter(item_data['hours_since_item_start'], 
                   item_data['amount_range'],
                   alpha=0.6,
                   s=50,
                   label=f'Item {item_id}')
    ax4.set_xlabel('Hours Since First Bid', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Bid Amount Range ($)', fontsize=11, fontweight='bold')
    ax4.set_title('Bid Amount Range per Burst Over Time', fontsize=12, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Additional analysis plot saved to: {output_path}")
    
    plt.show()
    
    return fig

def print_summary_stats(df, sample_items):
    """Print summary statistics for the sample"""
    df_sample = df[df['item_id'].isin(sample_items)]
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR SAMPLE ITEMS")
    print("="*80)
    
    for item_id in sample_items:
        item_data = df_sample[df_sample['item_id'] == item_id]
        print(f"\nItem {item_id}:")
        print(f"  Total bursts: {len(item_data)}")
        print(f"  Total bids: {item_data['num_bids'].sum()}")
        print(f"  First bid amount: ${item_data['first_bid_amount'].iloc[0]:.2f}")
        print(f"  Final bid amount: ${item_data['last_bid_amount'].max():.2f}")
        print(f"  Total increase: ${item_data['last_bid_amount'].max() - item_data['first_bid_amount'].iloc[0]:.2f}")
        print(f"  Bidding duration: {item_data['hours_since_item_start'].max():.2f} hours")
        print(f"  Avg bids per burst: {item_data['num_bids'].mean():.1f}")
        print(f"  Proxy bid ratio: {item_data['num_proxy_bids'].sum() / item_data['num_bids'].sum():.1%}")

def main():
    """Main execution function"""
    print("="*80)
    print("BID PROGRESSION ANALYSIS")
    print("="*80)
    
    # Define paths
    input_path = Path("/workspaces/maxsold/data/engineered_data/bid_history/bid_history_collapsed_1min_100items_20251201.parquet")
    output_dir = Path("/workspaces/maxsold/EDA/bid_history")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_collapsed_bid_data(input_path)
    
    # Prepare data
    df = prepare_bid_progressions(df)
    
    # Select sample items
    n_samples = 10
    strategy = 'diverse'  # Options: 'random', 'diverse', 'high_activity', 'price_range'
    sample_items = select_sample_items(df, n_samples=n_samples, strategy=strategy)
    
    # Print summary statistics
    print_summary_stats(df, sample_items)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_plot1 = output_dir / "bid_progressions.png"
    plot_bid_progressions(df, sample_items, output_path=output_plot1)
    
    output_plot2 = output_dir / "bid_analysis_additional.png"
    plot_additional_analysis(df, sample_items, output_path=output_plot2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
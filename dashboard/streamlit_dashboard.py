import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="MaxSold Auction Analytics",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("/workspaces/maxsold/data/auction/items_all_2025-11-21.csv")
    
    # Data cleaning and type conversion
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    df['current_bid'] = pd.to_numeric(df['current_bid'], errors='coerce')
    df['starting_bid'] = pd.to_numeric(df['starting_bid'], errors='coerce')
    df['minimum_bid'] = pd.to_numeric(df['minimum_bid'], errors='coerce')
    df['bid_count'] = pd.to_numeric(df['bid_count'], errors='coerce')
    df['viewed'] = pd.to_numeric(df['viewed'], errors='coerce')
    
    # Calculate metrics
    df['bid_increment'] = df['current_bid'] - df['starting_bid']
    df['bid_rate'] = df['bid_count'] / (df['viewed'] + 1)  # avoid division by zero
    
    # Calculate hours_active - handle NaT values
    time_diff = df['end_time'] - df['start_time']
    # Convert to timedelta explicitly
    df['hours_active'] = pd.to_timedelta(time_diff, errors='coerce').dt.total_seconds() / 3600
    
    return df

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🔨 MaxSold Auction Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure items_all_2025-11-21.csv exists.")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.exception(e)  # Show full traceback for debugging
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Auction filter
    auctions = ["All"] + sorted([str(x) for x in df['auction_id'].dropna().unique().tolist()])
    selected_auction = st.sidebar.selectbox("Select Auction", auctions)
    
    # Bid count filter
    max_bid_count = int(df['bid_count'].max()) if not pd.isna(df['bid_count'].max()) else 100
    min_bids = st.sidebar.slider(
        "Minimum Bid Count",
        min_value=0,
        max_value=max_bid_count,
        value=0
    )
    
    # View count filter
    max_views = int(df['viewed'].max()) if not pd.isna(df['viewed'].max()) else 1000
    min_views = st.sidebar.slider(
        "Minimum Views",
        min_value=0,
        max_value=max_views,
        value=0
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_auction != "All":
        filtered_df = filtered_df[filtered_df['auction_id'].astype(str) == selected_auction]
    filtered_df = filtered_df[filtered_df['bid_count'] >= min_bids]
    filtered_df = filtered_df[filtered_df['viewed'] >= min_views]
    
    # Key Metrics Row
    st.header("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Items",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if selected_auction != "All" else None
        )
    
    with col2:
        total_bids = filtered_df['bid_count'].sum()
        st.metric(
            label="Total Bids",
            value=f"{int(total_bids):,}"
        )
    
    with col3:
        avg_bid_price = filtered_df['current_bid'].mean()
        st.metric(
            label="Avg Current Bid",
            value=f"${avg_bid_price:.2f}" if not pd.isna(avg_bid_price) else "N/A"
        )
    
    with col4:
        total_value = filtered_df['current_bid'].sum()
        st.metric(
            label="Total Bid Value",
            value=f"${total_value:,.2f}" if not pd.isna(total_value) else "N/A"
        )
    
    with col5:
        active_items = filtered_df['bidding_extended'].sum()
        st.metric(
            label="Extended Bidding",
            value=f"{int(active_items):,}"
        )
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Bid Distribution")
        fig_hist = px.histogram(
            filtered_df,
            x='bid_count',
            nbins=30,
            title="Distribution of Bid Counts per Item",
            labels={'bid_count': 'Number of Bids', 'count': 'Number of Items'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("💰 Current Bid Distribution")
        fig_box = px.box(
            filtered_df[filtered_df['current_bid'] > 0],
            y='current_bid',
            title="Current Bid Price Distribution",
            labels={'current_bid': 'Current Bid ($)'},
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👁️ Views vs Bids")
        scatter_df = filtered_df[filtered_df['viewed'] > 0].copy()
        if len(scatter_df) > 0:
            fig_scatter = px.scatter(
                scatter_df,
                x='viewed',
                y='bid_count',
                size='current_bid',
                color='bidding_extended',
                hover_data=['title', 'auction_id', 'id'],
                title="Item Views vs Bid Count (sized by current bid)",
                labels={'viewed': 'Views', 'bid_count': 'Bids', 'bidding_extended': 'Extended'},
                color_discrete_map={True: '#ff7f0e', False: '#1f77b4'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No items with views in current filter")
    
    with col2:
        st.subheader("🏆 Top 10 Items by Bid Count")
        top_items = filtered_df.nlargest(10, 'bid_count')[['title', 'bid_count', 'current_bid', 'viewed']]
        if len(top_items) > 0:
            top_items = top_items.copy()
            top_items['title'] = top_items['title'].str[:50]  # truncate long titles
            
            fig_bar = px.bar(
                top_items,
                x='bid_count',
                y='title',
                orientation='h',
                title="Top 10 Most Bid Items",
                labels={'bid_count': 'Number of Bids', 'title': 'Item'},
                color='current_bid',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No items in current filter")
    
    # Charts Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Auction Timeline")
        timeline_df = filtered_df.groupby('auction_id').agg({
            'id': 'count',
            'bid_count': 'sum',
            'current_bid': 'sum'
        }).reset_index()
        timeline_df.columns = ['auction_id', 'items', 'total_bids', 'total_value']
        
        if len(timeline_df) > 0:
            fig_timeline = px.bar(
                timeline_df,
                x='auction_id',
                y=['items', 'total_bids'],
                title="Items and Bids per Auction",
                labels={'value': 'Count', 'auction_id': 'Auction ID'},
                barmode='group'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No data to display")
    
    with col2:
        st.subheader("💵 Bid Growth Analysis")
        bid_growth = filtered_df[filtered_df['bid_increment'] > 0].copy()
        
        if len(bid_growth) > 0:
            fig_growth = px.scatter(
                bid_growth,
                x='starting_bid',
                y='current_bid',
                size='bid_count',
                color='bid_increment',
                hover_data=['title', 'auction_id'],
                title="Starting Bid vs Current Bid (sized by bid count)",
                labels={'starting_bid': 'Starting Bid ($)', 'current_bid': 'Current Bid ($)'},
                color_continuous_scale='RdYlGn'
            )
            # Add diagonal line (y=x) to show items at starting price
            max_val = max(bid_growth['current_bid'].max(), bid_growth['starting_bid'].max())
            fig_growth.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='No Growth Line',
                showlegend=True
            ))
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.info("No items with bid growth in current filter")
    
    # Data Table
    st.markdown("---")
    st.header("📋 Item Details")
    
    # Column selection
    display_cols = st.multiselect(
        "Select columns to display",
        options=filtered_df.columns.tolist(),
        default=['auction_id', 'id', 'title', 'current_bid', 'bid_count', 'viewed', 'bidding_extended']
    )
    
    if display_cols:
        # Sort options
        sort_col = st.selectbox("Sort by", options=display_cols)
        sort_order = st.radio("Sort order", options=["Descending", "Ascending"], horizontal=True)
        
        display_df = filtered_df[display_cols].sort_values(
            by=sort_col,
            ascending=(sort_order == "Ascending")
        )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"maxsold_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Data source: items_all_2025-11-21.csv | Total records: {len(df):,}")

if __name__ == "__main__":
    main()
import pandas as pd

#Run from 01_extract_auction_search.py
def format_auction_search_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert numeric columns
    numeric_cols = ['distanceMeters', 'totalBids', 'numberLots', 'lat', 'lng']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean
    if 'hasShipping' in df.columns:
        df['hasShipping'] = df['hasShipping'].astype('boolean')
    
    # Convert datetime
    for col in ['openTime', 'closeTime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


#Run from 02_extract_auction_details.py
def format_auction_details_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert numeric columns
    numeric_cols = ['extended_bidding_interval', 'extended_bidding_threshold', 'catalog_lots']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean
    if 'extended_bidding' in df.columns:
        df['extended_bidding'] = df['extended_bidding'].astype('boolean')
    
    # Convert datetime columns
    datetime_cols = ['starts', 'ends', 'last_item_closes', 'pickup_time']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


#Run from 03_extract_items_details.py
def format_item_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format raw item data DataFrame to proper types"""
# Convert numeric columns
    numeric_cols = ['viewed', 'minimum_bid', 'starting_bid', 'current_bid', 
                    'proxy_bid', 'bid_count', 'buyer_premium']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert boolean columns
    bool_cols = ['taxable', 'bidding_extended']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('boolean')
    
    # Convert datetime columns
    datetime_cols = ['start_time', 'end_time']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


#Run from 04_extract_bid_history.py
def format_bid_history_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert numeric columns
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    if 'bid_number' in df.columns:
        df['bid_number'] = pd.to_numeric(df['bid_number'], errors='coerce').astype('Int64')
    
    # Convert boolean
    if 'isproxy' in df.columns:
        df['isproxy'] = df['isproxy'].astype('boolean')
    
    # Convert datetime
    if 'time_of_bid' in df.columns:
        df['time_of_bid'] = pd.to_datetime(df['time_of_bid'], errors='coerce')
    
    return df


#Run from 05_extract_item_enriched_data.py
def format_item_enriched_data(df: pd.DataFrame) -> pd.DataFrame:
        # Convert numeric columns
    numeric_cols = ['brands_count', 'categories_count', 'items_count', 'attributes_count', 'numItems']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Convert boolean columns more carefully
    #boolean_cols = ['working', 'singleKeyItem']
    boolean_cols = ['singleKeyItem']
    for col in boolean_cols:
        if col in df.columns:
            # Convert to boolean, handling None/NaN and various string representations
            df[col] = df[col].apply(lambda x: 
                None if pd.isna(x) else 
                bool(x) if isinstance(x, (bool, int)) else
                str(x).lower() in ('true', '1', 'yes')
            )
            # Convert to nullable boolean type
            df[col] = df[col].astype('boolean')

    return df
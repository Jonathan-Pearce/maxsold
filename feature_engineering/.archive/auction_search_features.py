import pandas as pd
import numpy as np

def feature_engineering(input_path):
    """
    Perform feature engineering on the auction dataset.
    
    Parameters:
    input_path (str): Path to the parquet file
    
    Returns:
    pd.DataFrame: Dataframe with engineered features
    """
    # Load the dataset
    df = pd.read_parquet(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Create a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # 0. Create average bids per lot feature
    if 'totalBids' in df_engineered.columns and 'numberLots' in df_engineered.columns:
        print(f"\nCreating average_bids_per_lot feature...")
        # Handle division by zero by replacing 0 numberLots with NaN
        df_engineered['average_bids_per_lot'] = df_engineered['totalBids'] / df_engineered['numberLots'].replace(0, np.nan)
        print(f"average_bids_per_lot statistics:")
        print(df_engineered['average_bids_per_lot'].describe())
    else:
        print(f"\nWarning: 'totalBids' or 'numberLots' column not found. Skipping average_bids_per_lot feature.")
    
    # 1. One-hot encoding for saleCategory
    if 'saleCategory' in df_engineered.columns:
        print(f"\nUnique saleCategory values: {df_engineered['saleCategory'].nunique()}")
        print(f"saleCategory value counts:\n{df_engineered['saleCategory'].value_counts()}")
        
        sale_category_dummies = pd.get_dummies(
            df_engineered['saleCategory'], 
            prefix='saleCategory',
            drop_first=False
        )
        df_engineered = pd.concat([df_engineered, sale_category_dummies], axis=1)
        print(f"Created {len(sale_category_dummies.columns)} saleCategory features")
    
    # 2. One-hot encoding for displayRegion
    if 'displayRegion' in df_engineered.columns:
        print(f"\nUnique displayRegion values: {df_engineered['displayRegion'].nunique()}")
        print(f"displayRegion value counts:\n{df_engineered['displayRegion'].value_counts()}")
        
        region_dummies = pd.get_dummies(
            df_engineered['displayRegion'], 
            prefix='region',
            drop_first=False
        )
        df_engineered = pd.concat([df_engineered, region_dummies], axis=1)
        print(f"Created {len(region_dummies.columns)} region features")
    
    # 3. One-hot encoding for top 15 cities
    if 'city' in df_engineered.columns:
        unique_cities = df_engineered['city'].nunique()
        print(f"\nUnique city values: {unique_cities}")
        top_cities = df_engineered['city'].value_counts().head(15)
        print(f"Top 15 cities:\n{top_cities}")
        
        top_city_list = top_cities.index.tolist()
        # Keep only top 15 city names, leave others as NaN so get_dummies creates columns only for top cities
        city_for_dummies = df_engineered['city'].where(df_engineered['city'].isin(top_city_list))
        city_dummies = pd.get_dummies(city_for_dummies, prefix='city', dummy_na=False)
        df_engineered = pd.concat([df_engineered, city_dummies], axis=1)
        print(f"Created {len(city_dummies.columns)} city features (top {len(top_city_list)})")
    
    print(f"\nFinal dataset shape: {df_engineered.shape}")
    
    return df_engineered


# Main execution
if __name__ == "__main__":
    input_file = '/workspaces/maxsold/data/raw_data/auction_search/auction_search_20251201.parquet'
    
    # Perform feature engineering
    df_processed = feature_engineering(input_file)

    print(df_processed.columns.tolist())

    # Keep only relevant columns for modeling
    columns_to_keep = [
        'amAuctionId', 'distanceMeters', 'totalBids',
        'hasShipping', 'lat', 'lng', 'average_bids_per_lot',
        'saleCategory_Business Downsizing', 'saleCategory_Charity/Fundraising',
        'saleCategory_Commercial Liquidation', 'saleCategory_Downsizing',
        'saleCategory_Estate Sale', 'saleCategory_Moving', 'saleCategory_Renovation',
        'saleCategory_Reseller', 'region_Barrie', 'region_GTA', 'region_Horseshoe',
        'region_Kingston', 'region_London', 'region_Peterborough', 'city_Aurora',
        'city_Belleville', 'city_Brampton', 'city_Burlington', 'city_Hamilton',
        'city_London', 'city_Markham', 'city_Mississauga', 'city_Newmarket',
        'city_Oakville', 'city_Pickering', 'city_Richmond Hill', 'city_St. Catharines',
        'city_Toronto', 'city_Vaughan'
    ]
    df_processed_model = df_processed[columns_to_keep]
    
    # Save the processed data
    output_file = '/workspaces/maxsold/data/engineered_data/auction_search/auction_search_20251201_engineered.parquet'
    df_processed_model.to_parquet(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

    #subset data to columns not kept for modeling and print first few rows
    columns_not_kept = [col for col in df_processed.columns if col not in columns_to_keep]
    print("\nColumns not kept for modeling:")
    print(columns_not_kept)
    print("\nFirst few rows of columns not kept for modeling:")
    print(df_processed[columns_not_kept].head())
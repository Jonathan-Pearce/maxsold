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
    
    # 2. One-hot encoding for display_Region
    if 'display_Region' in df_engineered.columns:
        print(f"\nUnique display_Region values: {df_engineered['display_Region'].nunique()}")
        print(f"display_Region value counts:\n{df_engineered['display_Region'].value_counts()}")
        
        region_dummies = pd.get_dummies(
            df_engineered['display_Region'], 
            prefix='region',
            drop_first=False
        )
        df_engineered = pd.concat([df_engineered, region_dummies], axis=1)
        print(f"Created {len(region_dummies.columns)} region features")
    
    # 3. One-hot encoding for city
    if 'city' in df_engineered.columns:
        print(f"\nUnique city values: {df_engineered['city'].nunique()}")
        print(f"Top 10 cities:\n{df_engineered['city'].value_counts().head(10)}")
        
        city_dummies = pd.get_dummies(
            df_engineered['city'], 
            prefix='city',
            drop_first=False
        )
        df_engineered = pd.concat([df_engineered, city_dummies], axis=1)
        print(f"Created {len(city_dummies.columns)} city features")
    
    print(f"\nFinal dataset shape: {df_engineered.shape}")
    
    return df_engineered


# Main execution
if __name__ == "__main__":
    input_file = '/workspaces/maxsold/data/auction_search/auction_search_20251201.parquet'
    
    # Perform feature engineering
    df_processed = feature_engineering(input_file)
    
    # Save the processed data
    output_file = '/workspaces/maxsold/data/auction_search/auction_search_20251201_engineered.parquet'
    df_processed.to_parquet(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    # Display sample of new features
    one_hot_columns = [col for col in df_processed.columns 
                       if col.startswith(('saleCategory_', 'region_', 'city_'))]
    print(f"\nSample of one-hot encoded features:")
    print(df_processed[one_hot_columns].head(10))
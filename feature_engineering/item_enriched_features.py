from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from collections import Counter

"""
This script performs feature engineering on enriched item details dataset.

Features created:
- Text length features (title, description, qualitative description)
- Brand features (has_brand, brand one-hot encoding)
- Category features (parsed from JSON, one-hot encoding for top categories)
- Condition features (one-hot encoding)
- Working status features
- Item complexity features (single vs multiple items)
- Attributes features (parsed from JSON, count features)
- Text quality features (description richness, keyword presence)
"""


def parse_json_field(value):
    """Safely parse JSON string field"""
    if pd.isna(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def extract_categories_list(df):
    """Extract all categories from JSON field into flat list"""
    all_categories = []
    for val in df['categories'].dropna():
        categories = parse_json_field(val)
        all_categories.extend(categories)
    return all_categories


def extract_brands_list(df):
    """Extract all brands from JSON field into flat list"""
    all_brands = []
    for val in df['brands'].dropna():
        brands = parse_json_field(val)
        all_brands.extend(brands)
    return all_brands


def extract_attribute_names(df):
    """Extract attribute names from JSON field"""
    all_attr_names = []
    for val in df['attributes'].dropna():
        attributes = parse_json_field(val)
        for attr in attributes:
            if isinstance(attr, dict) and 'name' in attr:
                all_attr_names.append(attr['name'])
    return all_attr_names


def feature_engineering(df):
    """
    Perform feature engineering on enriched item details dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with enriched item data
    
    Returns:
    pd.DataFrame: Dataframe with engineered features
    """
    # Create a copy
    df_engineered = df.copy()
    
    print("=" * 60)
    print("Starting feature engineering on enriched item details")
    print(f"Original shape: {df_engineered.shape}")
    print("=" * 60)
    
    # ===== TEXT LENGTH FEATURES =====
    print("\n1. Creating text length features...")
    
    df_engineered['title_length'] = df_engineered['title'].fillna('').str.len()
    df_engineered['description_length'] = df_engineered['description'].fillna('').str.len()
    df_engineered['qualitative_description_length'] = df_engineered['qualitativeDescription'].fillna('').str.len()
    
    # Binary: has description or not
    df_engineered['has_description'] = (df_engineered['description_length'] > 0).astype(int)
    df_engineered['has_qualitative_desc'] = (df_engineered['qualitative_description_length'] > 0).astype(int)
    
    # Description richness score (longer = more detailed)
    df_engineered['description_richness'] = (
        df_engineered['title_length'] * 0.3 + 
        df_engineered['description_length'] * 0.5 + 
        df_engineered['qualitative_description_length'] * 0.2
    )
    
    print(f"  - Created title_length (mean: {df_engineered['title_length'].mean():.1f})")
    print(f"  - Created description_length (mean: {df_engineered['description_length'].mean():.1f})")
    print(f"  - Created description_richness")
    
    # ===== BRAND FEATURES =====
    print("\n2. Creating brand features...")
    
    # Has brand (from single brand field)
    df_engineered['has_brand'] = df_engineered['brand'].notna().astype(int)
    
    # Has multiple brands (from brands array)
    df_engineered['has_multiple_brands'] = (df_engineered['brands_count'] > 1).astype(int)
    
    # One-hot encode top brands
    top_brands = df_engineered['brand'].value_counts().head(20).index.tolist()
    print(f"  - Top 20 brands: {', '.join(top_brands[:5])}...")
    
    for brand in top_brands:
        col_name = f"brand_{re.sub(r'[^a-zA-Z0-9]', '_', brand)}"
        df_engineered[col_name] = (df_engineered['brand'] == brand).astype(int)
    
    print(f"  - Created {len(top_brands)} brand indicator features")
    print(f"  - {df_engineered['has_brand'].sum()} items ({df_engineered['has_brand'].mean()*100:.1f}%) have brand")
    
    # ===== CATEGORY FEATURES =====
    print("\n3. Creating category features...")
    
    # Extract all categories
    all_categories = extract_categories_list(df_engineered)
    category_counts = Counter(all_categories)
    top_categories = [cat for cat, _ in category_counts.most_common(25)]
    
    print(f"  - Total unique categories: {len(category_counts)}")
    print(f"  - Top 5 categories: {', '.join(top_categories[:5])}...")
    
    # Create indicator for top categories
    def has_category(categories_json, target_category):
        categories = parse_json_field(categories_json)
        return int(target_category in categories)
    
    for category in top_categories:
        col_name = f"cat_{re.sub(r'[^a-zA-Z0-9]', '_', category)}"
        df_engineered[col_name] = df_engineered['categories'].apply(
            lambda x: has_category(x, category)
        )
    
    print(f"  - Created {len(top_categories)} category indicator features")
    
    # ===== CONDITION FEATURES =====
    print("\n4. Creating condition features...")
    
    # One-hot encode condition
    if 'condition' in df_engineered.columns:
        condition_dummies = pd.get_dummies(
            df_engineered['condition'], 
            prefix='condition',
            drop_first=False,
            dummy_na=False
        )
        df_engineered = pd.concat([df_engineered, condition_dummies], axis=1)
        print(f"  - Created {len(condition_dummies.columns)} condition features")
        print(f"  - Condition distribution:")
        for col in condition_dummies.columns:
            count = condition_dummies[col].sum()
            pct = (count / len(df_engineered)) * 100
            print(f"    {col}: {count} ({pct:.1f}%)")
    
    # ===== WORKING STATUS FEATURES =====
    print("\n5. Creating working status features...")
    
    # Convert working to numeric (True=1, False=0, None=NaN)
    if 'working' in df_engineered.columns:
        df_engineered['is_working'] = df_engineered['working'].apply(
            lambda x: 1 if x is True else (0 if x is False else np.nan)
        )
        working_known = df_engineered['is_working'].notna().sum()
        working_yes = df_engineered['is_working'].sum()
        print(f"  - Working status known: {working_known} ({working_known/len(df_engineered)*100:.1f}%)")
        if working_known > 0:
            print(f"  - Items working: {working_yes} ({working_yes/working_known*100:.1f}% of known)")
    
    # ===== ITEM COMPLEXITY FEATURES =====
    print("\n6. Creating item complexity features...")
    
    # Single vs multiple items
    if 'singleKeyItem' in df_engineered.columns:
        df_engineered['is_single_item'] = df_engineered['singleKeyItem'].apply(
            lambda x: 1 if x is True else (0 if x is False else np.nan)
        )
    
    # Number of items (from numItems field)
    if 'numItems' in df_engineered.columns:
        df_engineered['num_items_log'] = np.log1p(df_engineered['numItems'].fillna(1))
        
        # Binned number of items
        df_engineered['items_category'] = pd.cut(
            df_engineered['numItems'].fillna(1),
            bins=[0, 1, 2, 5, 10, float('inf')],
            labels=['single', 'pair', 'few', 'several', 'many']
        )
        
        # One-hot encode items category
        items_cat_dummies = pd.get_dummies(
            df_engineered['items_category'],
            prefix='items',
            drop_first=False
        )
        df_engineered = pd.concat([df_engineered, items_cat_dummies], axis=1)
        
        print(f"  - numItems range: {df_engineered['numItems'].min():.0f} to {df_engineered['numItems'].max():.0f}")
        print(f"  - Created items_category with bins: single, pair, few, several, many")
    
    # Item count from items array
    if 'items_count' in df_engineered.columns:
        df_engineered['has_items_detail'] = (df_engineered['items_count'] > 0).astype(int)
        df_engineered['items_detail_log'] = np.log1p(df_engineered['items_count'].fillna(0))
    
    # ===== ATTRIBUTES FEATURES =====
    print("\n7. Creating attributes features...")
    
    # Attributes count
    if 'attributes_count' in df_engineered.columns:
        df_engineered['has_attributes'] = (df_engineered['attributes_count'] > 0).astype(int)
        df_engineered['attributes_log'] = np.log1p(df_engineered['attributes_count'].fillna(0))
        
        print(f"  - Items with attributes: {df_engineered['has_attributes'].sum()} ({df_engineered['has_attributes'].mean()*100:.1f}%)")
        print(f"  - Avg attributes per item: {df_engineered['attributes_count'].mean():.1f}")
    
    # Extract top attribute names
    all_attr_names = extract_attribute_names(df_engineered)
    attr_counts = Counter(all_attr_names)
    top_attrs = [attr for attr, _ in attr_counts.most_common(15)]
    
    print(f"  - Total unique attributes: {len(attr_counts)}")
    print(f"  - Top 5 attributes: {', '.join(top_attrs[:5])}...")
    
    # Create indicator for top attributes
    def has_attribute(attributes_json, target_attr):
        attributes = parse_json_field(attributes_json)
        for attr in attributes:
            if isinstance(attr, dict) and attr.get('name') == target_attr:
                return 1
        return 0
    
    for attr_name in top_attrs:
        col_name = f"attr_{re.sub(r'[^a-zA-Z0-9]', '_', attr_name)}"
        df_engineered[col_name] = df_engineered['attributes'].apply(
            lambda x: has_attribute(x, attr_name)
        )
    
    print(f"  - Created {len(top_attrs)} attribute indicator features")
    
    # ===== SERIES LINE FEATURE =====
    print("\n8. Creating series line features...")
    
    df_engineered['has_series_line'] = df_engineered['seriesLine'].notna().astype(int)
    print(f"  - Items with series line: {df_engineered['has_series_line'].sum()} ({df_engineered['has_series_line'].mean()*100:.1f}%)")
    
    # ===== TEXT QUALITY FEATURES =====
    print("\n9. Creating text quality features...")
    
    # Count keywords in description (luxury, vintage, new, etc.)
    quality_keywords = {
        'luxury': ['luxury', 'premium', 'high-end', 'designer'],
        'vintage': ['vintage', 'antique', 'retro', 'classic'],
        'new': ['new', 'unused', 'brand new', 'mint'],
        'damaged': ['damaged', 'broken', 'cracked', 'scratched', 'worn']
    }
    
    for keyword_type, keywords in quality_keywords.items():
        pattern = '|'.join(keywords)
        col_name = f'desc_has_{keyword_type}'
        df_engineered[col_name] = df_engineered['description'].fillna('').str.lower().str.contains(pattern, regex=True).astype(int)
        count = df_engineered[col_name].sum()
        print(f"  - {col_name}: {count} items ({count/len(df_engineered)*100:.1f}%)")
    
    # ===== DATA COMPLETENESS SCORE =====
    print("\n10. Creating data completeness score...")
    
    completeness_fields = [
        'has_brand', 'has_description', 'has_qualitative_desc', 
        'has_attributes', 'categories_count', 'items_count'
    ]
    
    # Normalize each to 0-1 and average
    completeness_scores = []
    for field in completeness_fields:
        if field in df_engineered.columns:
            if field.endswith('_count'):
                # Normalize count fields (cap at 10)
                normalized = np.minimum(df_engineered[field].fillna(0) / 10, 1)
            else:
                normalized = df_engineered[field].fillna(0)
            completeness_scores.append(normalized)
    
    if completeness_scores:
        df_engineered['data_completeness_score'] = np.mean(completeness_scores, axis=0)
        print(f"  - Average completeness score: {df_engineered['data_completeness_score'].mean():.3f}")
    
    print("\n" + "=" * 60)
    print(f"Final shape: {df_engineered.shape}")
    print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
    print("=" * 60)
    
    return df_engineered


if __name__ == "__main__":
    # Load raw data
    data_path = Path('/workspaces/maxsold/data/item_enriched_details/item_enriched_details_20251201.parquet')
    
    if not data_path.exists():
        print(f"Error: File not found at {data_path}")
        print("Please ensure the enriched item details file exists.")
        exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"Loaded data from: {data_path}")
    print(f"Original data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Process features
    df_processed = feature_engineering(df)
    
    print("\n" + "=" * 60)
    print("Feature Engineering Complete")
    print("=" * 60)
    print(f"\nProcessed data shape: {df_processed.shape}")
    
    # Select columns for modeling (excluding raw text and JSON fields)
    columns_to_exclude = [
        'title', 'description', 'qualitativeDescription', 'brand', 'seriesLine',
        'condition', 'working', 'singleKeyItem', 'brands', 'categories', 
        'items', 'attributes', 'items_category', 'amLotId'
    ]
    
    columns_to_keep = [col for col in df_processed.columns if col not in columns_to_exclude]
    
    df_processed_model = df_processed[columns_to_keep]
    
    print(f"\nColumns kept for modeling: {len(columns_to_keep)}")
    print(f"Columns excluded (raw text/JSON): {len(columns_to_exclude)}")
    
    # Save to engineered data directory
    output_dir = Path('/workspaces/maxsold/data/engineered_data/item_enriched_details')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'item_enriched_details_20251201_engineered.parquet'
    df_processed_model.to_parquet(output_path, index=False, compression='snappy')
    
    print(f"\n{'=' * 60}")
    print(f"Saved processed data to: {output_path}")
    print(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'=' * 60}")
    
    # Display summary statistics for key features
    print("\nKey Feature Statistics:")
    print("-" * 60)
    
    numeric_features = df_processed_model.select_dtypes(include=[np.number]).columns[:10]
    print("\nFirst 10 numeric features:")
    print(df_processed_model[numeric_features].describe())
    
    print("\nSample of processed data:")
    print(df_processed_model.head())
    
    print("\n✓ Feature engineering complete!")
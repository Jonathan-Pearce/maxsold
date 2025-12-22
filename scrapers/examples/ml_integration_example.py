#!/usr/bin/env python3
"""
Example: ML Model Integration
Shows how to integrate the extractors with your ML model pipeline
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import json

# Add scrapers to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors import (
    fetch_auction_details, extract_auction_from_json,
    fetch_auction_items, extract_items_from_json,
    fetch_enriched_details, extract_enriched_data,
    fetch_item_bid_history, extract_bids_from_json
)


class MaxSoldDataFetcher:
    """
    Wrapper class for fetching MaxSold data for ML predictions
    Use this in your ML pipeline to get real-time data
    """
    
    def __init__(self, include_bid_history: bool = False):
        """
        Args:
            include_bid_history: Whether to fetch bid history (slower but more features)
        """
        self.include_bid_history = include_bid_history
    
    def fetch_item_features(self, item_id: str, auction_id: str) -> Dict[str, Any]:
        """
        Fetch all features for a single item
        
        Args:
            item_id: MaxSold item ID
            auction_id: MaxSold auction ID containing the item
            
        Returns:
            Dictionary with all features ready for ML model
        """
        features = {
            'item_id': item_id,
            'auction_id': auction_id,
            'status': 'success',
            'error': None
        }
        
        try:
            # Fetch auction-level data
            auction_json = fetch_auction_details(auction_id)
            auction_data = extract_auction_from_json(auction_json, auction_id)
            features['auction'] = auction_data
            
            # Fetch item-level data
            items_json = fetch_auction_items(auction_id)
            all_items = extract_items_from_json(items_json, auction_id)
            
            # Find our specific item
            target_item = None
            for item in all_items:
                if str(item.get('id')) == str(item_id):
                    target_item = item
                    break
            
            if not target_item:
                features['status'] = 'error'
                features['error'] = f'Item {item_id} not found'
                return features
            
            features['item'] = target_item
            
            # Fetch enriched AI data
            try:
                enriched_json = fetch_enriched_details(item_id)
                enriched_data = extract_enriched_data(enriched_json, item_id)
                features['enriched'] = enriched_data
            except Exception as e:
                print(f"Warning: Could not fetch enriched data: {e}")
                features['enriched'] = None
            
            # Optionally fetch bid history
            if self.include_bid_history:
                try:
                    bid_json = fetch_item_bid_history(auction_id, item_id)
                    bids = extract_bids_from_json(bid_json, auction_id, item_id)
                    features['bids'] = bids
                except Exception as e:
                    print(f"Warning: Could not fetch bid history: {e}")
                    features['bids'] = []
            
        except Exception as e:
            features['status'] = 'error'
            features['error'] = str(e)
        
        return features
    
    def fetch_auction_features(self, auction_id: str, max_items: int = None) -> Dict[str, Any]:
        """
        Fetch features for all items in an auction
        
        Args:
            auction_id: MaxSold auction ID
            max_items: Optional limit on number of items to fetch
            
        Returns:
            Dictionary with auction and all item features
        """
        result = {
            'auction_id': auction_id,
            'status': 'success',
            'error': None,
            'items': []
        }
        
        try:
            # Fetch auction details
            auction_json = fetch_auction_details(auction_id)
            auction_data = extract_auction_from_json(auction_json, auction_id)
            result['auction'] = auction_data
            
            # Fetch all items
            items_json = fetch_auction_items(auction_id)
            all_items = extract_items_from_json(items_json, auction_id)
            
            # Limit items if specified
            if max_items:
                all_items = all_items[:max_items]
            
            result['items'] = all_items
            result['item_count'] = len(all_items)
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result


class MockMLModel:
    """
    Mock ML model to demonstrate integration
    Replace this with your actual trained model
    """
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Path to your trained model (pickle, joblib, etc.)
        """
        self.model_path = model_path
        # In reality, you'd load your model here:
        # self.model = joblib.load(model_path)
        # self.scaler = joblib.load(scaler_path)
        # etc.
    
    def extract_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from raw data
        This is where you'd implement your feature engineering
        """
        features = {}
        
        # Item-level features
        if 'item' in raw_data:
            item = raw_data['item']
            features['current_bid'] = float(item.get('current_bid') or 0)
            features['minimum_bid'] = float(item.get('minimum_bid') or 0)
            features['starting_bid'] = float(item.get('starting_bid') or 0)
            features['bid_count'] = int(item.get('bid_count') or 0)
            features['buyer_premium'] = float(item.get('buyer_premium') or 0)
            features['viewed'] = int(item.get('viewed') or 0)
            
            # Text features (you'd use NLP/embeddings here)
            features['title_length'] = len(item.get('title', ''))
            features['desc_length'] = len(item.get('description', ''))
        
        # Auction-level features
        if 'auction' in raw_data:
            auction = raw_data['auction']
            features['catalog_lots'] = int(auction.get('catalog_lots') or 0)
            features['extended_bidding'] = int(auction.get('extended_bidding') or 0)
        
        # Enriched AI features
        if 'enriched' in raw_data and raw_data['enriched']:
            enriched = raw_data['enriched']
            features['has_brand'] = 1 if enriched.get('brand') else 0
            features['brands_count'] = int(enriched.get('brands_count') or 0)
            features['categories_count'] = int(enriched.get('categories_count') or 0)
            features['items_count'] = int(enriched.get('items_count') or 0)
            features['attributes_count'] = int(enriched.get('attributes_count') or 0)
        
        # Bid history features
        if 'bids' in raw_data and raw_data['bids']:
            bids = raw_data['bids']
            features['total_bids'] = len(bids)
            features['proxy_bid_ratio'] = sum(1 for b in bids if b.get('isproxy')) / len(bids)
            
            # Bid timing features
            if len(bids) > 1:
                amounts = [b.get('amount', 0) for b in bids if b.get('amount')]
                if amounts:
                    features['bid_mean'] = sum(amounts) / len(amounts)
                    features['bid_max'] = max(amounts)
        
        return features
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction using extracted features
        
        Args:
            features: Dictionary of numerical features
            
        Returns:
            Prediction results
        """
        # In reality, you'd do:
        # feature_vector = self.prepare_feature_vector(features)
        # prediction = self.model.predict(feature_vector)
        # confidence = self.model.predict_proba(feature_vector)
        
        # Mock prediction
        current_bid = features.get('current_bid', 0)
        bid_count = features.get('bid_count', 0)
        
        # Simple mock logic
        predicted_final = current_bid * (1 + (bid_count * 0.1))
        
        return {
            'predicted_final_price': round(predicted_final, 2),
            'confidence': 0.75,
            'features_count': len(features)
        }
    
    def predict_from_url(self, item_id: str, auction_id: str) -> Dict[str, Any]:
        """
        End-to-end prediction from item URL/ID
        
        Args:
            item_id: MaxSold item ID
            auction_id: MaxSold auction ID
            
        Returns:
            Prediction with confidence and metadata
        """
        # Fetch data
        fetcher = MaxSoldDataFetcher(include_bid_history=True)
        raw_data = fetcher.fetch_item_features(item_id, auction_id)
        
        if raw_data['status'] == 'error':
            return {'error': raw_data['error'], 'prediction': None}
        
        # Extract features
        features = self.extract_features(raw_data)
        
        # Make prediction
        prediction = self.predict(features)
        
        # Add metadata
        result = {
            'item_id': item_id,
            'auction_id': auction_id,
            'item_title': raw_data.get('item', {}).get('title', 'N/A'),
            'current_bid': raw_data.get('item', {}).get('current_bid', 0),
            'prediction': prediction,
            'timestamp': pd.Timestamp.now().isoformat() if 'pd' in dir() else None
        }
        
        return result


def example_single_prediction():
    """Example: Predict final price for a single item"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Item Prediction")
    print("="*60 + "\n")
    
    # Initialize model
    model = MockMLModel()
    
    # Make prediction (use real IDs)
    auction_id = "12345"  # Replace with real auction ID
    item_id = "67890"      # Replace with real item ID
    
    print(f"Making prediction for item {item_id} in auction {auction_id}...")
    print("(Note: Using mock IDs - replace with real ones to test)\n")
    
    # This is how you'd use it:
    # result = model.predict_from_url(item_id, auction_id)
    # print(json.dumps(result, indent=2))


def example_batch_predictions():
    """Example: Predict for multiple items"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Predictions for Auction")
    print("="*60 + "\n")
    
    fetcher = MaxSoldDataFetcher(include_bid_history=False)
    model = MockMLModel()
    
    auction_id = "12345"  # Replace with real auction ID
    
    print(f"Fetching all items from auction {auction_id}...")
    print("(Note: Using mock ID - replace with real one to test)\n")
    
    # This is how you'd use it:
    # auction_data = fetcher.fetch_auction_features(auction_id, max_items=10)
    # 
    # if auction_data['status'] == 'success':
    #     predictions = []
    #     for item in auction_data['items']:
    #         item_id = item.get('id')
    #         features = model.extract_features({'item': item, 'auction': auction_data['auction']})
    #         pred = model.predict(features)
    #         predictions.append({
    #             'item_id': item_id,
    #             'title': item.get('title'),
    #             'prediction': pred
    #         })
    #     
    #     print(f"Made {len(predictions)} predictions")


def main():
    """Main demo"""
    print("="*60)
    print("ML INTEGRATION EXAMPLES")
    print("="*60)
    print("\nThis shows how to integrate the extractors with your ML model.")
    print("The key advantage: fetch data on-demand without saving files!")
    
    example_single_prediction()
    example_batch_predictions()
    
    print("\n" + "="*60)
    print("INTEGRATION SUMMARY")
    print("="*60)
    print("""
Key Components:
1. MaxSoldDataFetcher - Wraps extractors for easy data fetching
2. Your ML Model - Loads trained model and makes predictions
3. Feature Engineering - Extracts numerical features from raw data

Usage in Production:
    fetcher = MaxSoldDataFetcher()
    model = YourMLModel('path/to/model.pkl')
    
    # Single prediction
    result = model.predict_from_url(item_id, auction_id)
    
    # Batch predictions
    auction_data = fetcher.fetch_auction_features(auction_id)
    for item in auction_data['items']:
        prediction = model.predict(...)

No files are saved - perfect for a live API or web service!
    """)


if __name__ == "__main__":
    main()

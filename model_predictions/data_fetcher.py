"""
Data fetching utilities for model prediction pipeline.
Handles API requests to MaxSold endpoints.
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class MaxSoldDataFetcher:
    """Fetches data from MaxSold APIs for a given item URL"""
    
    BASE_ENRICHED_URL = "https://api.maxsold.com/listings/am/{item_id}/enriched"
    BASE_AUCTION_URL = "https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}"
    BASE_BID_HISTORY_URL = "https://maxsold.maxsold.com/msapi/auctions/items?auctionid={auction_id}&itemid={item_id}"
    
    def __init__(self, output_dir: str = "data/model_predictions"):
        """
        Initialize the data fetcher.
        
        Args:
            output_dir: Directory to save fetched data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @staticmethod
    def extract_item_id_from_url(url: str) -> Optional[str]:
        """
        Extract item ID from MaxSold listing URL.
        
        Args:
            url: MaxSold listing URL (e.g., https://maxsold.com/listing/7561262/tascam-da-60-stereo-dat-recorder)
        
        Returns:
            Item ID as string, or None if not found
        
        Examples:
            >>> MaxSoldDataFetcher.extract_item_id_from_url("https://maxsold.com/listing/7561262/tascam-da-60-stereo-dat-recorder")
            '7561262'
        """
        # Pattern: /listing/{item_id}/
        pattern = r'/listing/(\d+)/'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        
        # Also try pattern without trailing slash or description
        pattern2 = r'/listing/(\d+)'
        match = re.search(pattern2, url)
        if match:
            return match.group(1)
        
        return None
    
    def fetch_enriched_data(self, item_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetch enriched item data from API.
        
        Args:
            item_id: Item ID to fetch
        
        Returns:
            Tuple of (JSON data dict, error message if any)
        """
        url = self.BASE_ENRICHED_URL.format(item_id=item_id)
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Save raw JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enriched_item_{item_id}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved enriched data to {filepath}")
            return data, None
            
        except requests.RequestException as e:
            error_msg = f"Error fetching enriched data for item {item_id}: {e}"
            print(error_msg)
            return None, error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON for item {item_id}: {e}"
            print(error_msg)
            return None, error_msg
    
    def fetch_auction_data(self, auction_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetch auction data from API.
        
        Args:
            auction_id: Auction ID to fetch
        
        Returns:
            Tuple of (JSON data dict, error message if any)
        """
        url = self.BASE_AUCTION_URL.format(auction_id=auction_id)
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Save raw JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auction_{auction_id}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved auction data to {filepath}")
            return data, None
            
        except requests.RequestException as e:
            error_msg = f"Error fetching auction data for auction {auction_id}: {e}"
            print(error_msg)
            return None, error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON for auction {auction_id}: {e}"
            print(error_msg)
            return None, error_msg
    
    def fetch_bid_history(self, auction_id: str, item_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetch bid history from API.
        
        Args:
            auction_id: Auction ID
            item_id: Item ID
        
        Returns:
            Tuple of (JSON data dict, error message if any)
        """
        url = self.BASE_BID_HISTORY_URL.format(auction_id=auction_id, item_id=item_id)
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Save raw JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bid_history_auction_{auction_id}_item_{item_id}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved bid history to {filepath}")
            return data, None
            
        except requests.RequestException as e:
            error_msg = f"Error fetching bid history for auction {auction_id}, item {item_id}: {e}"
            print(error_msg)
            return None, error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON for bid history: {e}"
            print(error_msg)
            return None, error_msg

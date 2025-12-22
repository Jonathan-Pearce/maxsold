"""
Quick test script to verify the refactored structure works correctly
Tests basic imports and function availability
"""

import sys
from pathlib import Path

# Add scrapers to path
scrapers_path = Path(__file__).parent
sys.path.insert(0, str(scrapers_path))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    # Test utils imports
    print("  ✓ Testing utils...")
    from utils.config import HEADERS, API_URLS, DEFAULT_OUTPUT_DIRS
    from utils.file_io import save_to_parquet, load_from_parquet
    assert isinstance(HEADERS, dict)
    assert isinstance(API_URLS, dict)
    print("    ✓ Utils imports successful")
    
    # Test extractor imports
    print("  ✓ Testing extractors...")
    from extractors.auction_search import fetch_sales_search, extract_sales_from_json
    from extractors.auction_details import fetch_auction_details, extract_auction_from_json
    from extractors.item_details import fetch_auction_items, extract_items_from_json
    from extractors.bid_history import fetch_item_bid_history, extract_bids_from_json
    from extractors.item_enriched import fetch_enriched_details, extract_enriched_data
    print("    ✓ Extractor imports successful")
    
    # Test pipeline imports
    print("  ✓ Testing pipelines...")
    from pipelines.auction_search_pipeline import run_auction_search_pipeline
    from pipelines.auction_details_pipeline import run_auction_details_pipeline
    from pipelines.item_details_pipeline import run_item_details_pipeline
    from pipelines.bid_history_pipeline import run_bid_history_pipeline
    from pipelines.item_enriched_pipeline import run_item_enriched_pipeline
    print("    ✓ Pipeline imports successful")
    
    print("\n✅ All imports successful!")
    return True


def test_extractor_availability():
    """Test that extractor functions are callable"""
    print("\nTesting extractor function signatures...")
    
    from extractors import (
        fetch_sales_search,
        extract_sales_from_json,
        fetch_auction_details,
        extract_auction_from_json,
        fetch_auction_items,
        extract_items_from_json,
        fetch_item_bid_history,
        extract_bids_from_json,
        fetch_enriched_details,
        extract_enriched_data
    )
    
    # Check they are callable
    assert callable(fetch_sales_search)
    assert callable(extract_sales_from_json)
    assert callable(fetch_auction_details)
    assert callable(extract_auction_from_json)
    assert callable(fetch_auction_items)
    assert callable(extract_items_from_json)
    assert callable(fetch_item_bid_history)
    assert callable(extract_bids_from_json)
    assert callable(fetch_enriched_details)
    assert callable(extract_enriched_data)
    
    print("  ✓ All extractor functions are callable")
    print("\n✅ Function availability test passed!")
    return True


def test_pipeline_availability():
    """Test that pipeline functions are callable"""
    print("\nTesting pipeline function signatures...")
    
    from pipelines import (
        run_auction_search_pipeline,
        run_auction_details_pipeline,
        run_item_details_pipeline,
        run_bid_history_pipeline,
        run_item_enriched_pipeline
    )
    
    # Check they are callable
    assert callable(run_auction_search_pipeline)
    assert callable(run_auction_details_pipeline)
    assert callable(run_item_details_pipeline)
    assert callable(run_bid_history_pipeline)
    assert callable(run_item_enriched_pipeline)
    
    print("  ✓ All pipeline functions are callable")
    print("\n✅ Pipeline availability test passed!")
    return True


def test_config():
    """Test configuration values"""
    print("\nTesting configuration...")
    
    from utils.config import HEADERS, API_URLS, DEFAULT_OUTPUT_DIRS
    
    # Check API URLs
    required_urls = ['auction_search', 'auction_items', 'item_enriched']
    for url_key in required_urls:
        assert url_key in API_URLS, f"Missing API URL: {url_key}"
        assert API_URLS[url_key].startswith('http'), f"Invalid URL: {API_URLS[url_key]}"
    
    print("  ✓ API URLs configured correctly")
    
    # Check headers
    assert 'User-Agent' in HEADERS
    print("  ✓ HTTP headers configured correctly")
    
    # Check default dirs
    required_dirs = ['auction_search', 'auction_details', 'item_details', 'bid_history', 'item_enriched']
    for dir_key in required_dirs:
        assert dir_key in DEFAULT_OUTPUT_DIRS, f"Missing default dir: {dir_key}"
    
    print("  ✓ Default directories configured correctly")
    
    print("\n✅ Configuration test passed!")
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("REFACTORING VERIFICATION TESTS")
    print("="*60)
    
    try:
        test_imports()
        test_extractor_availability()
        test_pipeline_availability()
        test_config()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nThe refactored structure is working correctly.")
        print("\nYou can now:")
        print("  1. Import extractors for live ML predictions")
        print("  2. Import pipelines for batch scraping")
        print("  3. Use the CLI scripts as before")
        print("\nSee REFACTORING_GUIDE.md for usage examples.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

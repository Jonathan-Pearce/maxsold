import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_auction_data(sale: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract relevant fields from a single sale object."""
    try:
        # Extract nested address fields
        address = sale.get("address", {})
        lat_lng = address.get("latLng", {})
        
        return {
            "amAuctionId": sale.get("amAuctionId", ""),
            "title": sale.get("title", ""),
            "saleType": sale.get("saleType", ""),
            "saleCategory": sale.get("saleCategory", ""),
            "openTime": sale.get("openTime", ""),
            "closeTime": sale.get("closeTime", ""),
            "totalBids": sale.get("totalBids", 0),
            "numberLots": sale.get("numberLots", 0),
            "city": address.get("city", ""),
            "region": address.get("region", ""),
            "country": address.get("country", ""),
            "Lat": lat_lng.get("Lat", ""),
            "Lng": lat_lng.get("Lng", ""),
        }
    except Exception as e:
        print(f"Error extracting auction data: {e}")
        return None


def process_json_files(input_dir: str, output_csv: str) -> int:
    """
    Process all JSON files in input_dir and write extracted data to CSV.
    
    Args:
        input_dir: Directory containing JSON files
        output_csv: Path to output CSV file
        
    Returns:
        Total number of auctions extracted
    """
    json_dir = Path(input_dir)
    
    if not json_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return 0
    
    # Find all JSON files
    json_files = sorted(json_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0
    
    print(f"Found {len(json_files)} JSON files")
    
    all_auctions = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract sales from the "data" array
            sales = data.get("data", [])
            
            if not sales:
                print(f"Warning: No sales found in {json_file.name}")
                continue
            
            # Extract data from each sale
            for sale in sales:
                auction_data = extract_auction_data(sale)
                if auction_data:
                    all_auctions.append(auction_data)
            
            print(f"Processed {json_file.name}: {len(sales)} auctions")
            
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file.name}: {e}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # Write to CSV
    if all_auctions:
        fieldnames = [
            "amAuctionId", "title", "saleType", "saleCategory",
            "openTime", "closeTime", "totalBids", "numberLots",
            "city", "region", "country", "Lat", "Lng"
        ]
        
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_auctions)
        
        print(f"\nSuccessfully wrote {len(all_auctions)} auctions to {output_csv}")
    else:
        print("No auction data extracted")
    
    return len(all_auctions)


if __name__ == "__main__":
    import sys
    
    # Default paths
    default_input = "/workspaces/maxsold/data/auction/raw_json/2025-11-21"
    default_output = "/workspaces/maxsold/data/auction/clean/sales_2025-11-21.csv"
    
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_csv = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    count = process_json_files(input_dir, output_csv)
    print(f"\nTotal auctions extracted: {count}")
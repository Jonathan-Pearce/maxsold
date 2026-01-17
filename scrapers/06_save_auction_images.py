"""
Image Saving Pipeline for MaxSold Auction Items

This script downloads and processes images from MaxSold auction items:
- Fetches auction IDs from Kaggle dataset
- Downloads first 3 images per item
- Resizes images to 256px max dimension
- Saves in WebP format for memory efficiency
- Processes in batches to avoid memory issues
"""

import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from datetime import datetime
from PIL import Image
from io import BytesIO
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

API_URL = "https://maxsold.maxsold.com/msapi/auctions/items"
MAX_IMAGE_DIMENSION = 256
MAX_IMAGES_PER_ITEM = 3
IMAGE_FORMAT = "webp"
IMAGE_QUALITY = 85
REQUEST_TIMEOUT = 30
BATCH_SIZE = 10  # Process auctions in batches


def fetch_auction_items(auction_id: str, timeout: int = REQUEST_TIMEOUT) -> Optional[Dict[str, Any]]:
    """Fetch auction items from MaxSold API"""
    params = {"auctionid": auction_id, "limit": 1000}
    
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ✗ Error fetching auction {auction_id}: {e}", file=sys.stderr)
        return None


def extract_image_urls(data: Dict[str, Any], auction_id: str) -> List[Dict[str, Any]]:
    """
    Extract image URLs from auction items JSON response.
    Returns list of dicts with item_id, auction_id, image_url, and image_index.
    """
    items = None
    
    # Navigate to items array
    if isinstance(data, dict):
        if "auction" in data and isinstance(data["auction"], dict):
            auction_obj = data["auction"]
            if "items" in auction_obj and isinstance(auction_obj["items"], list):
                items = auction_obj["items"]
        
        if items is None and "items" in data and isinstance(data["items"], list):
            items = data["items"]
    
    if not items:
        return []
    
    image_entries = []
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        item_id = item.get("id")
        if not item_id:
            continue
        
        # Extract images array
        images = item.get("images", [])
        if not isinstance(images, list):
            continue
        
        # Take only first 3 images
        for idx, image_url in enumerate(images[:MAX_IMAGES_PER_ITEM]):
            if image_url and isinstance(image_url, str):
                image_entries.append({
                    "auction_id": auction_id,
                    "item_id": str(item_id),
                    "image_url": image_url,
                    "image_index": idx
                })
    
    return image_entries


def resize_image(image: Image.Image, max_dimension: int = MAX_IMAGE_DIMENSION) -> Image.Image:
    """
    Resize image so the larger dimension is max_dimension pixels,
    scaling the other dimension proportionally.
    """
    width, height = image.size
    
    if width <= max_dimension and height <= max_dimension:
        return image
    
    if width > height:
        new_width = max_dimension
        new_height = int((max_dimension / width) * height)
    else:
        new_height = max_dimension
        new_width = int((max_dimension / height) * width)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def download_and_save_image(
    image_url: str,
    output_path: Path,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    timeout: int = REQUEST_TIMEOUT
) -> bool:
    """
    Download an image from URL, resize it, and save in WebP format.
    Returns True if successful, False otherwise.
    """
    try:
        # Download image
        response = requests.get(image_url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        
        # Open image
        image = Image.open(BytesIO(response.content))
        
        # Convert to RGB for consistent output format
        # This ensures all saved images are in RGB mode regardless of source format
        original_mode = image.mode
        if original_mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if original_mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif original_mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        resized_image = resize_image(image, max_dimension)
        
        # Save as WebP
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized_image.save(output_path, format="WEBP", quality=IMAGE_QUALITY)
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error processing image {image_url}: {e}", file=sys.stderr)
        return False


def process_auction_batch(
    auction_ids: List[str],
    output_dir: Path,
    metadata_list: List[Dict[str, Any]]
) -> int:
    """
    Process a batch of auctions and save their images.
    Returns count of successfully saved images.
    """
    images_saved = 0
    
    for auction_id in auction_ids:
        # Fetch auction items
        data = fetch_auction_items(auction_id)
        if not data:
            continue
        
        # Extract image URLs
        image_entries = extract_image_urls(data, auction_id)
        
        if not image_entries:
            continue
        
        print(f"  Processing {len(image_entries)} images from auction {auction_id}...")
        
        # Download and save each image
        for entry in image_entries:
            image_url = entry["image_url"]
            item_id = entry["item_id"]
            image_index = entry["image_index"]
            
            # Create filename: auction_id/item_id_image_index.webp
            auction_dir = output_dir / auction_id
            filename = f"{item_id}_{image_index}.{IMAGE_FORMAT}"
            output_path = auction_dir / filename
            
            # Skip if already exists
            if output_path.exists():
                continue
            
            # Download and save
            if download_and_save_image(image_url, output_path):
                images_saved += 1
                
                # Add metadata
                metadata_list.append({
                    "auction_id": auction_id,
                    "item_id": item_id,
                    "image_index": image_index,
                    "image_url": image_url,
                    "saved_path": str(output_path.relative_to(output_dir.parent)),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
    
    return images_saved


def save_metadata(metadata_list: List[Dict[str, Any]], output_path: Path):
    """Save image metadata to parquet file"""
    if not metadata_list:
        print("No metadata to save.", file=sys.stderr)
        return
    
    df = pd.DataFrame(metadata_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved metadata: {len(df)} records to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")


def load_auction_ids_from_kaggle(
    dataset_name: str,
    file_name: str,
    download_path: Path,
    limit: Optional[int] = None
) -> List[str]:
    """
    Load auction IDs from Kaggle dataset.
    
    Parameters:
    dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
    file_name: Name of the file in the dataset
    download_path: Local path to download the dataset
    limit: Maximum number of auction IDs to return
    
    Returns:
    List of auction IDs as strings
    """
    try:
        # Import here to avoid requiring kaggle for basic functionality
        from utils.kaggle_pipeline import KaggleDataPipeline
        
        print(f"Downloading dataset from Kaggle: {dataset_name}")
        kaggle_pipeline = KaggleDataPipeline()
        
        # Download specific file
        kaggle_pipeline.download_dataset(
            dataset_name=dataset_name,
            download_path=download_path,
            file_name=file_name
        )
        
        # Load the file
        file_path = download_path / file_name
        df = kaggle_pipeline.load_dataset(file_path)
        
        # Extract auction IDs
        if 'amAuctionId' not in df.columns:
            print(f"Column 'amAuctionId' not found in {file_name}", file=sys.stderr)
            print(f"Available columns: {', '.join(df.columns.tolist())}", file=sys.stderr)
            return []
        
        # Get unique auction IDs
        auction_ids = df['amAuctionId'].dropna().unique().astype(str).tolist()
        
        # Apply limit if specified
        if limit:
            auction_ids = auction_ids[:limit]
        
        print(f"✓ Loaded {len(auction_ids)} auction IDs")
        return auction_ids
    
    except Exception as e:
        print(f"✗ Error loading auction IDs from Kaggle: {e}", file=sys.stderr)
        return []


def main(
    auction_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    kaggle_dataset: Optional[str] = None,
    kaggle_file: Optional[str] = None,
    limit_auctions: int = 100
):
    """
    Main function to run the image saving pipeline.
    
    Parameters:
    auction_ids: List of auction IDs to process (if not loading from Kaggle)
    output_dir: Directory to save images
    kaggle_dataset: Kaggle dataset identifier (e.g., 'username/dataset-name')
    kaggle_file: File name in Kaggle dataset
    limit_auctions: Maximum number of auctions to process
    """
    print("=" * 60)
    print("MaxSold Image Saving Pipeline")
    print("=" * 60)
    
    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"data/images/images_{timestamp}")
    
    # Load auction IDs
    if kaggle_dataset and kaggle_file:
        download_path = Path("data/raw_data/kaggle_temp")
        auction_ids = load_auction_ids_from_kaggle(
            dataset_name=kaggle_dataset,
            file_name=kaggle_file,
            download_path=download_path,
            limit=limit_auctions
        )
    elif not auction_ids:
        print("No auction IDs provided. Use --kaggle-dataset and --kaggle-file or provide auction IDs.", file=sys.stderr)
        return
    
    if not auction_ids:
        print("No auction IDs to process.", file=sys.stderr)
        return
    
    # Apply limit if specified
    if limit_auctions and len(auction_ids) > limit_auctions:
        auction_ids = auction_ids[:limit_auctions]
    
    print(f"\nConfiguration:")
    print(f"  Auctions to process: {len(auction_ids)}")
    print(f"  Max images per item: {MAX_IMAGES_PER_ITEM}")
    print(f"  Max image dimension: {MAX_IMAGE_DIMENSION}px")
    print(f"  Image format: {IMAGE_FORMAT}")
    print(f"  Output directory: {output_path}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 60)
    
    # Process auctions in batches
    metadata_list = []
    total_images_saved = 0
    # Use same timestamp as output directory for consistency
    metadata_timestamp = output_path.name.replace('images_', '')
    metadata_path = output_path.parent / f"image_metadata_{metadata_timestamp}.parquet"
    
    # Split auction IDs into batches
    num_batches = (len(auction_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(auction_ids))
        batch_auction_ids = auction_ids[start_idx:end_idx]
        
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing auctions {start_idx + 1}-{end_idx}...")
        
        # Process batch
        images_saved = process_auction_batch(
            auction_ids=batch_auction_ids,
            output_dir=output_path,
            metadata_list=metadata_list
        )
        
        total_images_saved += images_saved
        print(f"  ✓ Batch complete: {images_saved} images saved")
        
        # Save metadata after each batch
        if metadata_list:
            save_metadata(metadata_list, metadata_path)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"  Total auctions processed: {len(auction_ids)}")
    print(f"  Total images saved: {total_images_saved}")
    print(f"  Output directory: {output_path}")
    if metadata_list:
        print(f"  Metadata file: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and process images from MaxSold auction items"
    )
    parser.add_argument(
        "auction_ids",
        nargs="*",
        help="Auction ID(s) to process (optional if using Kaggle dataset)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for images"
    )
    parser.add_argument(
        "--kaggle-dataset",
        help="Kaggle dataset identifier (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--kaggle-file",
        help="File name in Kaggle dataset containing auction IDs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of auctions to process (default: 100)"
    )
    
    args = parser.parse_args()
    
    main(
        auction_ids=args.auction_ids if args.auction_ids else None,
        output_dir=args.output_dir,
        kaggle_dataset=args.kaggle_dataset,
        kaggle_file=args.kaggle_file,
        limit_auctions=args.limit
    )

# Image Saving Pipeline

This pipeline downloads and processes images from MaxSold auction items for machine learning and analysis purposes.

## Features

- ✅ Fetches auction IDs from Kaggle dataset or accepts manual input
- ✅ Limits processing to 100 auctions (configurable for testing/demo)
- ✅ Downloads first 3 images per item (or fewer if unavailable)
- ✅ Resizes images to 256px max dimension with proportional scaling
- ✅ Saves in WebP format for memory efficiency
- ✅ Processes in batches to avoid memory issues
- ✅ Generates metadata file tracking all downloaded images

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `Pillow>=10.0.0` - Image processing
- `pandas>=2.0.0` - Data handling
- `requests>=2.31.0` - API calls
- `kaggle>=1.5.16` - Kaggle dataset access

## Usage

### Using Kaggle Dataset

Download images using auction IDs from a Kaggle dataset:

```bash
python scrapers/06_save_auction_images.py \
    --kaggle-dataset "username/dataset-name" \
    --kaggle-file "auction_search.parquet" \
    --limit 100 \
    -o data/images/run_20260117
```

### Using Manual Auction IDs

Process specific auction IDs:

```bash
python scrapers/06_save_auction_images.py 103293 103482 103500 \
    --limit 3 \
    -o data/images/test_run
```

### Command Line Arguments

- `auction_ids` - Space-separated auction IDs (optional if using Kaggle)
- `--kaggle-dataset` - Kaggle dataset identifier (e.g., 'username/dataset-name')
- `--kaggle-file` - File name in Kaggle dataset containing auction IDs
- `--limit` - Maximum number of auctions to process (default: 100)
- `-o, --output-dir` - Output directory for images

## Output Structure

The pipeline creates the following structure:

```
data/images/
└── images_20260117/           # Timestamped output directory
    ├── 103293/                # Auction ID directory
    │   ├── 7433915_0.webp    # item_id_image_index.webp
    │   ├── 7433915_1.webp
    │   ├── 7433915_2.webp
    │   ├── 7433916_0.webp
    │   └── ...
    └── 103482/
        └── ...

data/images/
└── image_metadata_20260117.parquet  # Metadata file
```

## Metadata File

The metadata parquet file contains:
- `auction_id` - Auction identifier
- `item_id` - Item identifier
- `image_index` - Image index (0-2)
- `image_url` - Original image URL
- `saved_path` - Relative path to saved image
- `timestamp` - When the image was downloaded

## Technical Details

### Image Processing

- **Max Dimension**: 256 pixels (configurable via `MAX_IMAGE_DIMENSION`)
- **Scaling**: Proportional - if width > height, width becomes 256px and height scales proportionally, and vice versa
- **Format**: WebP with 85% quality for optimal compression
- **Color Mode**: All images converted to RGB for consistency

### Memory Efficiency

- **Batch Processing**: Processes 10 auctions at a time (configurable via `BATCH_SIZE`)
- **Incremental Saves**: Metadata saved after each batch
- **Skip Existing**: Automatically skips already downloaded images

### API Details

- **Endpoint**: `https://maxsold.maxsold.com/msapi/auctions/items`
- **Rate Limiting**: 0.1s delay between image downloads
- **Timeout**: 30 seconds per request
- **Max Images**: First 3 images per item

## Example API Response Structure

```json
{
  "auction": {
    "items": [
      {
        "id": "7433915",
        "title": "Item Title",
        "images": [
          "https://example.com/image1.jpg",
          "https://example.com/image2.jpg",
          "https://example.com/image3.jpg"
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Network Errors

If you encounter connection errors:
- Check your internet connection
- Verify the MaxSold API is accessible
- Consider increasing the `REQUEST_TIMEOUT` value

### Memory Issues

If running out of memory:
- Reduce `BATCH_SIZE` (default: 10)
- Reduce `--limit` to process fewer auctions
- Check available disk space

### Missing Dependencies

If imports fail:
```bash
pip install -r requirements.txt --upgrade
```

## Future Enhancements

Potential improvements:
- Add progress bars using tqdm
- Parallel download support
- Resume capability from last checkpoint
- Configurable image dimensions per use case
- Support for additional image formats

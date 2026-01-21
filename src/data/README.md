# Backfilling Scripts

This folder contains scripts for backfilling historical MaxSold auction data.

## Auction Location Data Scraper

### Overview
`extract_auction_location_data.py` extracts auction location data from the MaxSold API and filters auctions based on their distance from downtown Toronto.

### Features
- Extracts: `amAuctionId`, `lat`, `lng`, `postalCode`
- Calculates distance from downtown Toronto (43.653226, -79.383184) using Haversine formula
- Filters auctions within 201,168 meters (201.168 km) of Toronto
- **Parallel processing with configurable workers for faster scraping**
- Saves results in Parquet format
- Includes checkpoint saving for long-running scrapes
- Rate limiting to avoid API throttling
- Thread-safe concurrent request handling

### Usage

Basic usage (scrape IDs 1-100000):
```bash
python backfilling/extract_auction_location_data.py
```

With custom parameters:
```bash
python backfilling/extract_auction_location_data.py \
    --start-id 1 \
    --end-id 100000 \
    --output data/backfilling/auction_location_data.parquet \
    --delay 0.05 \
    --checkpoint 1000 \
    --workers 10
```

Fast mode (20 workers, minimal delay):
```bash
python backfilling/extract_auction_location_data.py \
    --workers 20 \
    --delay 0.01
```

### Parameters
- `--start-id`: Starting auction ID (default: 1)
- `--end-id`: Ending auction ID (default: 100000)
- `--output`: Output parquet file path (default: data/backfilling/auction_location_data.parquet)
- `--delay`: Delay between requests in seconds per worker (default: 0.05)
- `--checkpoint`: Save checkpoint every N records (default: 1000)
- `--workers`: Number of parallel workers (default: 10)

### Output
The script creates a Parquet file with the following columns:
- `amAuctionId`: The auction ID
- `lat`: Latitude
- `lng`: Longitude
- `postalCode`: Postal code (may be null)
- `distance_from_toronto_meters`: Calculated distance from downtown Toronto in meters

### Notes
- Only auctions within 201,168 meters of Toronto are saved
- Checkpoints are saved periodically to prevent data loss on long scrapes
- The script uses the Haversine formula for accurate distance calculation
- **Parallel processing**: Default 10 workers can be increased for faster scraping (adjust `--workers`)
- **Performance**: With 10 workers, expect ~10x speedup compared to sequential processing
- **Rate limiting**: Adjust `--delay` based on API rate limits (lower = faster but risk of throttling)

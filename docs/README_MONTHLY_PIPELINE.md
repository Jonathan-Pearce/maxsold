# Monthly Scraping Pipeline

Automated monthly scraping pipeline for MaxSold auction data with Kaggle dataset updates.

## Overview

This pipeline runs automatically on the 1st of every month and:

1. **Scrapes** the last 30 days of MaxSold auction data:
   - Auction search results
   - Auction details
   - Item details
   - Bid history
   - Enriched item details (AI-generated descriptions, brands, categories)

2. **Appends** new data to existing Kaggle datasets (deduplicates automatically)

3. **Uploads** updated datasets to Kaggle:
   - [raw-maxsold-auction](https://www.kaggle.com/datasets/pearcej/raw-maxsold-auction)
   - [raw-maxsold-item](https://www.kaggle.com/datasets/pearcej/raw-maxsold-item)
   - [raw-maxsold-bid](https://www.kaggle.com/datasets/pearcej/raw-maxsold-bid)
   - [raw-maxsold-item-enriched](https://www.kaggle.com/datasets/pearcej/raw-maxsold-item-enriched)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MONTHLY PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: AUCTION SEARCH
┌────────────────────────┐
│ 01_extract_auction_    │  Scrapes last 30 days
│    search.py           │  Outputs: auction IDs
│ (days=30)              │
└────────┬───────────────┘
         │
         ├──────────────────────┬────────────────────────┐
         ▼                      ▼                        ▼
┌────────────────────┐  ┌────────────────────┐  ┌──────────────┐
│ PHASE 2: DETAILS   │  │ PHASE 2: DETAILS   │  │              │
├────────────────────┤  ├────────────────────┤  │              │
│ 02_extract_auction_│  │ 03_extract_items_  │  │              │
│    details.py      │  │    details.py      │  │              │
└────────┬───────────┘  └────────┬───────────┘  │              │
         │                       │               │              │
         │                       ├───────────────┼──────┐       │
         │                       │               │      │       │
         │                       ▼               ▼      ▼       │
         │              ┌────────────────┐  ┌────────────────┐ │
         │              │ PHASE 3: BID & │  │ PHASE 3: BID & │ │
         │              │   ENRICHED     │  │   ENRICHED     │ │
         │              ├────────────────┤  ├────────────────┤ │
         │              │ 04_extract_bid_│  │ 05_extract_    │ │
         │              │    history.py  │  │  item_enriched │ │
         │              └────────┬───────┘  └────────┬───────┘ │
         │                       │                   │         │
         │                       ▼                   ▼         │
         │              ┌────────────────┐  ┌────────────────┐│
         │              │   Download     │  │   Download     ││
         │              │   from Kaggle  │  │   from Kaggle  ││
         │              │   + Append     │  │   + Append     ││
         │              │   + Upload     │  │   + Upload     ││
         │              └────────────────┘  └────────────────┘│
         ▼                                                     │
┌────────────────┐                                            │
│   Download     │                                            │
│   from Kaggle  │◄───────────────────────────────────────────┘
│   + Append     │
│   + Upload     │
└────────────────┘
```

## Setup

### 1. GitHub Secrets Configuration

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key

To get your Kaggle API credentials:
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json` containing your credentials

### 2. Local Setup (Optional)

To run the pipeline locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set Kaggle credentials as environment variables
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"

# Or create ~/.kaggle/kaggle.json:
mkdir -p ~/.kaggle
echo '{"username":"your-username","key":"your-api-key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Run the pipeline
python monthly_scraping_pipeline.py
```

## GitHub Action

**File:** `.github/workflows/monthly-scraping.yml`

**Schedule:** Runs automatically on the 1st of every month at 2:00 AM UTC

**Manual Trigger:** You can manually trigger the workflow from the Actions tab in GitHub

**Timeout:** 6 hours (for large scraping jobs)

## Pipeline Scripts

### Main Pipeline

**`monthly_scraping_pipeline.py`** - Orchestrates the entire workflow:
- Executes scrapers in sequence
- Manages data dependencies
- Handles Kaggle downloads/uploads
- Performs deduplication

### Individual Scrapers

1. **`scrapers/01_extract_auction_search.py`**
   - Searches for auctions in the last 30 days
   - Output: `auction_search_*.parquet` (temporary, not uploaded)

2. **`scrapers/02_extract_auction_details.py`**
   - Input: Auction IDs from step 1
   - Output: Auction details → Uploaded to `raw-maxsold-auction`

3. **`scrapers/03_extract_items_details.py`**
   - Input: Auction IDs from step 1
   - Output: Item details → Uploaded to `raw-maxsold-item`

4. **`scrapers/04_extract_bid_history.py`**
   - Input: Item IDs from step 3
   - Output: Bid history → Uploaded to `raw-maxsold-bid`

5. **`scrapers/05_extract_item_enriched_details.py`**
   - Input: Item IDs from step 3
   - Output: Enriched details → Uploaded to `raw-maxsold-item-enriched`

## Data Flow

```
01 (auction_search)
    ├─> 02 (auction_details) ─> Kaggle: raw-maxsold-auction
    └─> 03 (items_details) ──┬─> Kaggle: raw-maxsold-item
                             ├─> 04 (bid_history) ─> Kaggle: raw-maxsold-bid
                             └─> 05 (item_enriched) ─> Kaggle: raw-maxsold-item-enriched
```

## Deduplication Strategy

The pipeline automatically deduplicates data when appending to existing Kaggle datasets:

- **Auction Details**: Deduplicated by `auction_id`
- **Item Details**: Deduplicated by `id` + `auction_id`
- **Bid History**: Deduplicated by `auction_id` + `item_id` + `bid_number`
- **Item Enriched**: Deduplicated by `id` + `auction_id`

## Monitoring

### GitHub Actions UI

Check pipeline status at: `https://github.com/<username>/maxsold/actions`

### Artifacts

Pipeline artifacts (parquet files) are retained for 7 days and include:
- Temporary scraping outputs
- Raw data before Kaggle upload

### Summary

Each pipeline run creates a summary with:
- Execution timestamp
- Generated files and sizes
- Links to updated Kaggle datasets

## Troubleshooting

### Pipeline Fails

1. Check GitHub Actions logs for error details
2. Verify Kaggle credentials are set correctly
3. Ensure Kaggle datasets exist and you have write access
4. Check for API rate limits or network issues

### Kaggle Upload Issues

If upload fails:
```bash
# Test Kaggle authentication locally
kaggle datasets list --user your-username

# Verify dataset exists
kaggle datasets status pearcej/raw-maxsold-auction
```

### Re-running Failed Pipeline

The GitHub Action can be manually triggered:
1. Go to Actions tab
2. Select "Monthly MaxSold Scraping Pipeline"
3. Click "Run workflow"

## Customization

### Change Scraping Frequency

Edit `.github/workflows/monthly-scraping.yml`:

```yaml
schedule:
  # Run weekly on Mondays at 2 AM
  - cron: '0 2 * * 1'
  
  # Run bi-weekly (1st and 15th)
  - cron: '0 2 1,15 * *'
```

### Change Time Window

Edit `monthly_scraping_pipeline.py`:

```python
# Change days parameter
scrape_auction_search(output_path)  # Currently uses 30 days

# Modify in scrape_auction_search function:
cmd = [
    sys.executable,
    'scrapers/01_extract_auction_search.py',
    '--output', output_path,
    '--days', '60'  # Change to 60 days
]
```

### Add New Datasets

To upload to additional Kaggle datasets, add entries to `KAGGLE_DATASETS` dict in `monthly_scraping_pipeline.py`:

```python
KAGGLE_DATASETS = {
    'new_dataset': {
        'slug': 'username/dataset-name',
        'scraper_output': 'data/raw/new_dataset',
        'temp_file': 'new_dataset.parquet'
    }
}
```

## Performance

- **Average Runtime**: 2-4 hours (depending on data volume)
- **Data Volume**: ~30 days of auctions
- **Parallel Processing**: Used for bid history and enriched details
- **Rate Limiting**: Built-in delays to respect API limits

## Version History

- **v1.0** (2025-01-01): Initial pipeline with monthly scheduling
  - 5 scrapers orchestrated
  - Automatic Kaggle append & upload
  - GitHub Actions integration

## License

See main repository LICENSE file.

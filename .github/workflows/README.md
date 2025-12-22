# GitHub Actions Workflow - Data Scraping Pipeline

## Overview

The GitHub Actions workflow (`.github/workflows/run-scrapers.yml`) runs the complete data scraping pipeline to fetch MaxSold auction data.

## Workflow Execution Order

```
1. Auction Search (01_extract_auction_search.py)
   ↓ Produces: auction_search_YYYYMMDD.parquet
   
2. Auction Details (02_extract_auction_details.py)
   ↓ Input: auction_search file
   ↓ Produces: auction_details_YYYYMMDD.parquet
   
3. Item Details (03_extract_items_details.py)
   ↓ Input: auction_search file
   ↓ Produces: items_details_YYYYMMDD.parquet
   
4. Bid History (04_extract_bid_history.py)
   ↓ Input: items_details file
   ↓ Produces: bid_history_YYYYMMDD.parquet
   
5. Item Enriched Details (05_extract_item_enriched_details.py)
   ↓ Input: items_details file
   ↓ Produces: item_enriched_details_YYYYMMDD.parquet
```

## What Was Updated

### ✅ Added Item Enriched Details Scraper
- Now runs the 5th scraper to get AI-generated item details
- Uses 5 parallel workers for faster processing

### ✅ Added Parallel Processing
- Bid History: `--workers 5`
- Item Enriched: `--workers 5`
- Speeds up scraping for large datasets

### ✅ Improved Error Handling
- Added `|| true` to git add commands to prevent failures if no files exist
- Added `if: always()` to upload artifacts step to ensure outputs are saved even on partial failures

### ✅ Complete Data Pipeline
All 5 scrapers now run in sequence:
1. **Auction Search** - Find auctions by location/date
2. **Auction Details** - Get metadata for each auction
3. **Item Details** - Get all items/lots in each auction
4. **Bid History** - Get bid timeline for each item
5. **Item Enriched** - Get AI-generated categories, brands, attributes

## How to Run

### Manual Trigger (Recommended)

1. Go to your GitHub repository
2. Click **Actions** tab
3. Select **Run Scrapers** workflow
4. Click **Run workflow** button
5. Choose the branch (default: `feature/scraping_pipeline_001`)
6. Click **Run workflow**

### Automatic Trigger (Optional)

To run on a schedule, add this to the workflow:

```yaml
on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 2 * * 0'  # Run every Sunday at 2 AM UTC
```

## Outputs

### 1. Committed Files

The workflow automatically commits the generated parquet files to the repository:

```
data/
├── auction_search/
│   └── auction_search_20251222.parquet
├── auction_details/
│   └── auction_details_20251222.parquet
├── items_details/
│   └── items_details_20251222.parquet
├── bid_history/
│   └── bid_history_20251222.parquet
└── item_enriched_details/
    └── item_enriched_details_20251222.parquet
```

### 2. Artifacts

The workflow also uploads the files as GitHub Actions artifacts:
- **Name**: `scraper-outputs-YYYYMMDD`
- **Retention**: 30 days
- **Download**: From the workflow run page

## Configuration

### Adjust Number of Workers

For faster/slower scraping, modify the `--workers` parameter:

```yaml
# Faster (more API requests)
--workers 10

# Slower (fewer API requests)
--workers 2
```

### Change Data Retention

Modify artifact retention period:

```yaml
retention-days: 30  # Keep for 30 days (default)
retention-days: 90  # Keep for 90 days
```

### Change Output Paths

All output paths follow the pattern:
```
data/{scraper_type}/{scraper_type}_${TIMESTAMP}.parquet
```

To change, modify the `OUTPUT_FILE` variables in each step.

## Monitoring

### View Progress

1. Go to **Actions** tab in GitHub
2. Click on the running workflow
3. Expand each step to see logs

### Check Outputs

Each scraper step outputs:
- Number of records fetched
- Processing time
- Any errors encountered

### Download Artifacts

1. Go to completed workflow run
2. Scroll to **Artifacts** section
3. Click on `scraper-outputs-YYYYMMDD` to download

## Troubleshooting

### Workflow Fails at Auction Search

**Cause**: API rate limiting or network issues

**Solution**:
- Wait a few minutes and re-run
- Check if MaxSold API is accessible

### Workflow Fails at Later Steps

**Cause**: Empty input file from previous step

**Solution**:
- Check if previous scraper produced valid output
- Verify input file paths are correct

### Git Commit Fails

**Cause**: No new files or permission issues

**Solution**:
- This is normal if no new data was scraped
- The `|| true` prevents workflow failure
- Check if files actually changed

### Parallel Processing Too Slow/Fast

**Cause**: Wrong number of workers

**Solution**:
- Increase `--workers` for faster processing
- Decrease `--workers` to avoid rate limiting
- Recommended: 5-10 workers

## Best Practices

### 1. Run Weekly or Bi-Weekly
- MaxSold auctions close over time
- Weekly runs capture recent auction activity

### 2. Monitor Disk Usage
- Parquet files accumulate over time
- Consider cleaning old files or using external storage

### 3. Check for Duplicates
- Each run creates new timestamped files
- Merge/deduplicate in downstream processing

### 4. Error Notifications
Add Slack/email notifications for failures:

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Data Flow

```
GitHub Actions Runner
        │
        ├─ 1. Fetch auction search results
        │      (Location: Toronto area, Days: 180)
        │      ↓
        │  auction_search_20251222.parquet
        │      ↓
        ├─ 2. For each auction → Get details
        │      ↓
        │  auction_details_20251222.parquet
        │      ↓
        ├─ 3. For each auction → Get all items
        │      ↓
        │  items_details_20251222.parquet
        │      ↓
        ├─ 4. For each item → Get bid history (parallel)
        │      ↓
        │  bid_history_20251222.parquet
        │      ↓
        └─ 5. For each item → Get AI enriched data (parallel)
               ↓
           item_enriched_details_20251222.parquet
               ↓
        All files committed to repository
        All files uploaded as artifacts
```

## Estimated Runtime

Based on typical data volumes:

| Scraper | Items Processed | Est. Time |
|---------|----------------|-----------|
| Auction Search | ~500 auctions | 2-5 min |
| Auction Details | ~500 auctions | 5-10 min |
| Item Details | ~500 auctions | 5-10 min |
| Bid History | ~75,000 items | 30-60 min |
| Item Enriched | ~75,000 items | 30-60 min |

**Total**: ~1.5 - 2.5 hours for a full run

## Cost Considerations

### GitHub Actions
- Free tier: 2,000 minutes/month for public repos
- Each full run: ~120-150 minutes
- Can run ~13 times per month on free tier

### API Rate Limits
- MaxSold API has rate limits
- Parallel workers increase speed but use more quota
- Adjust `--workers` based on your needs

## Next Steps

After scraping completes:

1. **Data Processing**: Run feature engineering scripts
2. **Model Training**: Use the data to train ML models
3. **Validation**: Check data quality and completeness
4. **Storage**: Consider moving old data to external storage

---

*Workflow file: `.github/workflows/run-scrapers.yml`*
*Updated: December 22, 2025*

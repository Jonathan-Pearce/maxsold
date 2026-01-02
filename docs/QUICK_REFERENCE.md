# Monthly Scraping Pipeline - Quick Reference

## 🚀 Quick Start

### GitHub Action (Automated)
The pipeline runs automatically **on the 1st of every month at 2:00 AM UTC**.

**Required GitHub Secrets:**
- `KAGGLE_USERNAME` - Your Kaggle username
- `KAGGLE_KEY` - Your Kaggle API key

### Manual Trigger
1. Go to GitHub Actions tab
2. Select "Monthly MaxSold Scraping Pipeline"
3. Click "Run workflow"

### Local Execution
```bash
# Set credentials
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"

# Run pipeline
python monthly_scraping_pipeline.py
```

## 📊 What Gets Updated

Every month, these Kaggle datasets are updated with new data:

| Dataset | URL | Description |
|---------|-----|-------------|
| raw-maxsold-auction | https://www.kaggle.com/datasets/pearcej/raw-maxsold-auction | Auction details |
| raw-maxsold-item | https://www.kaggle.com/datasets/pearcej/raw-maxsold-item | Item details |
| raw-maxsold-bid | https://www.kaggle.com/datasets/pearcej/raw-maxsold-bid | Bid history |
| raw-maxsold-item-enriched | https://www.kaggle.com/datasets/pearcej/raw-maxsold-item-enriched | AI-enriched item details |

## 🔄 Pipeline Flow

```
1. Scrape auction search (30 days) → auction IDs
   ↓
2. Scrape auction details (from auction IDs)
3. Scrape item details (from auction IDs)
   ↓
4. Scrape bid history (from item IDs)
5. Scrape enriched details (from item IDs)
   ↓
6. Download existing Kaggle data
7. Append new data (with deduplication)
8. Upload back to Kaggle
```

## ⚙️ Configuration

### Change Schedule

Edit [.github/workflows/monthly-scraping.yml](.github/workflows/monthly-scraping.yml):

```yaml
schedule:
  # Weekly on Mondays
  - cron: '0 2 * * 1'
  
  # Bi-weekly (1st and 15th)
  - cron: '0 2 1,15 * *'
```

### Change Time Window

Edit [monthly_scraping_pipeline.py](monthly_scraping_pipeline.py) line ~96:

```python
'--days', '30'  # Change to '60' for 60 days
```

## 🔍 Monitoring

### Check Status
- GitHub Actions: https://github.com/Jonathan-Pearce/maxsold/actions
- View logs, artifacts, and summaries

### Artifacts
Pipeline outputs are saved as GitHub artifacts for 7 days.

## 🛠️ Troubleshooting

### Pipeline Fails
1. Check GitHub Actions logs
2. Verify Kaggle credentials
3. Ensure datasets exist on Kaggle
4. Check API rate limits

### Test Locally
```bash
# Verify setup
bash verify_pipeline_setup.sh

# Run pipeline with debugging
python -u monthly_scraping_pipeline.py 2>&1 | tee pipeline.log
```

## 📁 Files Created

### Main Pipeline
- `monthly_scraping_pipeline.py` - Main orchestration script

### GitHub Action
- `.github/workflows/monthly-scraping.yml` - Monthly workflow

### Documentation
- `README_MONTHLY_PIPELINE.md` - Full documentation
- `QUICK_REFERENCE.md` - This file
- `verify_pipeline_setup.sh` - Setup verification script

### Scrapers (Existing)
All in `scrapers/` directory, now integrated into the pipeline.

## 💡 Tips

1. **First Run**: Manually trigger the workflow to test before waiting for the scheduled run
2. **Monitoring**: Enable GitHub Actions notifications in your repository settings
3. **Data Volume**: The pipeline typically takes 2-4 hours depending on data volume
4. **Rate Limits**: Built-in delays respect API rate limits
5. **Deduplication**: Automatic - old data is preserved, only new rows added

## 🔒 Security

- Kaggle credentials stored as GitHub Secrets (encrypted)
- Never commit kaggle.json to repository
- Credentials only accessible during GitHub Action runs

## 📞 Support

See full documentation: [README_MONTHLY_PIPELINE.md](README_MONTHLY_PIPELINE.md)

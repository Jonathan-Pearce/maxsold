# 📋 Setup Checklist - Monthly Scraping Pipeline

Use this checklist to ensure the pipeline is properly configured.

## ✅ Pre-Setup Verification

- [ ] Repository is at: https://github.com/Jonathan-Pearce/maxsold
- [ ] You have admin access to the repository
- [ ] You have a Kaggle account
- [ ] Kaggle datasets exist:
  - [ ] https://www.kaggle.com/datasets/pearcej/raw-maxsold-auction
  - [ ] https://www.kaggle.com/datasets/pearcej/raw-maxsold-item
  - [ ] https://www.kaggle.com/datasets/pearcej/raw-maxsold-bid
  - [ ] https://www.kaggle.com/datasets/pearcej/raw-maxsold-item-enriched

## 🔑 Kaggle Credentials Setup

### Step 1: Get Kaggle API Credentials
- [ ] Go to https://www.kaggle.com/settings/account
- [ ] Scroll to "API" section
- [ ] Click "Create New Token"
- [ ] Download `kaggle.json` file
- [ ] Open the file and note your username and key

### Step 2: Add to GitHub Secrets
- [ ] Go to https://github.com/Jonathan-Pearce/maxsold/settings/secrets/actions
- [ ] Click "New repository secret"
- [ ] Add `KAGGLE_USERNAME`:
  - Name: `KAGGLE_USERNAME`
  - Value: [your Kaggle username from kaggle.json]
- [ ] Click "Add secret"
- [ ] Click "New repository secret" again
- [ ] Add `KAGGLE_KEY`:
  - Name: `KAGGLE_KEY`
  - Value: [your API key from kaggle.json]
- [ ] Click "Add secret"
- [ ] Verify both secrets appear in the list

## 🧪 Testing

### Step 3: Verify Files Exist (Optional Local Check)
```bash
# Clone repository if not already
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold

# Run verification script
bash verify_pipeline_setup.sh
```

### Step 4: Manual Test Run
- [ ] Go to https://github.com/Jonathan-Pearce/maxsold/actions
- [ ] Click on "Monthly MaxSold Scraping Pipeline" in the left sidebar
- [ ] Click "Run workflow" button (top right)
- [ ] Select branch: `feature/data_pipeline` (or your branch)
- [ ] Click green "Run workflow" button
- [ ] Wait for workflow to start (refresh page after a few seconds)
- [ ] Click on the running workflow to see logs
- [ ] Monitor progress:
  - [ ] Checkout repository - should complete quickly
  - [ ] Set up Python - should complete in ~30 seconds
  - [ ] Install dependencies - should complete in ~1-2 minutes
  - [ ] Configure Kaggle credentials - should complete instantly
  - [ ] Run Monthly Scraping Pipeline - takes 2-4 hours
    - Watch for "PHASE 1: AUCTION SEARCH"
    - Watch for "PHASE 2: AUCTION & ITEM DETAILS"
    - Watch for "PHASE 3: BID HISTORY & ENRICHED DETAILS"
    - Watch for "PHASE 4: KAGGLE UPLOAD"
  - [ ] Workflow completes successfully ✅

### Step 5: Verify Results
After the workflow completes:

- [ ] Check GitHub Actions summary for execution details
- [ ] Download artifacts (optional - available for 7 days)
- [ ] Verify Kaggle datasets were updated:
  - [ ] Go to each Kaggle dataset URL
  - [ ] Check "Last Updated" timestamp
  - [ ] Verify new rows were added (check row count)

## 📅 Ongoing Monitoring

### After First Successful Run
- [ ] Mark calendar for 1st of next month
- [ ] Enable GitHub Actions email notifications (optional):
  - [ ] Go to https://github.com/settings/notifications
  - [ ] Under "Actions" section
  - [ ] Enable notifications for workflow runs

### Monthly Checks (1st of each month)
- [ ] Check GitHub Actions around 2:00 AM UTC
- [ ] Verify workflow started automatically
- [ ] Monitor for completion (2-4 hours)
- [ ] Check for any error notifications
- [ ] Verify Kaggle datasets were updated

## ⚠️ Troubleshooting

If the workflow fails:

1. **Check Logs**
   - [ ] Go to Actions tab
   - [ ] Click on failed workflow
   - [ ] Expand failed step
   - [ ] Read error message

2. **Common Issues**
   - [ ] Kaggle credentials incorrect → Re-add GitHub Secrets
   - [ ] Rate limiting → Wait and retry manually
   - [ ] Network timeout → Increase timeout in workflow (currently 6 hours)
   - [ ] Kaggle dataset permission → Verify you own the datasets

3. **Get Help**
   - [ ] Check full documentation: `README_MONTHLY_PIPELINE.md`
   - [ ] Review implementation summary: `PIPELINE_IMPLEMENTATION_SUMMARY.md`

## 🎉 Success!

Once you've completed this checklist:
- ✅ Pipeline runs automatically monthly
- ✅ Kaggle datasets stay up-to-date
- ✅ Historical data is preserved
- ✅ No manual intervention needed

---

## 📋 Quick Reference

**GitHub Actions URL:**
https://github.com/Jonathan-Pearce/maxsold/actions

**GitHub Secrets URL:**
https://github.com/Jonathan-Pearce/maxsold/settings/secrets/actions

**Kaggle Datasets:**
- https://www.kaggle.com/datasets/pearcej/raw-maxsold-auction
- https://www.kaggle.com/datasets/pearcej/raw-maxsold-item
- https://www.kaggle.com/datasets/pearcej/raw-maxsold-bid
- https://www.kaggle.com/datasets/pearcej/raw-maxsold-item-enriched

**Schedule:** 1st of every month at 2:00 AM UTC

**Typical Duration:** 2-4 hours

**Data Window:** Last 30 days

---

**Last Updated:** December 23, 2025

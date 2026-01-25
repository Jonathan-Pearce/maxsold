# GitHub Actions Migration Guide

This document provides guidance for updating GitHub Actions workflows to use the new Cookiecutter Data Science structure.

## Current Workflows

The repository has the following GitHub Actions workflows that reference old paths:

1. `.github/workflows/monthly-scraping.yml`
2. `.github/workflows/monthly-feature-engineering.yml`
3. `.github/workflows/run_auction_backfill.yml`

## Required Updates

### 1. Monthly Scraping Workflow

**File:** `.github/workflows/monthly-scraping.yml`

**Current path (line 50):**
```yaml
python scrapers/monthly_scraping_pipeline.py
```

**New path:**
```yaml
python src/data/monthly_scraping_pipeline.py
```

**Artifact paths (lines 59-60):**
```yaml
# Current
path: |
  data/temp/*.parquet
  data/raw/**/*.parquet

# Should remain the same (data/ structure is already updated)
```

### 2. Feature Engineering Workflow

**File:** `.github/workflows/monthly-feature-engineering.yml`

**Current path (line 50):**
```yaml
python feature_engineering/run_pipeline.py
```

**New path:**
```yaml
python src/features/run_pipeline.py
```

**Artifact paths (lines 59-61):**
```yaml
# Current
path: |
  data/engineered/**/*.parquet
  data/final/*.parquet
  data/models/**/*

# Update to new structure
path: |
  data/interim/**/*.parquet
  data/processed/*.parquet
  models/**/*
```

**Summary paths (lines 74-82):**
```yaml
# Current
if [ -d "data/engineered" ]; then
  find data/engineered -name "*.parquet"
fi
if [ -d "data/final" ]; then
  find data/final -name "*.parquet"
fi

# Update to new structure
if [ -d "data/interim" ]; then
  find data/interim -name "*.parquet"
fi
if [ -d "data/processed" ]; then
  find data/processed -name "*.parquet"
fi
```

### 3. Auction Backfill Workflow

**File:** `.github/workflows/run_auction_backfill.yml`

Check this file for any references to:
- `scrapers/` → should be `src/data/`
- `backfilling/` → should be `src/data/`
- `data/engineered/` → should be `data/interim/`
- `data/final/` → should be `data/processed/`

## Python Path Configuration

If scripts in `src/` import from each other, you may need to add the project root to PYTHONPATH:

```yaml
- name: Configure Python Path
  run: |
    echo "PYTHONPATH=$PYTHONPATH:$PWD" >> $GITHUB_ENV

- name: Run Pipeline
  run: |
    python src/data/monthly_scraping_pipeline.py
```

## Backward Compatibility

The old directories (`scrapers/`, `feature_engineering/`, etc.) are currently maintained for backward compatibility. This means:

1. **Workflows continue to work** with old paths during transition
2. **No breaking changes** to existing automation
3. **Gradual migration** is possible

## Migration Strategy

### Option 1: Immediate Migration (Recommended)
1. Update all workflow files to use new paths
2. Test workflows manually using `workflow_dispatch`
3. Remove old directories after verification

### Option 2: Gradual Migration
1. Keep old directories and workflows as-is
2. Create new workflow files for new structure
3. Run both in parallel during transition
4. Switch over after validation
5. Remove old workflows and directories

### Option 3: Symbolic Links (Quick Fix)
1. Create symbolic links from old to new locations
2. Keep workflows unchanged
3. Scripts work with either path

```bash
# Example symbolic links
ln -s src/data scrapers_new
ln -s src/features feature_engineering_new
```

## Testing Workflows

Test the updated workflows manually before relying on scheduled runs:

1. Go to Actions tab in GitHub
2. Select the workflow
3. Click "Run workflow"
4. Select branch and click "Run workflow"
5. Monitor the run and check logs

## Implementation Checklist

- [ ] Update `.github/workflows/monthly-scraping.yml`
  - [ ] Update script path to `src/data/`
  - [ ] Test workflow manually
- [ ] Update `.github/workflows/monthly-feature-engineering.yml`
  - [ ] Update script path to `src/features/`
  - [ ] Update artifact paths to new data structure
  - [ ] Update summary paths
  - [ ] Test workflow manually
- [ ] Update `.github/workflows/run_auction_backfill.yml` (if exists)
  - [ ] Update script paths
  - [ ] Test workflow manually
- [ ] Add PYTHONPATH configuration if needed
- [ ] Test all workflows end-to-end
- [ ] Remove old directories after verification
- [ ] Update workflow documentation

## Notes

- The workflows are scheduled to run monthly, so test carefully
- Consider running test workflows on a schedule before switching production
- Keep old directories until all workflows are verified
- Update any documentation that references workflow behavior

## Related Documentation

- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Full reorganization details
- [STRUCTURE_GUIDE.md](../STRUCTURE_GUIDE.md) - Visual reference guide
- [src/README.md](../src/README.md) - Source code overview

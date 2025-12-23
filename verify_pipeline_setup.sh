#!/bin/bash
# Test script to verify the pipeline setup

echo "=================================================="
echo "Monthly Scraping Pipeline - Setup Verification"
echo "=================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -n "Checking Python version... "
python_version=$(python --version 2>&1 | awk '{print $2}')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
    echo -e "${GREEN}✓${NC} Python $python_version"
else
    echo -e "${RED}✗${NC} Python $python_version (requires >= 3.8)"
    exit 1
fi

# Check if requirements.txt exists
echo -n "Checking requirements.txt... "
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} Not found"
    exit 1
fi

# Check if main pipeline file exists
echo -n "Checking monthly_scraping_pipeline.py... "
if [ -f "monthly_scraping_pipeline.py" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} Not found"
    exit 1
fi

# Check if GitHub Action exists
echo -n "Checking GitHub Action workflow... "
if [ -f ".github/workflows/monthly-scraping.yml" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} Not found"
    exit 1
fi

# Check all scraper files
echo ""
echo "Checking scraper files:"
scrapers=(
    "scrapers/01_extract_auction_search.py"
    "scrapers/02_extract_auction_details.py"
    "scrapers/03_extract_items_details.py"
    "scrapers/04_extract_bid_history.py"
    "scrapers/05_extract_item_enriched_details.py"
)

for scraper in "${scrapers[@]}"; do
    echo -n "  $scraper... "
    if [ -f "$scraper" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        exit 1
    fi
done

# Check Kaggle credentials
echo ""
echo "Checking Kaggle credentials:"
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo -e "  Environment variables: ${GREEN}✓${NC} Set"
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "  ~/.kaggle/kaggle.json: ${GREEN}✓${NC} Found"
else
    echo -e "  ${YELLOW}⚠${NC} Kaggle credentials not configured"
    echo "  Set environment variables or create ~/.kaggle/kaggle.json"
    echo "  The pipeline will fail without credentials."
fi

# Check required Python packages
echo ""
echo "Checking key Python packages:"
packages=("pandas" "requests" "kaggle")

for package in "${packages[@]}"; do
    echo -n "  $package... "
    if python -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC} Not installed"
        echo "    Run: pip install -r requirements.txt"
    fi
done

# Syntax check on main pipeline
echo ""
echo -n "Syntax checking monthly_scraping_pipeline.py... "
if python -m py_compile monthly_scraping_pipeline.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} Syntax errors found"
    exit 1
fi

# Check directory structure
echo ""
echo "Checking directory structure:"
dirs=("data" "scrapers" "utils" ".github/workflows")

for dir in "${dirs[@]}"; do
    echo -n "  $dir/... "
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo -e "${GREEN}✓ All checks passed!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Configure GitHub Secrets (KAGGLE_USERNAME, KAGGLE_KEY)"
echo "2. Test locally: python monthly_scraping_pipeline.py"
echo "3. The pipeline will run automatically on the 1st of each month"
echo "4. Manual trigger: GitHub Actions → Monthly MaxSold Scraping Pipeline → Run workflow"
echo ""

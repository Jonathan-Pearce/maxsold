# Maxsold Data Project

A comprehensive data project for scraping, processing, and analyzing MaxSold auction data with machine learning capabilities.

**📁 This project follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) structure. See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.**

## 🐳 Docker Setup (Recommended)

The easiest way to get started is using Docker with the provided helper scripts.

**Quick Start (Linux/Mac):**
```bash
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold
chmod +x docker-run.sh
./docker-run.sh build
./docker-run.sh shell
```

**Quick Start (Windows):**
```cmd
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold
docker-run.bat build
docker-run.bat shell
```

**Documentation:**
- 📖 **[DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)** - Quick reference card
- 📘 **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Complete setup guide with troubleshooting

## 📋 Requirements

- Python 3.12+
- Dependencies listed in `requirements.txt`

## 🚀 Installation (Without Docker)

```bash
# Clone the repository
git clone https://github.com/Jonathan-Pearce/maxsold.git
cd maxsold

# Install dependencies
pip install -r requirements.txt
```

## 📚 Documentation

- 📁 **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project organization and Cookiecutter Data Science structure
- 📖 **[DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)** - Docker quick reference card
- 📘 **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Complete Docker setup guide
- 🤖 **[src/models/README.md](src/models/README.md)** - Machine learning pipeline documentation
- 📊 **[src/README.md](src/README.md)** - Source code overview

## 🔗 MaxSold Resources

https://support.maxsold.com/hc/en-us/articles/203144054-How-do-bid-increments-work
https://support.maxsold.com/hc/en-us/articles/203144064-What-does-soft-close-mean

## 📂 Project Structure

This project follows the Cookiecutter Data Science structure:

```
├── data/               <- Data directory (raw, interim, processed, external)
├── models/             <- Trained models
├── notebooks/          <- Jupyter notebooks for exploration
├── references/         <- Reference materials and data dictionaries
├── reports/            <- Generated analysis and figures
├── src/                <- Source code
│   ├── data/          <- Data collection scripts (scrapers)
│   ├── features/      <- Feature engineering
│   ├── models/        <- Model training and prediction
│   └── visualization/ <- Visualization scripts
└── docs/              <- Documentation
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete details and migration guide.

## Example terminal commands

```bash
# Data collection
python src/data/01_extract_auction_search.py
python src/data/02_extract_auction_details.py
python src/data/monthly_scraping_pipeline.py

# Feature engineering
python src/features/auction_features.py
python src/features/final_dataset_builder.py

# Model training
python src/models/scripts/train_model_minimal.py
```


## Design Plan


TBL_SEARCH_LOCATIONS
- columns: SEARCH_LOCATION_ID, LOCATION_STRING
- e.g. 1, canada/ontario/toronto


TBL_AUCTIONS
- columns: ID, SEARCH_LOCATION_ID, AUCTION_ID, AUCTION_METADATA (start time, end time, location, postal code, number of lots, etc.)
- e.g. 1, 1, 103482

TBL_LISTINGS
- columns: ID, AUCTION_ID, LISTING_ID, LISTING_METADATA (Title, total bids)
- 

TBL_BIDS
- columns: ID, LISTING_ID, BID_ID, BID_DATE, BID_AMOUNT

Other Considerations
- Currency



## URLs

- https://api.maxsold.com/places/address?address=toronto,+ontario,+canada
- https://api.maxsold.com/sales/search?lat=43.653226&lng=-79.3831843&radiusMetres=201168&country=canada&pageNumber=0&limit=24&saleState=closed&days=90&total=true


## Old Auctions

- https://maxsold.com/auction/100070/bidgallery?offset=144
- link works as of Thursday November 20 2025

- https://maxsold.com/auction/99395/bidgallery
- link works as of Thursday November 20 2025



## TODOs

manually backfilling data with iterative auction ID and location comparison 

# Notes

There is an API for item listings!
https://maxsold.maxsold.com/msapi/auctions/items?auctionid=103293&limit=72&offset=72&sort=bid_count&hideclosed=0&sort_direction=descending

https://maxsold.maxsold.com/msapi/auctions/items?auctionid=103293

https://maxsold.maxsold.com/msapi/auctions/items?auctionid=103293&itemid=7433915

https://api.maxsold.com/listings/am/7433915/enriched

https://maxsold.maxsold.com/msapi/auctions/items?auctionid=103293&itemid=7433850


## Design Plan

- clean up code
- - exports should be parquet (make sure data types are roughly correct)
- - no code required to export jsons
- - code should be saved to /data/bid list, auctions, items, bids/{date}_{filename}
- - add URLs to data save
- - Schedule code files (3 or 4 of them) to run via Github Actions every 3 months (?)
- - create aggregation script(s) and figure out how to publish to Hugging Face for data hosting

- Item price model/EDA
- - Average cost per item
- - geography, day of week, size of auction vs average cost per item
- - How to factor in views
- - NLP feature engineering

- Bidding modelling
- - RL agent to simulating bidding
- - 
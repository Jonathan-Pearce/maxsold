# Maxsold Data Projecct

Example terminal commands

```
python3 /workspaces/maxsold/scrapers/maxsold_bid_history.py "https://maxsold.com/listing/7436058/" > bids.json
```

```
python3 /workspaces/maxsold/scrapers/maxsold_auction_metadata.py "https://maxsold.com/auction/103482/date-times" ./auction_103482.json
```

```
python3 /workspaces/maxsold/scrapers/extract_bidgallery_links.py "https://maxsold.com/auction/103482/bidgallery" ./auction_103482_links.json
```

```
python3 /workspaces/maxsold/scrapers/extract_toronto_past_auctions.py "https://maxsold.com/canada/ontario/toronto/past" ./toronto_past_auctions.json
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
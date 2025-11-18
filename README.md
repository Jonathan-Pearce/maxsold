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
-Currency
-
import scrapy


class MaxsoldScraperSpider(scrapy.Spider):
    name = "maxsold_scraper"
    allowed_domains = ["maxsold.com"]
    start_urls = ["https://maxsold.com/auction/90505/bidgallery/item/6089314?offset=0"]

    def parse(self, response):
        pass

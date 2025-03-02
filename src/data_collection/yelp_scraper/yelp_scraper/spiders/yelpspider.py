# import scrapy
# import random
# import json

# class YelpSpider(scrapy.Spider):
    
    # start_urls = ['https://www.yelp.com/biz/home-thai-restaurant-sydney-3']
    
    
    # def start_requests(self):
    #     user_agents = [
    #         'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    #         'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    #         # Add more User-Agent strings
    #     ]
        
    #     headers = {
    #         'User-Agent': random.choice(user_agents),
    #     }
        
    #     yield scrapy.Request(url="https://www.yelp.com/biz/home-thai-restaurant-sydney-3", headers=headers)
    
    # def start_requests(self):
    #     user_agents = [
    #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36",
    #         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    #         "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    #     ]

    #     headers = {
    #         "User-Agent": random.choice(user_agents),
    #         "Accept-Language": "en-US,en;q=0.9",
    #         "Referer": "https://www.google.com/",
    #     }

    #     yield scrapy.Request(
    #         url="https://www.yelp.com/biz/home-thai-restaurant-sydney-3",
    #         headers=headers,
    #         callback=self.parse
    # )
    
#------------------------------------------old------------------------------------------
    # name = 'yelp'
    # start_urls = ['https://www.yelp.com/biz/home-thai-restaurant-sydney-3']
    
    # def parse(self, response):
    #     for comments in response.css('div.y-css-mhg9c5'):
    #         yield{
    #             "text": comments.css('span.raw__09f24__T4Ezm').get(),
    #             "rating": comments.css('div.y-css-dnttlc::attr(aria-label)').get(),
    #             "date": comments.css('span.y-css-1d8mpv1::text').get(),
    #             "location": comments.css('span.y-css-1wfz87z::text').get(),
    #         }
            
        
    #     next_page = response.css('a.next-link.navigation-button__09f24__m9qRz.y-css-1kw15cs').attrib['href']
    #     if next_page is not None:
    #         yield response.follow(next_page, callback=self.parse)


import scrapy
import logging
import os

# Setup logging
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../logs"))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "scraper.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class YelpSpider(scrapy.Spider):
    name = 'yelp'
    start_urls = ['https://www.yelp.com/biz/home-thai-restaurant-sydney-3']

    def parse(self, response):
        logging.info("Scraping started for URL: %s", response.url)

        for comments in response.css('div.y-css-mhg9c5'):
            data = {
                "text": comments.css('span.raw__09f24__T4Ezm').get(),
                "rating": comments.css('div.y-css-dnttlc::attr(aria-label)').get(),
                "date": comments.css('span.y-css-1d8mpv1::text').get(),
                "location": comments.css('span.y-css-1wfz87z::text').get(),
            }
            logging.info(f"Scraped Data: {data}")
            yield data

        next_page = response.css('a.next-link.navigation-button__09f24__m9qRz.y-css-1kw15cs').attrib.get('href')
        if next_page:
            logging.info(f"Following next page: {next_page}")
            yield response.follow(next_page, callback=self.parse)

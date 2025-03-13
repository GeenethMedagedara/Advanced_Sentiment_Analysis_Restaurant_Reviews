import scrapy
import logging
import os

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

from pathlib import Path
import scrapy as sp


class BiliScraperForTest(sp.Spider):
    name = "bili_scraper_test"

    start_urls = [
        "https://www.bilibili.com/",
        "https://www.bilibili.com/video/BV1Ga4y1u7fh/",
    ]

    allowed_domains = [
        "bilibili.com"
    ]

    def parse(self, response):
        url_name = response.url.split("/")[-2]
        file_name = f"bili-scraper-{url_name}.html"
        Path(file_name).write_bytes(response.body)
        self.log(f"{file_name} is saved successfully!")

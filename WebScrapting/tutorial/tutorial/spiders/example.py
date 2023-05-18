import scrapy


class MySpider(scrapy.Spider):
    name = "example.com"
    allowed_domains = ["example.com"]
    start_urls = [
        "http://www.example.com",
        "https://www.bilibili.com"
    ]

    def parse(self, response):
        self.logger.info("A response from %s just arrived!", response.url)

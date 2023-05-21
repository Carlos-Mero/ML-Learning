from pathlib import Path
import scrapy as sp
import re


def extract_integras(string):
    pattern = r'\d+'  # 匹配一个或多个数字
    integers = re.findall(pattern, string)
    return [int(i) for i in integers]


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


class BiliScraperWriteCSV(sp.Spider):
    name = "bili_csv"

    start_urls = [
        "https://www.bilibili.com/video/BV1Ga4y1u7fh/",
    ]

    allowed_domains = [
        "bilibili.com"
    ]

    def parse(self, response):
        discription = response.xpath("/html/head/meta[11]/@content").get()
        data_list = extract_integras(discription)
        yield {
            "视频播放量": data_list[0],
            "弹幕量": data_list[1],
            "点赞数": data_list[2],
            "投硬币枚数": data_list[3],
            "收藏人数": data_list[4],
            "转发人数": data_list[5],
        }

# Notes on Web Scrabing

> Remember that if there is a ready-made data sets that can be found,
> just use it and never tend to web scrabing.

### Tools for web scrabing

Here is a set of tools I choosed to use for scrabing the web.
Their repositories, websites and documentations are listed below

* **Scrapy**

[Website](https://scrapy.org), [documentation](https://docs.scrapy.org).

### Notes on Scrapy

#### basic usage

**Creating a project**

We can use scrapy through the command below to create a empty project.

```shell
scrapy startproject PROJECT_NAME
```

A spider is class that we define and Scrapy uses to scrape information from a website.
They must be a subclass of **Spider**, and define the initial requests to make.

There are several attributes with special use here.

* **name**: identifies the Spider. Within a project it must be unique.
* **start_request()**: this method must return an iterable of Requests(whether by returning
	a list of requests or writing a generatoe function by `yield` or etc.) which Spider
	will begin to crawl from.
	subsequent requests will be generated successively from the initial ones.
* **parse()**: this method will be called to handle the response downloaded for each of the
	requests made. The response parameter is an instance of `TextResponse` that holds the page
	content and has further helpful methods to handles it.

	The parse() method usually parses the response, extracting the scraped data as dicts
	and also finding new URLs to follow and creating new requests (Request) from them.

To sum up, the function **start_request()** sends the request to the given urls and returns
an iterable of requests, and they're passed and to be handled by the method **parse()**.

> Instead of implimenting the start_request() method, we can just define a `start_urls`
> class attribute with a list of URLs. This list will be used by default to create the init
> requests for the spider.

**Running The Spider** requests us to go back to the root level directory of this project and
run the following commands

```shell
scrapy crawl NAME_OF_THE_SPIDER
```

#### Extracting Data

Scrapy contains a module to help us extract data, which is called `scrapy shell`. Simply run

```shell
scrapy shell 'TARGET_URL'
```

Then we can interactively selecting elements using CSS with the response object. Say

```shell
>>> response.css("title")
[<Selector query='descendant-or-self::title' data='<title>Quotes to Scrape</title>'>]

>>> response.css("title::text").get()
'Quotes to Scrape'
```

To find the right CSS selectors to use, we may need to refer to the brower's developer tools to
inspect the HTML. Remember to call `view(response)` to open the response page in the browser.

**XPath**: XPath expressions are more powerful over the CSS selector we used above.
It'll be useful to learn some of this, which we'll cover later.

#### Exporting Results

A Scrapy spider typically generates many dictionaries containing the data extracted from the page.
To do that, we use the `yield` keyword in the callback, just in the method `parse()`.

If we want to export our results in a file, simply use the option `-o` by

```shell
scrapy crawl NAME_OF_THE_SPIDER -o result.csv
```

Here we have two different types of generation. Using the `-o` option means that we want to write
a new file or open an existing one and append new content to it.
On the other hand, we can also use `-O` option, which states that we want overwrite any content
in that file.

Many export formats are supported by Scrapy, the most frequently used ones may be `.csv` or `.json`.

#### Following Links

There're two ways to scrape links from a HTML source, say

```python
response.css("li.next a::attr(href)").get()

response.css("li.next a").attrib["href"]
```

After grabing the links in the current page, we may want to go through these links and scraping more
information of that.
A simple example may be like 

```python
    next_page = response.css("li.next a::attr(href)").get()
    if next_page is not None:
        next_page = response.urljoin(next_page)
        yield scrapy.Request(next_page, callback=self.parse)
```

These code will be writen in the body of the method `parse()`.

The method `urljoin()` is used to build a full absolute URL to the target site, since the href
we scraped may be relative.
After that, we will yield a new request to the next page, rigistering itself as callback to handle
the data extraction and keep crawling.

There is a shortcut for creating Request objects called `response.follow()`.

It can be called by multiple types of coeffcients, simply pass a relative URLs, or selector,
or even a HTML elements directly. Then it will work.

#### Important Notice!

> By default, scrapy will obey robots.txt and will not scrape any website with such license.
> To enable this, we can navigate to the project's setting file called `settings.py` and set
> `ROBOTSTXT_OBEY = False`

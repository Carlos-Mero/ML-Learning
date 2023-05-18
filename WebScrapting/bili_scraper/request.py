import requests


def download_html(url, headers):
    res = requests.get(url, headers=headers)
    return res.text


def main():

    url = "https://www.bilibili.com/"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/86.0.4240.111 Safari/537.36',
    }

    html = download_html(url, headers)
    print(html)


if __name__ == "__main__":
    main()

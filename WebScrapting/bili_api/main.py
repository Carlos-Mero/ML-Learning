import requests
import pandas as pd


video_data_url = "https://api.bilibili.com/x/web-interface/view"
video_suggest_url = "https://api.bilibili.com/x/web-interface/archive/related"

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
}

file_name = "bili.csv"
dt = []


def parse_data(data):
    useful_res = {
        "视频时长": data["duration"],
        "视频AV号": data["stat"]["aid"],
        "播放量": data["stat"]["view"],
        "弹幕数": data["stat"]["danmaku"],
        "评论数": data["stat"]["reply"],
        "点赞数": data["stat"]["like"],
        "投币数": data["stat"]["coin"],
        "分享数": data["stat"]["share"],
        "投稿时间": data["pubdate"],
    }
    dt.append(useful_res)


def get_data(bvid):
    params = {
        "bvid": bvid,
    }
    response = requests.get(video_data_url, params=params, headers=headers)
    if response.status_code == 200:
        data_set = response.json()
        if data_set["code"] == 0:
            parse_data(data_set["data"])
        else:
            print("出现一个问题，这一视频无法正常访问。")
    else:
        print("未收到回复。")


def get_related_data(bvid, induct_res):
    if induct_res == 0:
        return
    else:
        induct_res -= 1
        params = {
            "bvid": bvid,
        }
        response = requests.get(
            video_suggest_url, params=params, headers=headers)
        if response.status_code == 200:
            data_set = response.json()
            if data_set["code"] == 0:
                for i in range(0, len(data_set["data"])):
                    parse_data(data_set["data"][i])
                    get_related_data(data_set["data"][i]["bvid"], induct_res)
            else:
                print("出现一个问题，这一视频无法正常访问。")
        else:
            print("未收到回复。")


get_related_data("BV1Ga4y1u7fh", 2)
df = pd.DataFrame(dt)
df.to_csv(file_name, index=False, mode='a+', encoding='utf-8')

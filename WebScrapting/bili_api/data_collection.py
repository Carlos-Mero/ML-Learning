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
target_bvids = {
    "生活": "BV1Ga4y1u7fh",  # 一则生活区的短视频
    "历史": "BV1Fz4y117pV",  # 一则历史区的长视频
    "科技": "BV1Et4y1r7Eu",  # 一则科技区的长视频
    "动漫": "BV1jF411B7sw",  # 一则动漫区的中短视频
    "游戏": "BV1vs411z73L",  # 一则游戏区的长视频
    "鬼畜": "BV1bW411n7fY",  # 一则鬼畜区的短视频
    "舞蹈": "BV1f24y1K7f7",  # 一则舞蹈区的短视频
    "娱乐": "BV1gW4y1K7w8",  # 一则娱乐区的短视频
    "美食": "BV1VL41167Ug",  # 一则美食区的中短视频
    "汽车": "BV1Sa4y1u71e",  # 一则汽车区的中短视频
    "运动": "BV16g411H7mf",  # 一则运动区的中长视频
    "音乐": "BV1ps4y1Q7Tx",  # 一则音乐区的中短视频
    "影视": "BV1sT41127xF",  # 一则影视区的中短视频
    "知识": "BV1wb41127jT",  # 一则知识区的中长视频
    "资讯": "BV1ce4y1f7vJ",  # 一则资讯区的中短视频
    "时尚": "BV1mc411P7ne",  # 一则时尚区的中短视频
}


def parse_data(data, key):
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
        "分区": key,
    }
    dt.append(useful_res)


def get_data(bvid, key):
    params = {
        "bvid": bvid,
    }
    response = requests.get(video_data_url, params=params, headers=headers)
    if response.status_code == 200:
        data_set = response.json()
        if data_set["code"] == 0:
            parse_data(data_set["data"], key)
        else:
            print("出现一个问题，这一视频无法正常访问。")
    else:
        print("未收到回复。")


def get_related_data(bvid, induct_res, key):
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
                    parse_data(data_set["data"][i], key)
                    get_related_data(
                        data_set["data"][i]["bvid"], induct_res, key)
            else:
                print("出现一个问题，这一视频无法正常访问。")
        else:
            print("未收到回复。")


for key in target_bvids:
    get_related_data(target_bvids[key], 3, key)

df = pd.DataFrame(dt)
df.to_csv(file_name, index=False, mode='a+', encoding='utf-8')

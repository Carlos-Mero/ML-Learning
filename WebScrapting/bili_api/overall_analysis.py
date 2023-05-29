# 这个脚本用于执行爬取数据的预处理以及整体情况分析
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rc("font", family='Songti SC')


def refining_data(data):
    """用于对获取的数据进行数据清洗"""
    data.sort_values(by='视频AV号', inplace=True)

    data.drop_duplicates(subset='视频AV号', keep='first', inplace=True)

    last_date = 1684048770
    mask = data["投稿时间"] <= last_date
    data = data[mask]

    data.to_csv("bili-refined.csv", index=False, encoding="utf-8")


def draw_bar_chart(data_mean, data_median):
    """将相关数据绘制成条状图"""
    data_identifiers = {
        "视频时长/秒": True,
        "视频AV号": False,
        "播放量": True,
        "弹幕数x100": 100,
        "评论数": False,
        "点赞数x10": 10,
        "投币数x10": 10,
        "分享数": False,
        "投稿时间": False,
    }

    used_data_mean = data_mean[
        [bool(value) for value in data_identifiers.values()]]
    used_data_median = data_median[
        [bool(value) for value in data_identifiers.values()]]
    print(used_data_mean)
    print(used_data_median)

    selected_options = [
        key for key, value in data_identifiers.items() if value]
    for i in range(0, len(selected_options)):
        used_data_mean[i] *= data_identifiers[selected_options[i]]
        used_data_median[i] *= data_identifiers[selected_options[i]]

    print(used_data_mean)
    print(used_data_median)

    fig, ax_time = plt.subplots()
    ax_num = ax_time.twinx()

    rects = ax_time.bar(
        selected_options[0], used_data_mean[0], color="#bdd0f1", alpha=0.5,
        label='平均值')
    ax_time.bar_label(rects, padding=3)
    rects = ax_time.bar(
        selected_options[0], used_data_median[0], color="#3b60a0", alpha=0.5,
        label='中位数')
    ax_time.bar_label(rects, padding=3)
    rects = ax_num.bar(
        selected_options[1:], used_data_mean[1:], color="#bdd0f1", alpha=0.5,
        label='平均值')
    ax_num.bar_label(rects, padding=3)
    rects = ax_num.bar(
        selected_options[1:], used_data_median[1:], color="#3b60a0", alpha=0.5,
        label='中位数')
    ax_num.bar_label(rects, padding=3)

    ax_time.set_ylabel(selected_options[0])
    ax_num.set_ylabel("数量/次")
    ax_time.set_title("样本整体统计数据")
    ax_num.legend(loc='upper right', ncols=2)
    ax_time.legend(loc='upper right', ncols=2)

    plt.savefig("OverallView.svg", format='svg')

    plt.show()


def calc_stat(data):
    """用于计算各种整体上的统计量"""

    data = data[data["视频时长"] <= 3600]
    data_mean = data.mean(numeric_only=True).to_numpy()
    data_median = data.median(numeric_only=True).to_numpy()
    print(data_median[3])
    draw_bar_chart(data_mean, data_median)


if __name__ == "__main__":
    data = pd.read_csv("bili-refined.csv")
    calc_stat(data)

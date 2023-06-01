# 这一文件用于分析数据的精细结构
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rc("font", family='Songti SC')


def time_to_year(data):
    print(data.head(10))
    st_time = data.head(2)["投稿时间"][1]
    secs_in_a_year = 60.0 * 60 * 24 * 365
    data["投稿时间"] -= st_time
    diff = data["投稿时间"].diff()
    mask = (diff > 0)
    mask = mask.shift(-1)
    mask.iloc[-1] = True
    data = data[mask]
    data["投稿时间"].astype(float)
    data["投稿时间"] /= secs_in_a_year

    return data


def trending(data):

    data = time_to_year(data)
    bins = np.arange(0.0, 13.840634, 1 / 12)
    labels = np.arange(0, len(bins) - 1)
    data['month'] = pd.cut(data["投稿时间"], bins=bins, labels=labels)
    # print(data.head(10)['month'])
    result = data.groupby('month')["视频AV号"].agg(['min', 'max'])
    print(result)

    fig, ax = plt.subplots()
    ax.plot(data["投稿时间"], data["视频AV号"])
    ax.fill_between(data["投稿时间"], data["视频AV号"],
                    color='#bdd0f1', alpha=0.5)
    ax.set_xlabel("时间/年")
    ax.set_ylabel("新增视频数量")

    plt.show()


def trending_by_avid(data):

    data['group'] = pd.qcut(data["视频AV号"], 100, labels=False)
    time_range = pd.date_range(start=data["投稿时间"].min(),
                               end=data["投稿时间"].max(),
                               periods=100)
    data['time_group'] = pd.cut(data["投稿时间"], bins=time_range,
                                labels=False, include_lowest=True)
    group_size = data.groupby('group')["视频AV号"].count()
    gt_size = data.groupby('time_group')["投稿时间"].count()

    data_n = data[(720 < data["视频时长"]) & (data["视频时长"] <= 3600)]
    duration_total_means = data_n.groupby('group')["视频时长"].mean()
    views_total_means = data_n.groupby('group')["播放量"].mean()
    duration_total_medians = data_n.groupby('group')["视频时长"].median()
    views_total_medians = data_n.groupby('group')["播放量"].median()
    n_percentage = data_n.groupby('group')["视频AV号"].count() / group_size
    n_percentage *= 100
    data_n = data_n[data_n["time_group"] >= 33]
    np_perc = data_n.groupby('time_group')["视频时长"].count() / gt_size
    # np_time = data_n.groupby('time_group')["投稿时间"].median()
    np_perc *= 100

    data_s = data[(150 < data["视频时长"]) & (data["视频时长"] <= 720)]
    duration_short_means = data_s.groupby('group')["视频时长"].mean()
    views_short_means = data_s.groupby('group')["播放量"].mean()
    duration_short_medians = data_s.groupby('group')["视频时长"].median()
    views_short_medians = data_s.groupby('group')["播放量"].median()
    s_percentage = data_s.groupby('group')["视频AV号"].count() / group_size
    s_percentage *= 100
    data_s = data_s[data_s["time_group"] >= 33]
    sp_perc = data_s.groupby('time_group')["视频时长"].count() / gt_size
    # sp_time = data_s.groupby('time_group')["投稿时间"].median()
    sp_perc *= 100

    data_ss = data[data["视频时长"] <= 150]
    duration_sshort_means = data_ss.groupby('group')["视频时长"].mean()
    views_sshort_means = data_ss.groupby('group')["播放量"].mean()
    duration_sshort_medians = data_ss.groupby('group')["视频时长"].median()
    views_sshort_medians = data_ss.groupby('group')["播放量"].median()
    ss_percentage = data_ss.groupby('group')["视频AV号"].count() / group_size
    ss_percentage *= 100
    data_ss = data_ss[data_ss["time_group"] >= 33]
    ssp_perc = data_ss.groupby('time_group')["视频时长"].count() / gt_size
    # ssp_time = data_ss.groupby('time_group')["投稿时间"].median()
    ssp_perc *= 100

    st_time = data_s["投稿时间"].min()
    ed_time = data_s["投稿时间"].max()

    # data_l = data[data["视频时长"] >= 18000]
    # duration_long_means = data_l.groupby('group')["视频时长"].mean()
    # views_long_means = data_l.groupby('group')["播放量"].mean()

    fig1, ax_duration = plt.subplots()
    mark = np.arange(0, 100)
    x_ticks = np.linspace(0, 100, 10)
    x_time_ticks = np.arange(st_time.year, ed_time.year + 1, 1)
    x_tick_labels = [str(int(i)) for i in x_ticks]
    x_time_labels = [str(int(i)) for i in x_time_ticks]
    x_time_labels[0] += f".{st_time.month}"
    x_time_labels[-1] += f".{ed_time.month}"
    # t_ticks = np.linspace(st_time, ed_time, 9)

    ax_duration.bar(
        mark, duration_sshort_means, color="#315eab",
        alpha=1, label="短视频平均")
    ax_duration.bar(
        mark, duration_short_means, color="#3b60a0",
        alpha=0.5, label="中间视频平均")
    ax_duration.bar(
        mark, duration_total_means, color="#bdd0f1",
        alpha=0.5, label="长视频平均")
    ax_duration.plot(
        mark, duration_sshort_medians, color="#315eab",
        alpha=1, label="短视频中位")
    ax_duration.plot(
        mark, duration_short_medians, color="#3b60a0",
        alpha=0.5, label="中间视频中位")
    ax_duration.plot(
        mark, duration_total_medians, color="#bdd0f1",
        alpha=0.5, label="长视频中位")
    ax_duration.set_xticks(x_ticks)
    ax_duration.set_xticklabels(x_tick_labels)
    ax_duration.set_xlabel("视频序列位置/%")
    ax_duration.set_ylabel("视频时长/秒")
    ax_duration.legend(loc='upper right', ncols=2)
    ax_duration.set_title("视频时长与发布顺序的统计图")

    plt.savefig("duration-order.png", format="png", dpi=300)

    fig2, ax_views = plt.subplots()

    ax_views.bar(
        mark, views_sshort_means, color="#0f7a13",
        alpha=1, label="短视频平均")
    ax_views.bar(
        mark, views_short_means, color="#e9dea9",
        alpha=0.7, label="中间视频平均")
    ax_views.bar(
        mark, views_total_means, color="#90dc93",
        alpha=0.5, label="长视频平均")
    ax_views.plot(
        mark, views_sshort_medians, color="#0f7a13",
        alpha=1, label="短视频中位")
    ax_views.plot(
        mark, views_short_medians, color="#e9dea9",
        alpha=1, label="中间视频中位")
    ax_views.plot(
        mark, views_total_medians, color="#a9c6bd",
        alpha=1, label="长视频中位")

    ax_views.set_xticks(x_ticks)
    ax_views.set_xticklabels(x_tick_labels)
    ax_views.set_xlabel("视频序列位置/%")
    ax_views.set_ylabel("视频播放量/次")
    ax_views.legend(loc='upper right', ncols=2)
    ax_views.set_title("视频播放量与发布顺序的统计图")

    plt.savefig("views-order.png", format="png", dpi=300)

    fig3, ax_perc = plt.subplots()

    ax_perc.plot(
        mark, n_percentage, color="#b6a9c6", label="长视频占比")
    ax_perc.plot(
        mark, s_percentage, color="#90dc93", label="中间视频占比")
    ax_perc.plot(
        mark, ss_percentage, color="#80a1d9", label="短视频占比")
    ax_perc.fill_between(
        mark, n_percentage, color="#b6a9c6", alpha=0.3)
    ax_perc.fill_between(
        mark, s_percentage, color="#90dc93", alpha=0.3)
    ax_perc.fill_between(
        mark, ss_percentage, color="#80a1d9", alpha=0.3)

    ax_perc.set_xticks(x_ticks)
    ax_perc.set_xticklabels(x_tick_labels)
    ax_perc.set_xlabel("视频序列位置/%")
    ax_perc.set_ylabel("比例/%")
    ax_perc.legend(loc='upper right')
    ax_perc.set_title("各类视频数量占比与发布顺序的统计图")

    plt.savefig("percantage-order.png", format="png", dpi=300)

    fig4, ax_tperc = plt.subplots()

    mark = np.arange(0, 63)
    time_sep = np.linspace(0, 63, 10)
    np_perc = np_perc.iloc[33:]
    sp_perc = sp_perc.iloc[33:]
    ssp_perc = ssp_perc.iloc[33:]

    ax_tperc.plot(
        mark, np_perc, color="#b6a9c6", label="长视频占比")
    ax_tperc.plot(
        mark, sp_perc, color="#90dc93", label="中间视频占比")
    ax_tperc.plot(
        mark, ssp_perc, color="#80a1d9", label="短视频占比")
    ax_tperc.fill_between(
        mark, np_perc, color="#b6a9c6", alpha=0.3)
    ax_tperc.fill_between(
        mark, sp_perc, color="#90dc93", alpha=0.3)
    ax_tperc.fill_between(
        mark, ssp_perc, color="#80a1d9", alpha=0.3)

    ax_tperc.set_xticks(time_sep)
    ax_tperc.set_xticklabels(x_time_labels)
    ax_tperc.set_xlabel("发布时间/年(月)")
    ax_tperc.set_ylabel("所占比例/%")
    ax_tperc.legend(loc='upper left')
    ax_tperc.set_title("各类视频发布数量与发布时间的统计图")

    plt.savefig("num-time.png", format="png", dpi=300)

    plt.show()


def data_domain(data):

    data = data[data["视频时长"] <= 3600]
    grp_data = data.groupby('分区')
    durations_mean = grp_data["视频时长"].mean()
    durations_median = grp_data["视频时长"].median()
    views_mean = grp_data["播放量"].mean()
    views_median = grp_data["播放量"].median()
    x_labels = list(grp_data.groups.keys())
    index = np.arange(16)
    bar_width = 0.34

    fig, ax_duration = plt.subplots()
    ax_views = ax_duration.twinx()

    ax_duration.bar(
        index, durations_mean, bar_width, color="#b6a9c6",
        alpha=0.8, label="时长均值")
    ax_duration.bar(
        index, durations_median, bar_width, color="#670875",
        alpha=0.6, label="时长中位数")
    ax_views.bar(
        index + bar_width+0.03, views_mean, bar_width, color="#b7505d",
        alpha=0.8, label="播放量均值")
    ax_views.bar(
        index + bar_width+0.03, views_median, bar_width, color="#8b0012",
        alpha=0.6, label="播放量中位数")

    ax_duration.set_xticks(index + bar_width/2 + 0.015)
    ax_duration.set_xticklabels(x_labels, rotation=30)
    ax_duration.set_xlabel("视频分区")
    ax_duration.set_ylabel("视频时长/秒")
    ax_views.set_ylabel("播放量/次")
    fig.legend(loc='upper left', bbox_to_anchor=(0.13, 0.87))
    ax_duration.set_title("分区宏观数据")

    # plt.savefig("data-domain.png", format="png", dpi=300)

    likes_mean = grp_data["点赞数"].mean()
    likes_median = grp_data["点赞数"].median()
    coins_mean = grp_data["投币数"].mean()
    coins_median = grp_data["投币数"].median()

    fig1, ax_like = plt.subplots()
    ax_coins = ax_like.twinx()

    ax_like.bar(
        index, likes_mean, bar_width, color="#EAB1D2",
        alpha=0.8, label="点赞量平均数")
    ax_like.bar(
        index, likes_median, bar_width, color="#A44A7E",
        alpha=0.7, label="点赞量中位数")
    ax_coins.bar(
        index+bar_width+0.03, coins_mean, bar_width, color="#C3D69C",
        alpha=0.9, label="硬币量平均数")
    ax_coins.bar(
        index+bar_width+0.03, coins_median, bar_width, color="#79953E",
        alpha=0.7, label="硬币量中位数")

    ax_like.set_xticks(index + bar_width/2 + 0.015)
    ax_like.set_xticklabels(x_labels, rotation=30)
    ax_like.set_xlabel("视频分区")
    ax_like.set_ylabel("点赞数量/个")
    ax_coins.set_ylabel("硬币数量/个")
    fig1.legend(loc='upper left', bbox_to_anchor=(0.13, 0.87))
    ax_like.set_title("分区质量指标")

    # plt.savefig("quality-domain.png", format="png", dpi=300)

    danmaku_mean = grp_data["弹幕数"].mean()
    danmaku_median = grp_data["弹幕数"].median()
    comment_mean = grp_data["评论数"].mean()
    comment_median = grp_data["评论数"].median()

    fig2, ax_danmaku = plt.subplots()
    ax_comment = ax_danmaku.twinx()

    ax_danmaku.bar(
        index, danmaku_mean, bar_width, color="#B0B9D2",
        alpha=0.6, label="弹幕数平均值")
    ax_danmaku.bar(
        index, danmaku_median, bar_width, color="#6C87D0",
        alpha=0.9, label="弹幕数中位数")
    ax_comment.bar(
        index+bar_width+0.03, comment_mean, bar_width, color="#EFD0B5",
        alpha=0.6, label="评论数平均值")
    ax_comment.bar(
        index+bar_width+0.03, comment_median, bar_width, color="#E2A470",
        alpha=0.9, label="评论数中位数")

    ax_danmaku.set_xticks(index + bar_width/2 + 0.015)
    ax_danmaku.set_xticklabels(x_labels, rotation=30)
    ax_danmaku.set_xlabel("视频分区")
    ax_danmaku.set_ylabel("弹幕数/条")
    ax_comment.set_ylabel("评论数/个")
    fig2.legend(loc='upper left', bbox_to_anchor=(0.13, 0.87))
    ax_danmaku.set_title("分区质量指标")

    plt.savefig("quality1-domain.png", format="png", dpi=300)

    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("bili-refined.csv")
    data["投稿时间"] = pd.to_datetime(data["投稿时间"], unit='s')
    data_domain(data)

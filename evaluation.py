import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

import arg_config


def evaluate_typhoon(predictions, targets):
    # lat_long_data = pd.read_csv('result.csv')
    # data = lat_long_data.loc[lat_long_data.Id == 201801, ['LAT', 'LONG', 'UP', 'DOWN', 'LEFT', 'RIGHT']].values
    img_size = arg_config.img_size
    predictions = predictions * img_size
    targets = targets.astype(np.float64) * img_size

    # pd.dataFrame()
    # 轴向为行，删除索引为0的行
    # predictions = np.delete(predictions, 0, axis=0)
    # targets = np.delete(targets, 0, axis=0)

    # target_lats = []
    # target_longs = []
    # pre_lats = []
    # pre_longs = []
    # for i in range(predictions.shape[0]):
    #     pre_long = 0
    #     pre_lat = 0
    #     print(predictions[i][0], targets[i][0], data)
    #     if predictions[i][0] < targets[i][0]:
    #         dist_long = (data[i][1] - data[i][4]) / 64 * (targets[i][0] - predictions[i][0])
    #         pre_long = data[i][1] - dist_long
    #     else:
    #         dist_long = (data[i][5] - data[i][1]) / 64 * (predictions[i][0] - targets[i][0])
    #         pre_long = data[i][1] + dist_long
    #     if predictions[i][1] < targets[i][1]:
    #         dist_lat = (data[i][0] - data[i][2]) / 64 * (targets[i][1] - predictions[i][1])
    #         pre_lat = data[i][0] - dist_lat
    #     else:
    #         dist_lat = (data[i][3] - data[i][0]) / 64 * (predictions[i][1] - targets[i][1])
    #         pre_lat = data[i][0] + dist_lat
    #     target_lats.append(data[i][0])
    #     target_longs.append(data[i][1])
    #     pre_lats.append(pre_lat)
    #     pre_longs.append(pre_long)
    # all_overall = 0.0

    # # 2.散点图,只是用用scat函数来调用即可
    # for i in range(len(pre_lats)):
    #     print("============\n")
    #     print("真实纬度：", target_lats[i], "真实经度：", target_longs[i], "\n")
    #     print("检测纬度：", pre_lats[i], "检测经度：", pre_longs[i], "\n")

    # plt.figure(num=1, figsize=(10, 10))
    # myfig = plt.gcf()  # Get the current figure. If no current figure exists, a new one is created using figure().

    # plt.scatter(pre_longs, pre_lats, s=100, c='r')
    # plt.scatter(target_longs, target_lats, s=100, c='none', marker='o', linewidths=2, edgecolors='b')

    # 设置刻度范围和字体大小
    # plt.ylim(10, 40)
    # plt.xlim(130, 160)
    # plt.yticks(rotation=0, size=15)
    # plt.xticks(rotation=0, size=15)
    # plt.title('路径图', fontdict={'weight': 'normal', 'size': 25})
    # plt.xlabel('Longitude(°)', fontdict={'weight': 'normal', 'size': 18})
    # plt.ylabel('Latitude(°)', fontdict={'weight': 'normal', 'size': 18})
    # 设置图标
    # plt.rcParams.update({'font.size': 18})
    # plt.legend(['EF-LOCNet', 'Helianthus '])
    # 显示所画的图
    # plt.show()
    # plt.show()

    # myfig.savefig('JEBI.png', dpi=400)  # save myfig
    squared_diffs = []  # 存储每个预测值与目标值差的平方
    for k in range(len(predictions)):
        diff = targets[k] - predictions[k]  # 计算差
        diff = np.power(diff, 2)  # 计算差的平方
        squared_diffs.append(diff)  # 将差的平方添加到列表中

    # 计算所有差的平方的平均值
    mean_squared_diff = np.mean(squared_diffs)
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_diff)
    print(f"RMSE: {round(rmse, 10)}")
    return rmse


# def computeErrorOfLAtLon(pre_longs, pre_lats, target_longs, target_lats):
#     all = 0
#     for i in range(len(pre_lats)):
#         a = (pre_longs[i] + pre_lats[i]) / 2
#         b = (target_longs[i] + target_lats[i]) / 2
#         c = math.fabs(a - b)
#         all = all + c
#     error = all / len(pre_lats)
#     return error
def computeErrorOfLAtLon(pre_longs, pre_lats, target_longs, target_lats):
    all = 0
    for i in range(len(pre_lats)):
        a = abs(pre_lats[i] - target_lats[i])
        b = abs(pre_longs[i] - target_longs[i])
        c = (a + b) / 2
        all = all + c
    error = all / len(pre_lats)
    return error


def evaluate_detailed(predictions):
    predictions = predictions * 512.
    targets = np.load('data/TCLD/TEST_LABEL.npy').astype(np.float) * 512.
    easy_points = np.load("data/TCLD/TestEasyIndex.npy")
    hard_points = np.load("data/TCLD/TestHardIndex.npy")

    mean_easy = 0.0
    mean_hard = 0.0
    mean_overall = 0.0

    for i in range(len(easy_points)):
        mean_easypont = np.power((targets[easy_points[i]] - predictions[easy_points[i]]), 2)
        mean_easypont = np.sqrt(np.sum(mean_easypont))
        mean_easy = mean_easy + mean_easypont

    for m in range(len(hard_points)):
        mean_hardpoint = np.power((targets[hard_points[m]] - predictions[hard_points[m]]), 2)
        mean_hardpoint = np.sqrt(np.sum(mean_hardpoint))
        mean_hard = mean_hard + mean_hardpoint

    for k in range(len(predictions) - 1):
        mean_overpoint = np.power((targets[k + 1] - predictions[k + 1]), 2)
        mean_overpoint = np.sqrt(np.sum(mean_overpoint))
        mean_overall = mean_overall + mean_overpoint

    mean_easy = mean_easy / len(easy_points)
    mean_hard = mean_hard / len(hard_points)
    mean_overall = mean_overall / (len(predictions) - 1)

    return mean_overall, mean_easy, mean_hard


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(math.radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance = 2 * math.asin(math.sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance
if __name__ == '__main__':
    reg=np.load(arg_config.root+'runs/testing/mynet/regression.npy')
    print(reg)
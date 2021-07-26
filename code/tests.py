#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/24 15:17
# @Author : jackfrank
# @Version：V 0.1
# @File : tests.py
# @desc :
"""
    这是一个测试类
"""
import matplotlib
from Cython import inline

from code.utils import *
import time
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def test_1(path0):
    """
    测试utils.py里的read_data函数，以及如何将景区名称提取出来
    :param path0:
    :return:
    """
    print("开始读取excel文件...")
    data_dict0 = read_data(path0)
    name_list = []
    time.sleep(1.5)
    print("读取全部景区名称...")
    for i in data_dict0["景区名称"]:
        name = data_dict0["景区名称"][i]
        print(name)
        name_list.append(name)
    time.sleep(1.5)
    if len(name_list) > 0:
        print("添加成功！")
    time.sleep(1.5)
    print("出去重复添加的景区名称...")
    name_list = list(set(name_list))
    time.sleep(1.5)
    print("显示结果...")
    print(name_list)


def stacked_bar(data_entry, labels,
                title="the title of fig",
                x_label="x_label",
                y_label="y_label",
                width=0.35
                ):
    datas = list(data_entry.values())
    keys = list(data_entry.keys())
    fig, ax = plt.subplots()

    for i in range(len(data_entry)):
        if i == 0:
            ax.bar(labels, datas[0], width, align="center", label=keys[0])  # , yerr=men_std(表示误差大小)
            bottom_value = datas[0]
        else:
            ax.bar(labels, datas[i], width, align="center", bottom=bottom_value, label=keys[i])
            bottom_value = np.array(bottom_value) + np.array(datas[i])

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if x_label != "":
        ax.set_xlabel(x_label)
    if y_label != "":
        ax.set_ylabel(y_label)

    ax.set_ylim(0, 5000)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(0.5, 1.17), ncol=int(len(data_entry)))

    return plt


if __name__ == '__main__':
    # path = "../resources/review/景区评论.xlsx"
    # data_dict = read_data(path)
    # merge_data = merge_data(data_dict)
    # path1 = "../resources/merged_text/"
    # write2txt(merge_data, path1)  #将原始评论数据存储到TXT文件
    # cut_data = prep_data(merge_data)
    # print(cut_data['A01'])
    # path2 = "../resources/cut_text/"
    # write2txt(cut_data, path2)  # 将预处理后的评论数据存储到TXT文件
    path1 = "../resources/simhash_txt/"
    path2 = "../resources/dedup_txt/"
    all_files = os.listdir(path1)
    origin = {}
    rep = {}
    name = []
    quantity1 = []  # 去重之后的数据
    quantity2 = []  # 重复的数据
    for file_name in all_files:
        file1 = open(path1 + file_name, encoding='utf-8')
        file2 = open(path2 + file_name, encoding='utf-8')
        data1 = file1.readlines()
        data2 = file2.readlines()
        origin.update({file_name.replace(".txt", ""): len(data1)})
        rep.update({file_name.replace(".txt", ""): len(data2)})
        name.append(file_name.replace(".txt", ""))
        quantity1.append(len(data2))
        quantity2.append(len(data1) - len(data2))

    labels = name[:10]
    data_entry = {
        "vaild comment": quantity1[:10],
        "repeated comment": quantity2[:10]
    }
    plt = stacked_bar(data_entry, labels, x_label="")
    plt.savefig('全部景区评论详情.pdf')
    plt.show()


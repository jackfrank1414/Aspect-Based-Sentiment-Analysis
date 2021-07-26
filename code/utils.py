#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/24 15:12
# @Author : jackfrank
# @Version：V 0.1
# @File : utils.py
# @desc :
"""
    这是一个工具类
"""

import jieba
import jieba.analyse
import pandas as pd
import time
import re


def read_data(path):
    """
    读取excel文件，
    函数输入为文件所在的路径，并将文件转化成字典的形式输出
    :param path:    path为文件所在的路径
    :return dict_data0:    dict_data0是保存了excel所有信息的字典，key为excel表中的不同列名，value为对应列下的所有信息
    """
    origin_data = pd.read_excel(path)
    data_dict0 = origin_data.to_dict()
    return data_dict0


def merge_data(data_dict):
    """
    提取字典中存储的指定comment信息，
    函数输入为字典形式的所有文档信息，key为excel表中的不同列名，value为对应列下的所有信息，
    函数将comment信息从输入的字典中提取出来，并返回以字典形式保存的comment信息
    :param data_dict:    保存着excel所有信息的字典
    :return merged_data0：   提取出来的comment信息，key为不同的景区名称，value为对应的多条评论，评论以列表的形式保存
    """
    # time.sleep(2)
    print("提取景区名称...")
    name_list = []  # 保存景区名称的list
    for i in data_dict["景区名称"]:
        name = data_dict["景区名称"][i]
        # print(name)
        name_list.append(name)
    name_list = list(set(name_list))

    # time.sleep(2)
    print("提取评论内容...")
    # time.sleep(2)
    merge_data0 = {}
    for name in name_list:
        print("开始提取" + name + "景区的评论内容...")
        # time.sleep(4)
        review_list = []
        for i in data_dict["景区名称"]:
            if name == data_dict["景区名称"][i]:
                review_list.append(data_dict["评论内容"][i])
                # print(data_dict["评论内容"][i])
        review_dict = {name: review_list}
        merge_data0.update(review_dict)
        # review_list.clear()
        # review_dict.clear()
    return merge_data0


def write2txt(input_dict, folder_path):
    """
    将数据保存为TXT格式
    :param folder_path:
    :param input_dict:
    :return:
    """
    for key in input_dict:
        print("将" + key + "景区评论内容写入TXT文件...")
        file = open(folder_path + key + '.txt', 'a', encoding='utf-8')
        for line in input_dict[key]:
            file.write(line + '\n')


def stopwordslist(filepath):
    """
    加载停用词表，并生成停用词列表
    :param filepath:    停用词表的路径
    :return stopwords:  停用词列表
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def seg_sentence(sentence):
    """
    分词，并去除句子中的中文停用词
    :param sentence:    需处理的字符串
    :return outstr:     处理后的字符串
    """
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('../resources/哈工大停用词表扩展.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


def prep_data(input_data):
    """
    对评论进行分词、去停用词等
    :param input_data:
    :return input_data:
    """
    for key in input_data:
        for i in range(len(input_data[key])):
            input_data[key][i] = seg_sentence(input_data[key][i])
    return input_data




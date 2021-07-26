#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/25 19:24
# @Author : jackfrank
# @Version：V 0.1
# @File : word2vec.py
# @desc :
"""
    将预处理好的评论数据使用word2vec进行向量化处理
"""

from code.utils import *
import gensim


def pro_cut_data(input_data):
    """
    将评论数据打包处理成word2vec的输入list格式如：[['A','B','C'],['A','D','F'],[...],...]
    :param input_data:
    :return input_list:
    """
    input_list = []
    for key in input_data:
        for line in input_data[key]:
            word_list = line.split()
            input_list.append(word_list)
    return input_list


if __name__ == '__main__':
    path = "../resources/review/景区评论.xlsx"
    data_dict = read_data(path)
    merge_data = merge_data(data_dict)
    cut_data = prep_data(merge_data)
    word_list_list = pro_cut_data(cut_data)
    print(len(word_list_list))
    model = gensim.models.Word2Vec(word_list_list, sg=1, size=100, window=5, min_count=1, negative=3, sample=0.001,
                                   hs=1,
                                   workers=4)
    model.save("../model/scenic_spot")

#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/6 15:06
# @Author : jackfrank
# @Version：V 0.1
# @File : simhash.py
# @desc :
"""
    simhash算法用于计算文本相似度
"""

# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
import numpy as np
import os


class simhash:
    def __init__(self, content):
        self.simhash = self.simhash(content)

    def __str__(self):
        return str(self.simhash)

    def simhash(self, content):
        seg = jieba.cut(content)
        # jieba.analyse.set_stop_words('../resources/哈工大停用词表扩展.txt')
        keyWord = jieba.analyse.extract_tags(
            '|'.join(seg), topK=20, withWeight=True, allowPOS=())  # 在这里对jieba的tfidf.py进行了修改
        # 将tags = sorted(freq.items(), key=itemgetter(1), reverse=True)修改成tags = sorted(freq.items(), key=itemgetter(1,0), reverse=True)
        # 即先按照权重排序，再按照词排序
        keyList = []
        # print(keyWord)
        for feature, weight in keyWord:
            weight = int(weight * 20)
            feature = self.string_hash(feature)
            temp = []
            for i in feature:
                if (i == '1'):
                    temp.append(weight)
                else:
                    temp.append(-weight)
            # print(temp)
            keyList.append(temp)
        list1 = np.sum(np.array(keyList), axis=0)
        # print(list1)
        if keyList == []:  # 编码读不出来
            return '00'
        simhash = ''
        for i in list1:
            if (i > 0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'
        return simhash

    def string_hash(self, source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            # print(source, x)

            return str(x)

    # def hammingDis(self, com):
    #     t1 = '0b' + self.simhash
    #     t2 = '0b' + com.simhash
    #     n = int(t1, 2) ^ int(t2, 2)
    #     i = 0
    #     while n:
    #         n &= (n - 1)
    #         i += 1
    #     return i


def xor(x, y):
    n = int(x, 2) ^ int(y, 2)
    i = 0
    while n:
        n &= (n - 1)
        i += 1
    return i


if __name__ == '__main__':

    # 将所有景区网评文本的simhash值计算出来，并保存到TXT文件当中
    # path = "../resources/cut_text/"
    # all_files = os.listdir(path)
    # for file in all_files:
    #     print(file + "start!" + '\n')
    #     output = open("../resources/simhash_txt/" + file, 'a', encoding='utf-8')
    #     for line in open(path + file, encoding='utf-8'):
    #         s = simhash(line)
    #         output.write(s.__str__() + '\n')
    #     print(file + "complete!" + '\n')

    path1 = "../resources/simhash_txt/"
    path2 = "../resources/similar_txt/"
    path3 = "../resources/dedup_txt/"
    path4 = "../resources/cut_text/"
    all_files = os.listdir(path1)
    for file_name in all_files:
        file1 = open(path1 + file_name, encoding='utf-8')
        file2 = open(path2 + file_name, 'a', encoding='utf-8')
        file3 = open(path3 + file_name, 'a', encoding='utf-8')
        file4 = open(path4 + file_name, encoding='utf-8')
        data1 = file1.readlines()
        data4 = file4.readlines()
        length = len(data1)
        print(file_name + "start!")
        for i in range(0, length - 1):
            try:
                file3.write(data4[i])
            except:
                continue
            for j in range(i + 1, length):
                try:
                    if xor(data1[i], data1[j]) <= 3:
                        data1.pop(j)
                        data4.pop(j)
                        file2.write(data4[i])
                except:
                    continue
        print(file_name + "原来的长度：" + str(length) + ",去重之后的长度：" + str(len(data1)))
        print(file_name + "complete!")

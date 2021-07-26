#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/7 8:34
# @Author : jackfrank
# @Version：V 0.1
# @File : extract.py
# @desc :
"""
    
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from code.train import trunc_pad, word_to_ix

filename = '../resources/cut_text/A01.txt'
predict_word = []
with open(filename, 'r', encoding='UTF-8')as txt_file:
    lines = txt_file.readlines()
    for line in lines:
        predict_word.append(line.split())

preword_word_set = predict_word

for item_st in range(len(preword_word_set)):
    preword_word_set[item_st] = trunc_pad(preword_word_set[item_st], if_sentence=True)

for sentence in preword_word_set:
    for i in range(len(sentence)):
        if sentence[i] not in word_to_ix:
            sentence[i] = "[PAD]"

for i in range(len(preword_word_set)):
    preword_word_set[i] = [word_to_ix[t] for t in preword_word_set[i]]

model = torch.load('bilstm_crf.pkl')
# Check predictions after training
with torch.no_grad():
    precheck_sent = torch.tensor(preword_word_set, dtype=torch.long).cuda()
    precheck_label = model.predict(precheck_sent.view(len(precheck_sent), -1))



filename = '../resources/NER_txt/A01.txt'
with open(filename, 'w', encoding='UTF-8')as txt_file:
    for label in precheck_label:
        txt_file.write(' '.join(label) + '\n')
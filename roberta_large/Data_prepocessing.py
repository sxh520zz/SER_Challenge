#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import wave
import numpy as np
import python_speech_features as ps
import soundfile as sf
import os
import glob
import pickle
import torch
import torchaudio

with open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/MSP_Fea/MSP_Text.pickle', 'rb') as file:
    MSP_roberta = pickle.load(file)
with open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/MSP_Fea/MSP_Label.pickle', 'rb') as file:
    MSP_Label = pickle.load(file)

def emo_change(x):
    if x == 'A':
        x = 0
    if x == 'S':
        x = 1
    if x == 'H':
        x = 2
    if x == 'U':
        x = 3
    if x == 'F':
        x = 4
    if x == 'D':
        x = 5
    if x == 'C':
        x = 6
    if x == 'N':
        x = 7
    if x == 'O':
        x = 8
    if x == 'X':
        x = 9
    return x

def Seg_IEMOCAP(train_data_SSL,train_data_Label):
    for i in range(len(train_data_SSL)):
        for x in range(len(train_data_Label)):
                if (train_data_SSL[i]['id'] == train_data_Label[x]['Name']):
                    train_data_Label[x]['input_ids'] = train_data_SSL[i]['input_ids']
                    train_data_Label[x]['attention_mask'] = train_data_SSL[i]['attention_mask']
                    print(i)
    return train_data_Label

def Train_data(train_map):
    train_data_ALL_1 = []
    label_list= [0,1,2,3,4,5,6,7]
    num = 0
    for i in range(len(train_map)): 
        data = {}
        data['label'] = emo_change(train_map[i]['Emo_main'])
        data['input_ids'] = train_map[i]['input_ids']
        data['attention_mask'] = train_map[i]['attention_mask']
        data['id'] = train_map[i]['Name']
        data['Partitions'] = train_map[i]['Partitions']
        if(data['label'] in label_list):
            train_data_ALL_1.append(data)
            num = num + 1

    print(len(train_data_ALL_1))
    print(num)

    data_train = []
    data_dev = []
    #data_test = []

    for i in range(len(train_data_ALL_1)):
        if (train_data_ALL_1[i]['Partitions'] == 'Train'):
            data_train.append(train_data_ALL_1[i])
        if (train_data_ALL_1[i]['Partitions'] == 'Development'):
            data_dev.append(train_data_ALL_1[i])
        #if (train_data_ALL_1[i]['Partitions'] == 'Test3'):
            #data_test.append(train_data_ALL_1[i])
    data = []
    data.append(data_train)
    data.append(data_dev)
    #data.append(data_test)
    return data

if __name__ == '__main__':
    train_data_map = Seg_IEMOCAP(MSP_roberta,MSP_Label)
    Train_data = Train_data(train_data_map)

    file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/Baseline/roberta_large/Train_data.pickle', 'wb')
    pickle.dump(Train_data, file)
    file.close()
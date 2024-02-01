#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import wave
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import os
import glob
import pickle
import csv
import python_speech_features as ps

import re
import wave
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob
import pickle
import csv
import torch
import torchaudio


#Data information
Audio_Data_dir = '/mnt/data1/liyongwei/Database/MSP/Audios/'
Transcripts_Data_dir = '/mnt/data1/liyongwei/Database/MSP/Transcripts/'
Label_dir = '/mnt/data1/liyongwei/Database/MSP/Labels/labels.txt'

Partitions = '/mnt/data1/liyongwei/Database/MSP/Partitions.txt'
Speaker_dir = '/mnt/data1/liyongwei/Database/MSP/Speaker_ids.txt'

#SSL information
from transformers import AutoProcessor, HubertModel
processor = AutoProcessor.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/hubert-large-ls960-ft")

from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/roberta-large")
MAX_LEN = 128


def emo_change(x):
    if x == 'xxx' or x == 'oth':
        x = 0
    if x == 'neu':
        x = 1
    if x == 'hap':
        x = 2
    if x == 'ang':
        x = 3
    if x == 'sad':
        x = 4
    if x == 'exc':
        x = 5
    if x == 'sur':
        x = 6
    if x == 'fea':
        x = 7
    if x == 'dis':
        x = 8
    if x == 'fru':
        x = 9
    return x

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

def process_wav_file(wav_file, time):
    waveform, sample_rate = torchaudio.load(wav_file)
    target_length = time * sample_rate
    # 将WAV文件裁剪为目标长度
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        # 如果WAV文件长度小于目标长度，则使用填充进行扩展
        padding_length = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))

    return waveform, sample_rate

def Read_MSP_SSL():
    train_num = 0
    train_mel_data = []
    for sess in os.listdir(Audio_Data_dir):
        file_dir = os.path.join(Audio_Data_dir, sess)
        wavname = sess.split("/")[-1][:-4]
        # training set
        one_mel_data = {}
        audio_input, sample_rate = process_wav_file(file_dir,3)
        input_values = processor(audio_input, sampling_rate=sample_rate,
                                    return_tensors="pt").input_values
        one_mel_data['id'] = wavname
        one_mel_data['SSL_data'] = input_values
        train_mel_data.append(one_mel_data)
        train_num = train_num + 1
        print(train_num)
    return train_mel_data

def Read_MSP_Text():
    
    files_dict = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(Transcripts_Data_dir):
        # 确保文件以.txt为扩展名
        if filename.endswith(".txt"):
            text_map ={}
            file_path = os.path.join(Transcripts_Data_dir, filename)
            
            # 从文件名中提取id，即去掉扩展名的部分
            file_id = os.path.splitext(filename)[0]

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            text_map['id'] = file_id
            # 使用正则表达式匹配并替换 "[...]" 格式的内容
            content_without_brackets = re.sub(r'\[.*?\]', '', file_content)

            # 新增使用bert的tokenizer， 使用正则表达式去除标点符号
            str = re.sub(r'[^\w\s]', '', content_without_brackets)
            # 对输入文本进行tokenization并添加特殊token
            encoding = tokenizer(str, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
            input_ids = encoding.get('input_ids').squeeze()
            attention_mask = encoding.get('attention_mask').squeeze()
            text_map['input_ids'] = input_ids.float()
            text_map['attention_mask'] = attention_mask.float()

            files_dict.append(text_map)

    return files_dict

def Read_MSP_Spec():
    filter_num = 40
    train_num = 0
    train_mel_data = []
    for sess in os.listdir(Audio_Data_dir):
        file_dir = os.path.join(Audio_Data_dir, sess)
        wavname = sess.split("/")[-1][:-4]
        data, time, rate = read_file(file_dir)
        mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
        # training set
        mel_data = []
        one_mel_data = {}
        part = mel_spec
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)
        input_data_1 = np.concatenate((part, delta1), axis=1)
        input_data = np.concatenate((input_data_1, delta2), axis=1)
        mel_data.append(input_data)
        one_mel_data['id'] = wavname
        mel_data = np.array(mel_data)
        one_mel_data['spec_data'] = mel_data
        train_mel_data.append(one_mel_data)
        train_num = train_num + 1
        print(train_num)
    return train_mel_data

def Seg_IEMOCAP(train_data_spec,train_data_text,train_data_trad):
    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_trad)):
                if (train_data_text[i][x]['id'] == train_data_trad[y]['id']):
                    train_data_text[i][x]['hubert_data'] = train_data_trad[y]['wav2vec_data']

    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_spec)):
                if (train_data_text[i][x]['id'] == train_data_spec[y]['id']):
                    train_data_text[i][x]['spec_data'] = train_data_spec[y]['spec_data']

    num = 0
    train_data_map = []
    for i in range(len(train_data_text)):
        data_map_1 = []
        for x in range(len(train_data_text[i])):
            if (len(train_data_text[i][x]) == 8):
                data_map_1.append(train_data_text[i][x])
                num = num + 1
        train_data_map.append(data_map_1)
    print(num)
    #train_data_map = normalization(train_data_map,'trad_data')
    return train_data_map

def Read_MSP_Label():
    Label_file = []
    with open(Label_dir, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if(line[0] == 'M'):
            parts = line.strip().split('; ')
            data_dict = {}
            data_dict['Name'] = parts[0][:-4]
            data_dict['Emo_main'] = parts[1]
            data_dict['A'] = float(parts[2][2:-1])
            data_dict['V'] = float(parts[3][2:-1])
            data_dict['D'] = float(parts[4][2:-1])
            Label_file.append(data_dict)

    #读取 train dev test 索引
    with open(Partitions, 'r') as file:
        lines = file.readlines()

    # 将train,dev,test的id 存入列表
    Partition_list = {'Train': [], 'Development': [], 'Test3': []}
    for line in lines:
        # 分割每行字符串，以分号为分隔符
        parts = line.strip().split('; ')
        if len(parts) == 2 and parts[0] == 'Train':
            Partition_list['Train'].append(parts[1][:-4])
        if len(parts) == 2 and parts[0] == 'Development':
            Partition_list['Development'].append(parts[1][:-4])
        if len(parts) == 2 and parts[0] == 'Test3':
            Partition_list['Test3'].append(parts[1][:-4])

    for x in range(len(Label_file)):
        if(Label_file[x]['Name'] in Partition_list['Train']):
            Label_file[x]['Partitions'] = 'Train'
        elif(Label_file[x]['Name'] in Partition_list['Development']):
            Label_file[x]['Partitions'] = 'Development'
        elif(Label_file[x]['Name'] in Partition_list['Test3']):
            Label_file[x]['Partitions'] = 'Test3'



    #读取 train dev test 索引
    with open(Speaker_dir, 'r') as file:
        lines = file.readlines()

    Speaker = []
    Gander = {'Male': [], 'Female': []}
    print(len(lines))
    for line in lines:
        # 分割每行字符串，以分号为分隔符
        parts = line.strip().split(';')
        if(len(parts)== 2):
            if(parts[0][0] == 'S'):
                if(parts[1] == ' Male'):
                    Gander['Male'].append(parts[0])
                if(parts[1] == ' Female'):
                    Gander['Female'].append(parts[0])
            if(parts[0][0] == 'M'):
                speaker_dic = {}
                speaker_dic['Name'] = parts[0][:-4]
                speaker_dic['speaker_id'] = parts[1]
                speaker_name = 'Speaker_' + speaker_dic['speaker_id']
                if(speaker_name in Gander['Male']):
                    speaker_dic['Gander'] = 'Male'
                if(speaker_name in Gander['Female']):
                    speaker_dic['Gander'] = 'Female'
                if(len(speaker_dic) == 3):
                    Speaker.append(speaker_dic)
    print(len(Speaker))


    for i in range(len(Speaker)):
        for j in range(len(Label_file)):
            print(Speaker[i]['Name'])
            print(Label_file[j]['Name'])
            if(Speaker[i]['Name'] == Label_file[j]['Name']):
                Speaker[i]['Emo_main'] = Label_file[j]['Emo_main']
                Speaker[i]['A'] = Label_file[j]['A']
                Speaker[i]['V'] = Label_file[j]['V']
                Speaker[i]['D'] = Label_file[j]['D']
                Speaker[i]['Partitions'] = Label_file[j]['Partitions']
    Fin_data = []
    for i in range(len(Speaker)):
        if(len(Speaker[i]) == 8):
            Fin_data.append(Speaker[i])
    print(Fin_data[0])
    # 指定CSV文件路径
    csv_file_path = '/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/MSP_Fea/output.csv'

    # 将字典写入CSV文件
    with open(csv_file_path, 'w', newline='') as csv_file:
        # 定义CSV文件的表头（header）
        fieldnames = ['Name', 'speaker_id', 'Gander', 'Emo_main', 'A', 'V', 'D', 'Partitions']
        
        # 创建CSV写入器
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # 写入表头
        csv_writer.writeheader()
        
        # 写入数据
        csv_writer.writerows(Fin_data)

    return Fin_data

def normalization(data,name):
    need_norm = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            need_norm.append(data[i][j][name][0])
    Scaler = StandardScaler().fit(need_norm)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][name] = Scaler.transform(data[i][j][name])
    return data


if __name__ == '__main__':
    '''
    #Mel
    Train_data_spec = Read_MSP_Spec()
    file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/MSP_Fea/MSP_Spec.pickle', 'wb')
    pickle.dump(Train_data_spec, file)
    file.close()

    #OpenSmile 
    #train_data_trad = Read_MSP_Trad()

    #Text
    Train_data_text = Read_MSP_Text()
    file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/MSP_Fea/MSP_Text.pickle', 'wb')
    pickle.dump(Train_data_text, file)
    file.close()
    '''
    '''
    #SSL
    Train_data_SSL = Read_MSP_SSL()
    file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/MSP_Fea/MSP_Hubert.pickle', 'wb')
    pickle.dump(Train_data_SSL, file)
    file.close()
    '''
    #Label

    Train_data_label = Read_MSP_Label()
    file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/MSP_Fea/MSP_Label.pickle', 'wb')
    pickle.dump(Train_data_label, file)
    file.close()  



    '''
    train_data_map = Seg_IEMOCAP(Train_data_spec,Train_data_text,Train_data_SSL,Train_data_label)
    file = open('Speech_data_hubert.pickle', 'wb')
    pickle.dump(train_data_map, file)
    file.close()  
    '''

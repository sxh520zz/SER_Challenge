#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:10:17 2019

@author: jdang03
"""

import pickle
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open('/mnt/data1/liyongwei/Project/Yongwei_Code/IEM/SSL-IEM/Wav2vec_large/Final_result.pickle', 'rb') as file:
    final_result =pickle.load(file)
with open('/mnt/data1/liyongwei/Project/Yongwei_Code/IEM/SSL-IEM/Wav2vec_large/Final_f1.pickle', 'rb') as file:
    Final_f1 =pickle.load(file)

print(Final_f1)
'''
cret = 0    
num = 0   
for i in range(len(final_result)):
    for j in range(len(final_result[i])):
        for x in range(len(final_result[i][j]['predict_label'])):
            num = num + 1
            if(final_result[i][j]['predict_label'][x] == final_result[i][j]['true_label'][x]):
                cret  = cret + 1
print(cret,num,cret/num)
'''                
true_label = []    
predict_label = []   
predict_fea = []  
num = 0
for i in range(len(final_result)):
    for j in range(len(final_result[i])):
        num = num +1
        predict_label.append(final_result[i][j]['Predict_label'])
        true_label.append(final_result[i][j]['True_label'])
        predict_fea.append(np.array(final_result[i][j]['Predict_fea']))
print(num)            

accuracy_recall = recall_score(true_label, predict_label, average='macro')
accuracy_f1 = metrics.f1_score(true_label, predict_label, average='macro')
CM_test = confusion_matrix(true_label,predict_label)    
           
print(accuracy_recall,accuracy_f1)
print(CM_test)      

#-------------------------计算WA 和UA
predict_label = np.array(predict_label)
true_label = np.array(true_label)
wa = np.mean(predict_label.astype(int) == true_label.astype(int))

predict_label_onehot = np.eye(4)[predict_label.astype(int)]
true_label_onehot = np.eye(4)[true_label.astype(int)]
ua = np.mean(np.sum((predict_label_onehot == true_label_onehot)*true_label_onehot, axis =0 )/np.sum(true_label_onehot,axis =0))

print('UA={:.4f}, WA={:.4f}, F1={:.4f}' .format(ua,wa, accuracy_f1))
#print('WA={:.4f}'.format(wa))
#print(CM_test)

predict_fea = np.array(predict_fea)

#print(X_tsne)
#T_SNE(X_tsne,label_true)

# T-SNE 可视化
label_map = {label: idx for idx, label in enumerate(set(true_label))}
y_int = [label_map[label] for label in true_label]
plt.figure(figsize=(8, 6))
unique_labels = list(set(true_label))  # 获取标签的唯一值
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))  # 生成彩虹颜色
for i, label in zip(range(len(unique_labels)), unique_labels):
    plt.scatter(predict_fea[np.array(true_label) == label, 0], predict_fea[np.array(y_int) == label, 1],
                c=colors[i], label=f'Label {label}')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.savefig("/mnt/data1/liyongwei/Project/Yongwei_Code/IEM/SSL-IEM/Wav2vec_large/T-SNE_Wav2vec.jpg") 

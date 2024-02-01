import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
import torch.optim as optim
from torch.optim import AdamW
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from utils import Get_data
from torch.autograd import Variable
from models import SpeechRecognitionModel
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from transformers import Wav2Vec2Model

from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.enabled = False

with open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/Baseline/roberta_large/Train_data.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=8)
parser.add_argument('--utt_insize', type=int, default=1024)
args = parser.parse_args()

torch.manual_seed(args.seed)

def Train(epoch):
    train_loss = 0
    model.train()
    for batch_idx, (ids, att, target)  in enumerate(train_loader):
        if args.cuda:
            target, ids, att = target.cuda(), ids.cuda(), att.cuda()
        target, ids, att =  Variable(target),Variable(ids),Variable(att)
        target = target.squeeze()
        utt_optim.zero_grad()

        #data_1 = data_1.squeeze()
        #data_2 = data_2.squeeze(1)
        #data_2 = data_2.squeeze(1)

        line_out = model(ids, att)
        loss = torch.nn.CrossEntropyLoss()(line_out, target.long())

        loss.backward()

        utt_optim.step()
        train_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0

def Test():
    model.eval()
    label_pre = []
    label_true = []
    fea_pre = []
    with torch.no_grad():
        for batch_idx, (ids, att, target) in enumerate(test_loader):
            if args.cuda:
                target, ids, att = target.cuda(), ids.cuda(), att.cuda()
            target, ids, att =  Variable(target),Variable(ids),Variable(att)
            target = target.squeeze()
            utt_optim.zero_grad()

            #data_1 = data_1.squeeze()
            #data_2 = data_2.squeeze(1)
            #data_2 = data_2.squeeze(1)

            line_out = model(ids, att)
            output = torch.argmax(line_out, dim=1)
            fea_pre.extend(line_out.cpu().data.numpy())
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
        fea_pre = np.vstack(fea_pre)
        tsne = TSNE(n_components=2, learning_rate='auto', random_state=42)
        X_tsne = tsne.fit_transform(fea_pre)
        #print(X_tsne)
        #T_SNE(X_tsne,label_true)
    return accuracy_f1, accuracy_recall, label_pre, label_true,X_tsne

Final_result = []
Final_f1 = []
result_label = []

train = [0]
test = [1]

train_loader, test_loader, input_test_data_id, input_test_label_org = Get_data(data, train, test, args)
model = SpeechRecognitionModel(args)
if args.cuda:
    model = model.cuda()

lr = args.lr
utt_optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
utt_optim = optim.Adam(model.parameters(), lr=lr)
f1 = 0
recall = 0
predict = copy.deepcopy(input_test_label_org)
result_fea = []
for epoch in range(1, args.epochs + 1):
    Train(epoch)
    accuracy_f1, accuracy_recall, pre_label, true_label, pre_fea = Test()
    if (accuracy_recall > recall):
        predict_fea = copy.deepcopy(pre_fea)
        num = 0
        for x in range(len(predict)):
            predict[x] = pre_label[num]
            num = num + 1
        result_label = predict
        result_fea = pre_fea
        recall = accuracy_recall
    print("Best Result Until Now:")
    print(recall)

onegroup_result = []
for i in range(len(input_test_data_id)):
    a = {}
    a['id'] = input_test_data_id[i]
    a['Predict_label'] = result_label[i]
    a['Predict_fea'] = result_fea[i]
    a['True_label'] = input_test_label_org[i]
    onegroup_result.append(a)
Final_result.append(onegroup_result)
Final_f1.append(recall)
file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/Baseline/roberta_large/Final_result.pickle', 'wb')
pickle.dump(Final_result,file)
file.close()
file = open('/mnt/data1/liyongwei/Project/Xiaohan_code/Odyssey_SER_Challenge/Baseline/roberta_large/Final_f1.pickle', 'wb')
pickle.dump(Final_f1,file)
file.close()
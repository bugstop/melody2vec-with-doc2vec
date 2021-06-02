import json
import gensim
import collections
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def print_precison_recall_f1(y_true, y_pre):
    """打印精准率、召回率和F1值"""
    # print("打印精准率、召回率和F1值")
    # print(classification_report(y_true, y_pre))
    a = round(accuracy_score(y_true, y_pre), 4)
    f1 = round(f1_score(y_true, y_pre, average='weighted'), 4)
    p = round(precision_score(y_true, y_pre, average='weighted'), 4)
    r = round(recall_score(y_true, y_pre, average='weighted'), 4)
    # print("Accuracy: {}, Precision: {}, Recall: {}, F1: {} ".format(a, p, r, f1))
    return a, f1, p, r


with open('melody.json') as f:
    melody = json.load(f)
with open('top_magd.json') as f:
    top_magd = json.load(f)
with open('masd.json') as f:
    masd = json.load(f)

model = gensim.models.Doc2Vec.load('d2v.bin')

X = [model.docvecs[sample[-1]] for sample in masd]
Y = [sample[-2] for sample in masd]

class_names = list(sorted(set(Y)))
print(len(set(Y)))  # 25
print([(i, j) for i, j in enumerate(class_names)])
word_counts = collections.Counter(Y)
print(word_counts)

for i, name in enumerate(Y):
    Y[i] = class_names.index(name)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=1,
    stratify=Y
)

"""
xgb = XGBClassifier(min_child_weight=5, max_depth=20, objective='multi:softmax', num_class=13)
xgb.fit(np.array(X_train), Y_train)

importance_features_top('xgboost', xgb, X_train)

Y_pred = xgb.predict(X_val)
print_precison_recall_f1(Y_val, Y_pred)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = [[x.reshape(1, 1, 256), y] for x, y in zip(X_train, Y_train)]

test_dataset = [[x.reshape(1, 1, 256), y] for x, y in zip(X_val, Y_val)]

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(256, 512)
        self.d1 = nn.Dropout(p=0.25)
        self.l2 = nn.Linear(512, 256)
        self.d2 = nn.Dropout(p=0.25)
        self.l3 = nn.Linear(256, 128)
        self.d3 = nn.Dropout(p=0.25)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 25)

    def forward(self, x):
        # Flatten the data (n, 1, 28, 28) --> (n, 784)
        x = x.view(-1, 256)
        x = F.relu(self.l1(x))
        # x = F.relu(self.d1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.d2(x))
        x = F.relu(self.l3(x))
        # x = F.relu(self.d3(x))
        x = F.relu(self.l4(x))
        return F.log_softmax(self.l5(x), dim=1)
        # return self.l5(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    # 每次输入barch_idx个数据
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        # loss
        loss = F.nll_loss(output, target)
        loss.backward()
        # update
        optimizer.step()
        # if batch_idx % 200 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def val():
    test_loss = 0
    correct = 0

    t, p = np.array([]), np.array([])
    # 测试集
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]

        t = np.hstack((t, target))
        p = np.hstack((p, pred.reshape(-1).numpy()))

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    return t.tolist(), p.tolist()


fs = []
for epoch in range(1, 3000):
    train(epoch)
    t, p = val()
    report = print_precison_recall_f1(t, p)
    fs.append((epoch, report))
    if epoch % 50 == 0:
        print(epoch)
        print(sorted(fs, reverse=True, key=lambda z: z[1][0])[:3])

# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: Dataset&DataLoder.py
@date: 2023/9/9 16:51
@target: 学会使用 Pytorch 中的数据集加载模块 Dataset 和 Dataloader
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


# 定义收入数据集加载器
class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.length = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


# 定义逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.Linear1 = nn.Linear(8, 4)
        self.Linear2 = nn.Linear(4, 2)
        self.Linear3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        out = self.sigmoid(self.Linear3(x))
        return out


# 定义训练函数
def train(epochs, dataloader, model, loss_fn, optimizer):
    for step in range(epochs):
        step_loss = 0
        for data in dataloader:
            inputs, labels = data
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                step_loss += loss.item()
        if step % 20 == 0:
            print("Epoch: {:}, loss: {:.4f}.".format(step, step_loss))


# 定义预测函数
def predict(model, dataloader):
    total_label = []
    total_pred = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            preds = model(inputs)
            total_label.extend(labels.numpy().flatten())
            total_pred.extend(preds.numpy().flatten())

    return total_label, total_pred


# 定义评估指标
def eval(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


if __name__ == "__main__":
    # 准备数据
    train_path = "D:/python/data/Pytorch/diabetes_train.csv"
    test_path = "D:/python/data/Pytorch/diabetes_test.csv"
    train_set = DiabetesDataset(train_path)
    test_set = DiabetesDataset(test_path)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)
    # 实例化模型
    model = LogisticRegressionModel()
    # 定义损失和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
    # 训练模型
    epoch = 100
    train(epoch, train_loader, model, criterion, optimizer)
    # 预测数据
    y_true, y_pred = predict(model, test_loader)
    # 计算指标
    auc = eval(y_true, y_pred)
    print("AUC: {:.4f}".format(auc))

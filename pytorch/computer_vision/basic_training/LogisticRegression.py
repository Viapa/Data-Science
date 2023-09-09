# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: LogisticRegression.py
@date: 2023/9/9 11:52
@target: 使用 Pytorch 实现一个简单的逻辑回归任务
"""

import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score


# 定义逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.Linear = nn.Linear(1, 1, bias=True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Linear(x)
        out = self.Sigmoid(x)
        return out


# 定义训练函数
def train(epochs, x_data, y_data):
    for i in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        if i % 50 == 0:
            print("Epoch: {:}, loss: {:.4f}.".format(i, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 定义预测函数
def predict(x_test):
    with torch.no_grad():
        y_test = model(x_test)
    return y_test


# 定义评估指标
def eval(y_true, y_pred):
    acc = roc_auc_score(y_true, y_pred)
    return acc


if __name__ == "__main__":
    # 准备数据
    x_train = torch.Tensor([[10.0], [11.5], [15.8], [19.6], [25.8]])
    y_train = torch.Tensor([[0], [0], [1], [1], [1]])
    x_test = torch.Tensor([[5.9], [14.8], [16.2], [32.5]])
    y_test = torch.Tensor([[0], [0], [1], [1]])
    # 实例化模型
    model = LogisticRegressionModel()
    # 定义损失和优化器
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1E-4)
    # 训练模型
    epoch = 1000
    train(epoch, x_train, y_train)
    # 预测数据
    y_pred = predict(x_test)
    # 计算指标
    print("y_test: ", y_test)
    print("y_pred: ", y_pred)
    auc = eval(y_test.numpy(), y_pred.numpy())
    print("AUC: {:.4f}".format(auc))
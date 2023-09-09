# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: LinearRegression.py
@date: 2023/9/9 10:16
@target: 使用 Pytorch 实现一个简单的线性回归任务
"""

import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_squared_error


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.Linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        out = self.Linear(x)
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

    print("w = {:.6f}".format(model.Linear.weight.item()))
    print("b = {:.6f}".format(model.Linear.bias.item()))


# 定义预测函数
def predict(x_test):
    with torch.no_grad():
        y_test = model(x_test)
    return y_test


# 定义评估指标
def eval(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


if __name__ == "__main__":
    # 准备数据
    x_train = torch.Tensor([[10.0], [11.5], [15.8], [19.6], [25.8]])
    y_train = torch.Tensor([[20.2], [23.3], [31.7], [39.9], [51.2]])
    x_test = torch.Tensor([[5.9], [32.5]])
    y_test = torch.Tensor([[11.8], [65.0]])
    # 实例化模型
    model = LinearModel()
    # 定义损失和优化器
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1E-4)
    # 训练模型
    epoch = 1000
    train(epoch, x_train, y_train)
    # 预测数据
    y_pred = predict(x_test)
    # 计算指标
    print("y_test: ", y_test)
    print("y_pred: ", y_pred)
    rmse = eval(y_test.numpy(), y_pred.numpy())
    print("RMSE: {:.4f}".format(rmse))
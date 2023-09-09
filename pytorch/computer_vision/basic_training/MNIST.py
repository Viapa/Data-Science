# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: MNIST.py
@date: 2023/9/9 12:50
@target: 使用多层感知机模型(MLP)对MNIST数据集进行多分类任务
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1 * 28 * 28, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, inp):
        x = inp.view(-1, 1 * 28 * 28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        out = self.linear4(x)
        return out


# 定义训练函数
def train(epochs, dataloader, model, loss_fn, optimizer):
    for step in range(epochs):
        step_loss = 0.0
        step_correct = 0.0
        for data in dataloader:
            images, labels = data
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                step_loss += loss.item()
                step_correct += (logits.argmax(dim=1) == labels).float().sum().item()
        if step % 5 == 0:
            mean_loss = step_loss / len(dataloader)
            mean_accuracy = step_correct / len(dataloader.dataset)
            print("Epoch: {:}, loss: {:.4f}, accuracy: {:.4f}".format(step, mean_loss, mean_accuracy))


# 定义预测函数
def predict(model, dataloader):
    total_label = []
    total_pred = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            logits = model(images)
            preds = logits.argmax(dim=1)
            total_label.extend(labels.numpy())
            total_pred.extend(preds.numpy())

    return total_label, total_pred


# 定义评估指标
def eval(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    # 准备数据 (下载MINIST数据集 + 图像处理)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    save_path = "D:/python/data/Pytorch/MINIST/"
    train_set = datasets.MNIST(root=save_path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=save_path, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)
    # 实例化模型
    model = MNISTModel()
    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5E-3)
    # 训练模型
    epoch = 50
    train(epoch, train_loader, model, criterion, optimizer)
    # 预测数据
    y_true, y_pred = predict(model, test_loader)
    # 计算指标
    acc = eval(y_true, y_pred)
    print("Accuracy: {:.4f}".format(acc))

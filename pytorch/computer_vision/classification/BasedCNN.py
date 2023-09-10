# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: BasedCNN.py
@date: 2023/9/10 9:35
@target: 学会在 Pytorch 中定义基本的CNN模型, 了解卷积层Conv和池化层Pooling, 并将数据转移到GPU中进行训练
"""


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score


class WeatherModel(nn.Module):
    def __init__(self):
        super(WeatherModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64*12*12, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=len(idx_to_classes))
        self.relu = nn.ReLU()
        self.dropout2d = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.pooling(self.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pooling(self.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pooling(self.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.dropout2d(x)
        batch = x.size(0)
        x = x.view(batch, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out


# 定义训练函数
def train(epochs, dataloader, model, loss_fn, optimizer):
    for step in range(epochs):
        step_loss = 0.0
        step_correct = 0.0
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
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
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total_label.extend(labels.cpu().numpy())
            total_pred.extend(preds.cpu().numpy())

    return total_label, total_pred


# 定义评估指标
def eval(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    # 定义数据增强
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 数据准备（彩色天气数据集）
    train_dir = "D:/python/data/Pytorch/weather/train/"
    test_dir = "D:/python/data/Pytorch/weather/test/"
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)
    idx_to_classes = dict((v, k) for k, v in train_set.class_to_idx.items())
    print("Number of labels:", len(idx_to_classes))
    # 确认机器设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used device:", device)
    # 实例化模型
    model = WeatherModel()
    model = model.to(device)
    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5E-4)
    # 训练模型
    epoch = 50
    train(epoch, train_loader, model, criterion, optimizer)
    # 预测数据
    y_true, y_pred = predict(model, test_loader)
    # 计算指标
    acc = eval(y_true, y_pred)
    print("Accuracy: {:.4f}".format(acc))

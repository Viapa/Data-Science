# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: VGG16.py
@date: 2023/9/10 11:43
@target: 学会封装CNN定义中的基础模块, 减少重复代码。以VGG16为例, 对天气数据集进行分类训练
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1)
                              )
        self.use_pooling = pooling
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if not self.use_pooling:
            return x
        else:
            return self.pooling(x)


class LinearBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_dims, out_dims, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 全局池化层
        self.avgpooling = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # 卷积层
        self.features = nn.Sequential(
            ConvBlock(3, 64, False),
            ConvBlock(64, 64, True),
            ConvBlock(64, 128, False),
            ConvBlock(128, 128, True),
            ConvBlock(128, 256, False),
            ConvBlock(256, 256, False),
            ConvBlock(256, 256, True),
            ConvBlock(256, 512, False),
            ConvBlock(512, 512, False),
            ConvBlock(512, 512, True),
            ConvBlock(512, 512, False),
            ConvBlock(512, 512, False),
            ConvBlock(512, 512, True)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            LinearBlock(512*7*7, 4096),
            LinearBlock(4096, 4096),
            nn.Linear(in_features=4096, out_features=len(idx_to_classes))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpooling(x)
        x = x.view(-1, 512*7*7)
        out = self.classifier(x)
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
    model = VGG16()
    model = model.to(device)
    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
    # 训练模型
    epoch = 50
    train(epoch, train_loader, model, criterion, optimizer)
    # 预测数据
    y_true, y_pred = predict(model, test_loader)
    # 计算指标
    acc = eval(y_true, y_pred)
    print("Accuracy: {:.4f}".format(acc))

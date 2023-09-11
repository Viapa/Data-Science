# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: Inception.py
@date: 2023/9/10 22:21
@target: 学会封装式定义Inception-A网络, 并打印学习曲线和保存最佳模型
"""

import copy
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score


class InceptionABlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionABlock, self).__init__()
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=(1, 1), padding=0)
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 1), padding=0)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 1), padding=0)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 1), padding=0)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=(3, 3), stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        outputs = [branch_pool, branch1x1, branch5x5, branch3x3]
        outputs = torch.cat(outputs, dim=1)
        return outputs


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


class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(16+24+24+24, 20, kernel_size=(5, 5))
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU(inplace=True)
        # 卷积层
        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.pooling,
            InceptionABlock(10),
            self.conv2,
            self.relu,
            self.pooling,
            InceptionABlock(20)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            LinearBlock(88*21*21, 4096),
            LinearBlock(4096, 1024),
            nn.Linear(in_features=1024, out_features=len(idx_to_classes))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 88*21*21)
        out = self.classifier(x)
        return out


# 定义训练函数
def train(epochs, train_loader, valid_loader, model, loss_fn, optimizer):
    best_score = 0.1
    best_weights = copy.deepcopy(model.state_dict())
    for step in range(epochs):
        step_loss = 0.0
        step_correct = 0.0
        valid_correct = 0.0
        for data in train_loader:
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
        mean_loss = step_loss / len(train_loader)
        mean_accuracy = step_correct / len(train_loader.dataset)
        print("Epoch: {:}, train_loss: {:.4f}, train_acc: {:.4f}".format(step, mean_loss, mean_accuracy))
        # 计算每个epoch在验证集的准确率, 保留最佳分数对应的模型权重
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                valid_correct += (logits.argmax(dim=1) == labels).float().sum().item()
            valid_accuracy = valid_correct / len(valid_loader.dataset)
            print("valid_acc: {:.4f}".format(valid_accuracy))
            if valid_accuracy > best_score:
                best_weights = copy.deepcopy(model.state_dict())
                best_score = valid_accuracy
                print("Save best weights...")

    return best_weights


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
        transforms.Resize((96, 96)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
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
    model = InceptionNet()
    # 测试模型
    img, label = next(iter(train_loader))
    print(model(img).shape)
    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
    # 训练模型
    model = model.to(device)
    epoch = 50
    best_weights = train(epoch, train_loader, test_loader, model, criterion, optimizer)
    # 预测数据
    model.load_state_dict(best_weights)
    y_true, y_pred = predict(model, test_loader)
    # 计算指标
    acc = eval(y_true, y_pred)
    print("Accuracy: {:.4f}".format(acc))

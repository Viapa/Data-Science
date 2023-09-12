# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: ResNet.py
@date: 2023/9/10 22:26
@target: 学会加载Images数据集预训练好的ResNet50模型, 对其结构进行微调后训练
"""


import copy
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score


# 加载预训练模型
def load_model(weights_file=None):
    if not weights_file:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(weights_file), strict=False)

    return model


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
    # 加载预训练模型
    weights_file = "C:/Users/86181/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
    model = load_model(weights_file)
    # 修改FC层的输出维度
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_classes))
    # 释放卷积层最后部分和全连接层部分的训练参数
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    # 测试模型
    img, label = next(iter(train_loader))
    print(model(img).shape)
    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
    # 训练模型
    model = model.to(device)
    epoch = 20
    best_weight = train(epoch, train_loader, test_loader, model, criterion, optimizer)
    # 预测数据
    model.load_state_dict(best_weight)
    y_true, y_pred = predict(model, test_loader)
    # 计算指标
    acc = eval(y_true, y_pred)
    print("Test accuracy: {:.4f}".format(acc))

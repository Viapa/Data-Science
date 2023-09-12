# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: DenseNet.py
@date: 2023/9/10 22:28
@target: 学会加载预训练DenseNet121模型, 并在训练中加入学习率控制函数, 对鸟类数据集进行分类
"""


import copy
import pandas as pd
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score





if __name__ == "__main__":
    # 定义数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 数据准备（彩色鸟类数据集, 共400种）
    train_dir = "D:/python/data/Pytorch/birds400/train/"
    valid_dir = "D:/python/data/Pytorch/birds400/valid/"
    test_dir = "D:/python/data/Pytorch/birds400/test/"
    class_dir = "D:/python/data/Pytorch/birds400/class_dict.csv"
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_set = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)
    class_df = pd.read_csv(class_dir, usecols=['class_index', 'class'])
    idx_to_classes = dict((row['class_index'], row['class']) for idx, row in class_df.iterrows())
    print("Number of labels:", len(idx_to_classes))
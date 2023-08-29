# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: MajorMetrics.py
@date: 2023/8/29 23:50
@target: 定义一些主要的增量评估指标
"""

import time
import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


# 1. uplift by deciles graph
def uplift_by_deciles_graph(df, is_plot):
    """
    计算增量十分位数对比图。计算方式如下：
    1. 根据 uplift model 计算干预组T和对照组C中所有样本的 uplift 分数（uplift = P(Y|X,T=1) - P(Y|X,T=0)）
    2. 对 uplift 分数进行由高到低排序
    3. 将排序后的样本按等频（预估分）切分为10个组, 找到每一组的边界值b0, b1, ..., b10
    4. 计算每个bin的平均预估因果效应 predict-uplift, u(k,p) = (∑x:bk-1<u(x)<=bk u(x))/ (ntk + nck)
    其中, u(x) 表示 uplift, u(k,p) 表示第k个bin的平均预估因果效应, ntk 表示第k个bin中T组的样本数, nck 表示第k个bin中C组的样本数
    5. 计算每个bin的平均真实因果效应 true-uplift, u(k,a) = (rtk) / (ntk) - (rck) / (nck)
    其中, u(k,a) 表示第k个bin的平均真实因果效应, rtk 表示第k个bin中T组里Y=1的样本数, rck 表示第k个bin中C组里Y=1的样本数
    6. 根据每个bin的 predict-uplift 和 true-uplift 绘制出增量十分位对比图
    :param df: pd.DataFrame
    :param is_plot: Bool
    :return: List[Float], List[Float]
    """

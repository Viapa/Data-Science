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
def uplift_by_deciles_graph(df, k=10):
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
    :param k: Int
    :return: List[[Float]]
    """
    df_ = df.copy()
    # 对数据按uplift降序排序
    df_.sort_values(by='uplift', ascending=False, inplace=True)
    # 统计uplift的十分位数
    quantiles = [df_.uplift.quantile(q=i/k) for i in range(k)]
    quantiles = quantiles[::-1]
    # 建立结果列表
    deciles = [[0 for j in range(2)] for i in range(k)]
    # 遍历每个bin
    for i in range(k):
        # 获取当前bin的数据
        if i == 0:
            bin_sample = df_.query(f'{quantiles[i]} < uplift')
        else:
            bin_sample = df_.query(f'{quantiles[i]} < uplift <= {quantiles[i-1]}')
        n_t_k = len(bin_sample.query('groupid == 1'))
        n_c_k = len(bin_sample.query('groupid == 0'))
        r_t_k = len(bin_sample.query('groupid == 1 and label == 1'))
        r_c_k = len(bin_sample.query('groupid == 0 and label == 1'))
        # 计算预估平均因果效应
        predict_uplift = sum(bin_sample.uplift) / (n_t_k + n_c_k)
        # 计算真实平均因果效应
        true_uplift = 0
        if n_t_k != 0 and n_c_k != 0:
            true_uplift = (r_t_k / n_t_k) - (r_c_k / n_c_k)
        # 添加该组结果
        deciles[i][0] = predict_uplift
        deciles[i][1] = true_uplift

    return deciles


def plot_uplift_deciles_graph(deciles, k=10):
    """
    绘制增量十分位数对比图
    :param deciles: List[[Float]]
    :param k: Int
    :return: None
    """
    # 设置绘图风格
    config = {
        'font.family': 'serif',
        'font.size': 18,
        'font.style': 'normal',
        'font.weight': 'normal',
        'font.serif': ['cmb10'],
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False
    }
    plt.rcParams.update(config)
    plt.style.use('fivethirtyeight')

    # 设置绘图参数
    graph_size = (12, 7)
    total_width = 0.8
    legend_num = 2
    bar_width = total_width / legend_num
    # 开始绘图
    plt.figure(figsize=graph_size)
    plt.bar([i for i in range(k)], height=[x[0] for x in deciles], label='predicted', width=bar_width)
    plt.bar([i + bar_width for i in range(k)], height=[x[1] for x in deciles], label='actual', width=bar_width)
    plt.xticks([i + bar_width / 2 * (legend_num -1 ) for i in range(k)],
               [str(idx * k) + '%' for idx in range(1, k+1)])
    plt.xlabel('Bins')
    plt.ylabel('Uplift')
    plt.title('Uplift by Deciles Graph')
    plt.legend(fontsize=16)
    plt.show()

    return None

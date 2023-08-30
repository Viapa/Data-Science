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


# 1. Uplift by deciles graph
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
    plt.xticks([i + bar_width / 2 * (legend_num - 1) for i in range(k)],
               [str(idx * k) + '%' for idx in range(1, k+1)])
    plt.xlabel('Bins')
    plt.ylabel('Uplift')
    plt.title('Uplift by Deciles Graph')
    plt.legend(fontsize=16)
    plt.show()

    return None


# 2. Cumulative gain
def cumulative_gain(df, k=10):
    """
    计算累积增益图
    1. 根据 uplift model 计算干预组T和对照组C中所有样本的 uplift 分数（uplift = P(Y|X,T=1) - P(Y|X,T=0)）
    2. 对 uplift 分数进行由高到低排序
    3. 将排序后的样本按等频（预估分）切分为10个组, 找到每一组的边界值b0, b1, ..., b10
    4. 计算TopK个bin的平均累积因果效应 cumulative-gain, u(k,a) = ((Rtk) / (Ntk) - (Rck) / (Nck)) * (Ntk + Nck)
    其中, u(k,a) 表示从第1个到第k个bin的平均累积真实因果效应, Rtk 表示第1到k个bin中T组里Y=1的样本数, Ntk 表示1到k个bin中T组的样本数,
    Rck 表示第1到k个bin中C组里Y=1的样本数, Nck 表示1到k个bin中C组的样本数
    6. 根据TopK个bin的 cumulative-gain 绘制出累积增益图
    :param df: pd.DataFrame
    :param k: Int
    :return: List[Float]
    """
    df_ = df.copy()
    # 对数据按uplift降序排序
    df_.sort_values(by='uplift', ascending=False, inplace=True)
    # 统计uplift的十分位数
    quantiles = [df_.uplift.quantile(q=i / k) for i in range(k)]
    quantiles = quantiles[::-1]
    # 建立结果列表
    gains = [0 for i in range(k)]
    for i in range(k):
        # 划分截止TopK个bin中的数据
        bin_sample = df_.query(f'{quantiles[i]} <= uplift')
        N_t_k = len(bin_sample.query('groupid == 1'))
        N_c_k = len(bin_sample.query('groupid == 0'))
        R_t_k = len(bin_sample.query('groupid == 1 and label == 1'))
        R_c_k = len(bin_sample.query('groupid == 0 and label == 1'))
        # 计算平均累积增益值
        cumsum_gain = 0
        if N_t_k != 0 and N_c_k != 0:
            cumsum_gain = (R_t_k / N_t_k - R_c_k / N_c_k) * (N_t_k + N_c_k)
        # 添加当前结果
        gains[i] = cumsum_gain

    return gains


def plot_cumulative_gain(gains, k=10):
    """
    :param gains: List[Float]
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
    graph_size = (10, 6)
    total_width = 0.8
    legend_num = 2
    bar_width = total_width / legend_num
    # 开始绘图
    plt.figure(figsize=graph_size)
    plt.bar([i for i in range(k)], height=[x for x in gains], width=bar_width, color='steelblue')
    plt.xticks([i for i in range(k)], [idx for idx in range(1, k+1)])
    plt.xlabel('Percentile')
    plt.ylabel('Mean(uplift)')
    plt.title('Cumulative Gain')
    plt.show()

    return None


# 3. Uplift cruve & AUUC
def uplift_cruve_with_auuc(df):
    """
    计算增量曲线和auuc值
    1. 基于 cumulative_gain 的计算方式, 将数据集继续细分, 计算截止前x个样本的累积增益值
    f(x) = ((Rtx) / (Ntx) - (Rcx) / (Ncx)) * (Ntx + Ncx)
    2. 根据样本数和对应的f(x)值绘制uplift曲线, 横坐标为样本量, 纵坐标为累积增益
    3. 根据uplift曲线和随机曲线的面积之差求得auuc值
    :param df: pd.DataFrame
    :return: Int, List[Float], float
    """
    df_ = df.copy()
    # 对数据按uplift降序排序
    df_.sort_values(by='uplift', ascending=False, inplace=True)
    # 将数据转换为numpy格式
    arr = df_[['label', 'groupid']].to_numpy()
    # 计算总样本数
    n = len(df_)
    # 建立结果列表
    gains = []
    # 初始化变量
    N_t_x, N_c_x = 0, 0
    R_t_x, R_c_x = 0, 0
    # 遍历每个元素
    for row in arr:
        label, group_id = row[0], row[1]
        # 统计变量值
        if group_id == 1:
            N_t_x += 1
            if label == 1:
                R_t_x += 1
        else:
            N_c_x += 1
            if label == 1:
                R_c_x += 1
        # 计算累积到当前元素的增益值
        cumsum_gain = 0
        if N_t_x != 0 and N_c_x != 0:
            cumsum_gain = (R_t_x / N_t_x - R_c_x / N_c_x) * (N_t_x + N_c_x)
        # 添加当前结果
        gains.append(cumsum_gain)

    # 计算uplift-cruve的面积
    area_cruve = np.trapz(gains)
    # 计算random-cruve的面积
    area_random = n * gains[-1] / 2
    # 计算AUUC值
    auuc = round(area_cruve - area_random, 2)

    return n, gains, auuc


def plot_uplift_cruve(gains, n):
    """
    :param n: Int
    :param gains: List[Float]
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
    graph_size = (12, 8)
    plt.figure(figsize=graph_size)
    plt.plot([i for i in range(n)], gains, label='uplift-model', color='steelblue', linestyle='-')
    plt.plot([0, n], [0, gains[-1]], label='random', color='orange', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Mean(uplift)')
    plt.title('Uplift Cruve')
    plt.legend(fontsize=16)
    plt.show()

    return None


# 4. Qini cruve & Qini coefficient
def qini_cruve_with_coef(df):
    """
    计算qini曲线和qini系数
    1. 针对T组和C组的样本比例不平衡问题, 修改样本的累积增益计算公式为: g(x) = Rtx - Rcx * (Ntx / Ncx)
    f(x) 与 g(x) 的关系为： f(x) = g(x) * (Ntx + Ncx) / Ntx
    2. 根据样本数和对应的g(x)值绘制qini-uplift曲线, 横坐标为样本量, 纵坐标为修正的累积增益
    3. 按4种人群排序, 满足 P(Y=1|T=1) > P(Y=0|T=1) 以及 P(Y=0|T=0) > P(Y=1|T=0)
    4. 绘制完美曲线perfect-cruve
    5. 根据perfect-cruve和qini-uplift曲线与随机曲线的面积之差求得qini-coef的值
    :param df: pd.DataFrame
    :return: Int, List[Float], List[Float], float
    """
    df_ = df.copy()
    # 对数据按uplift降序排序
    df_.sort_values(by='uplift', ascending=False, inplace=True)
    # 将数据转换为numpy格式
    arr = df_[['label', 'groupid']].to_numpy()
    # 计算总样本数
    n = len(df_)
    # 建立结果列表
    gains = []
    # 初始化变量
    N_t_x, N_c_x = 0, 0
    R_t_x, R_c_x = 0, 0
    # 遍历每个元素
    for row in arr:
        label, group_id = row[0], row[1]
        # 统计变量值
        if group_id == 1:
            N_t_x += 1
            if label == 1:
                R_t_x += 1
        else:
            N_c_x += 1
            if label == 1:
                R_c_x += 1
        # 计算累积到当前元素的增益值
        cumsum_gain = 0
        if N_c_x != 0:
            cumsum_gain = R_t_x - R_c_x * (N_t_x / N_c_x)
        # 添加当前结果
        gains.append(cumsum_gain)

    # 定义完美增量样本
    df_treat = df_.query('groupid == 1').sort_values(by='label', ascending=False)
    df_control = df_.query('groupid == 0').sort_values(by='label', ascending=True)
    perfect_arr = df_treat.append(df_control)[['label', 'groupid']].to_numpy()
    # 建立完美结果列表
    perfect_gains = []
    # 初始化变量
    N_t_x, N_c_x = 0, 0
    R_t_x, R_c_x = 0, 0
    # 遍历每个元素
    for row in perfect_arr:
        label, group_id = row[0], row[1]
        # 统计变量值
        if group_id == 1:
            N_t_x += 1
            if label == 1:
                R_t_x += 1
        else:
            N_c_x += 1
            if label == 1:
                R_c_x += 1
        # 计算累积到当前元素的增益值
        if N_c_x != 0:
            cumsum_gain = R_t_x - R_c_x * (N_t_x / N_c_x)
        else:
            cumsum_gain = R_t_x
        # 添加当前结果
        perfect_gains.append(cumsum_gain)

    # 计算qini-cruve的面积
    area_cruve = np.trapz(gains)
    # 计算perfect-cruve的面积
    area_perfect = np.trapz(perfect_gains)
    # 计算random-cruve的面积
    area_random = n * gains[-1] / 2
    # 计算qini-coefficient值
    qini_coef = round((area_cruve - area_random) / (area_perfect - area_random), 4)

    return n, gains, perfect_gains, qini_coef


def plot_qini_cruve(gains, perfect_gains, n):
    """
    :param gains: List[Float]
    :param perfect_gains: List[Float]
    :param n: Int
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
    graph_size = (12, 8)
    plt.figure(figsize=graph_size)
    plt.plot([i for i in range(n)], perfect_gains, label='perfect', color='limegreen', linestyle='-')
    plt.plot([i for i in range(n)], gains, label='uplift-model', color='steelblue', linestyle='-')
    plt.plot([0, n], [0, gains[-1]], label='random', color='orange', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Mean(uplift)')
    plt.title('Qini Cruve')
    plt.legend(fontsize=16)
    plt.show()

    return None


# 测试入口
def main():
    df = pd.read_csv('xxxxx.csv')
    # 1. Uplift by deciles graph
    res = uplift_by_deciles_graph(df)
    plot_uplift_deciles_graph(res)
    # 2. Cumulative gain
    res = cumulative_gain(df)
    plot_cumulative_gain(res)
    # 3. Uplift cruve & AUUC
    n, res, auuc = uplift_cruve_with_auuc(df)
    plot_uplift_cruve(res, n)
    # 4. Qini cruve & Qini coefficient
    n, gains, perfect_gains, qini_coef = qini_cruve_with_coef(df)
    plot_qini_cruve(gains, perfect_gains, n)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: Constant.py
@date: 2023/9/2 14:34
@target: 用于存放一些定义的常量
"""

class CONFIG(object):
    # 与数据相关的常量
    HIVE_TABLE = "mart_grocerygrowth.xxxx"
    DATE_TABLE = "dw.xxxx"
    FUTURE_DATE_FILE = "future_date.csv"
    TARGET_CITY = "深圳"
    TRAIN_BEGIN = "20210101"
    TRAIN_END = "20230228"
    VALID_BEGIN = "20230301"
    VALID_END = "20230331"
    PRED_BEGIN = "20230401"
    PRED_END = "20230430"
    SIMILAR_K = 10
    SCALE_ORDER = 1e6
    LOOKBACK = 7
    # 与模型相关的常量
    BATCH_SIZE = 16
    EPOCH = 50
    HIDDEN_SIZE = 32
    # 与绘图相关的常量
    PLOT_STYLE = "fivethirtyeight"
    PLOT_CONFIG = {
        'font.size': 16,
        'font.style': 'normal',
        'font.weight': 'normal',
        'font.serif': ['cmb10'],
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False
    }
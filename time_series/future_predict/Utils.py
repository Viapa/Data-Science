# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: Utils.py
@date: 2023/9/2 9:17
@target: 定义一些时序预估中的工具类和函数
"""

import pickle
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model


def read_hive_to_df(spark, table_name, cols=None, conditions=None):
    """
    使用spark读取hive表数据, 然后将其转换为dataframe格式
    """
    # 选择查询字段
    if not cols:
        column = '*'
    else:
        column = ','.join(cols)
    # 选择查询条件
    if not conditions:
        conditions = '1=1'
    # SQL查询并返回
    df = spark.sql(
        f"""
        select {column}
        from {table_name}
        where {conditions}
        """
    ).toPandas()

    return df


def find_similar_city(df, target_city, k=10):
    """
    基于皮尔森相关性计算目标城市与其他城市在销售额上的相关性
    其中, 需要同时考虑销售额绝对值和量级的差异
    最后返回topk个最相关的城市名称
    """
    # 获取目标城市的销售额
    group = df.groupby('city')
    sales = group.get_group(target_city)['sale_amt']

    # 计算各城市与目标城市的相关性
    def cal_corr(x, sale):
        sale_amt = x['sale_amt'].corr(sale, min_period=10) + 1
        sale_order = abs(np.log10(x['sale_amt'].mean() - np.log10(sale.mean()))) + 1
        res = sale_amt / sale_order
        return res

    corr = group.apply(lambda x: cal_corr(x, sales))
    top_k = corr.rank(ascending=False).sort_values().head(k + 1)
    top_k = [i for i in top_k.index.tolist() if i != target_city]

    return top_k


def data_split(df, train_start, train_end, valid_start, valid_end, target_city):
    """
    根据给定训练、验证的日期, 划分训练集和测试集样本
    """
    df_train = df.query(f"df >= '{train_start}') & (dt <= '{train_end}')")
    df_valid = df.query(f"df >= '{valid_start}') & (dt <= '{valid_end}')")
    train_y = df_train[target_city].values
    valid_y = df_train[target_city].values
    drop_cols = ['dt', target_city]
    train_X = df_train.drop(drop_cols, axis=1).values
    valid_X = df_valid.drop(drop_cols, axis=1).values

    return df_train, df_valid, train_X, train_y, valid_X, valid_y


def create_time_series_data(df, target_city):
    """
    构建用于时序模型训练的数据格式
    """
    time_data = df[target_city].to_frame().set_index(df.dt)
    time_data.index = pd.to_datetime(time_data.index, format='%Y%m%d')
    time_data.index.freq = 'D'

    return time_data


def datekey_to_date(datekey_str):
    """
    将String类型, 格式为'%Y%m%d'的日期变量转换为'%Y-%m-%d'
    """
    date = datetime.datetime.strptime(datekey_str, '%Y%m%d')
    date_str = date.strftime('%Y-%m-%d')

    return date_str


def holiday_info():
    """
    手动创建中国节假日的信息表, 返回dataframe格式
    """
    new_year = pd.DataFrame({
            'holiday': 'new_year',
            'ds': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
            'lower_window': -1,
            'upper_window': 1
    })
    spring_festival = pd.DataFrame({
            'holiday': 'spring_festival',
            'ds': pd.to_datetime(['2021-02-11', '2022-01-31', '2023-01-22']),
            'lower_window': -1,
            'upper_window': 5
    })

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
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model
from pyspark import SparkConf
from pyspark.sql import SparkSession


def build_saprk():
    """
    启动spark环境
    """
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    print("Spark application id: ", spark.sparkContext.applicationId)

    return spark


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


def data_split(df, train_begin, train_end, valid_begin, valid_end, target_city):
    """
    根据给定训练、验证的日期, 划分训练集和测试集样本
    """
    df_train = df.query(f"df >= '{train_begin}') & (dt <= '{train_end}')")
    df_valid = df.query(f"df >= '{valid_begin}') & (dt <= '{valid_end}')")
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


def create_sequential_data(array, lookback):
    """
    创建用于LSTM等模型的序列格式数据
    """
    n_rows = len(array)
    X, Y = [], []
    for i in range(lookback, n_rows):
        X.append(array[i - lookback: i, :])
        Y.append((array[i, 0]))

    return np.array(X), np.array(Y)


def datekey_to_date(datekey_str):
    """
    将String类型, 格式为'%Y%m%d'的日期变量转换为'%Y-%m-%d'
    """
    date = datetime.datetime.strptime(datekey_str, '%Y%m%d')
    date_str = date.strftime('%Y-%m-%d')

    return date_str


def rmse(y_true, y_pred):
    """
    计算真实样本和预估样本之间的RMSE指标
    """
    mse = mean_squared_error(y_true, y_pred)

    return np.sqrt(mse)


def mape(y_true, y_pred):
    """
    计算真实样本和预估样本之间的MAPE指标
    """
    ape = np.abs((y_true - y_pred) / y_true)

    return np.mean(ape) * 100


def r2(y_true, y_pred):
    """
    计算真实样本和预估样本之间的R2-Score指标
    """
    return r2_score(y_true, y_pred)


def saves_model(model, model_name, model_type='ml'):
    """
    保存ML或NN格式的模型文件
    """
    if model_type == 'ml':
        with open(model_name + '.pkl', 'wb') as f:
            pickle.dump(model, f)
    elif model_type == 'nn':
        model.save(model_name + '.h5')
    else:
        raise ValueError('Unsupported model file format!')

    print('Model has been saved.')

    return None


def loads_model(model_file, model_type='ml'):
    """
    加载ML或NN格式的模型文件
    """
    if model_type == 'ml':
        with open(model_file + '.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_type == 'nn':
        model = load_model(model_file + '.h5')
    else:
        raise ValueError('Unsupported model file format!')

    print('Model has been loaded.')

    return model


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
            'lower_window': -2,
            'upper_window': 5
    })
    tomb_sweep = pd.DataFrame({
            'holiday': 'tomb_sweep',
            'ds': pd.to_datetime(['2021-04-03', '2022-04-05', '2023-04-05']),
            'lower_window': -1,
            'upper_window': 1
    })
    labor_day = pd.DataFrame({
            'holiday': 'labor_day',
            'ds': pd.to_datetime(['2021-05-01', '2022-05-01', '2023-05-01']),
            'lower_window': -1,
            'upper_window': 3
    })
    dragon_boat = pd.DataFrame({
            'holiday': 'dragon_boat',
            'ds': pd.to_datetime(['2021-06-12', '2022-06-03', '2023-06-22']),
            'lower_window': -1,
            'upper_window': 1
    })
    mid_autumn = pd.DataFrame({
            'holiday': 'mid_autumn',
            'ds': pd.to_datetime(['2021-09-19', '2022-09-10', '2023-09-29']),
            'lower_window': -1,
            'upper_window': 1
    })
    national_day = pd.DataFrame({
            'holiday': 'national_day',
            'ds': pd.to_datetime(['2021-10-01', '2022-10-01', '2023-10-01']),
            'lower_window': -2,
            'upper_window': 5
    })

    df_holiday = pd.concat((
        new_year, spring_festival, tomb_sweep,
        labor_day, dragon_boat, mid_autumn, national_day
    ))

    return df_holiday


def plot_series_cruve(series, city, tag):
    """
    绘制时序趋势曲线图
    """
    plt.figure(figsize=(12, 6), dpi=100)
    series.plot()
    plt.title(f"{tag} for {city}")
    plt.xlabel('Date')
    plt.ylabel(f'{tag}')
    plt.show()

    return None


def plot_similar_series(df, target_city, topk_ls):
    """
    绘制相似topk个序列的对比曲线图
    """
    group = df.groupby('city')
    target_group = group.get_group(target_city)
    target_sales = target_group['sale_amt']
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(target_group.index, target_sales, color='blue', linestyle='--', label=target_city, linewidth=3)
    for city in topk_ls:
        city_group = group.get_group(city)
        city_sales = city_group['sale_amt']
        ax.plot(city_group.index, city_sales, linestyle='-', label=f'{city}', linewidth=2, alpha=0.9)
    ax.set_title('TopK Similar Series of City')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend(loc='best')

    return None


def plot_pred_error_cruve(begin_dt, end_dt, y_true, y_pred):
    """
    绘制真实与预估的误差曲线图
    """
    date_range = pd.date_range(begin_dt, end_dt)
    date_index = date_range.strftime('%Y%m%d')
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(date_index, y_true, label='Actual', linestyle='-', linewidth=3)
    plt.plot(date_index, y_pred, label='Predicted', linestyle='--', linewidth=3)
    plt.title('Predict Error Cruve')
    plt.xlabel('Date')
    plt.xticks(rotation=45, fontsize=15)
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    return None

# -*- coding: utf-8 -*-
"""
@project: Data-Science
@Author: via
@file: LSTM.py
@date: 2023/9/2 9:10
@target: 使用LSTM时序模型预估目标城市的未来销售额数据
"""


import tensorflow as tf
from tensorflow.keras import layers, optimizers

from Utils import *
from Constant import *


def lstm_pipline(df_merge, df_date, target_city,
                 valid_begin, pred_begin, pred_end,
                 scale, future_file, lookback,
                 epoch, batch_size, hidden_size):
    """
    定义LSTM模型训练、预测和评估的流程
    """
    # 数据准备
    lstm_data = df_merge[df_date.columns]
    lstm_data.insert(0, 'y', df_merge[target_city].values)
    test_num = len(lstm_data.query(f"dt >= '{valid_begin}'")) + lookback
    lstm_data.drop('dt', axis=1, inplace=True)
    train = lstm_data[: -test_num]
    valid = lstm_data[-test_num:]
    train.loc[:, 'y'] /= scale
    valid.loc[:, 'y'] /= scale
    train_X, train_y = create_sequential_data(train.values, lookback)
    valid_X, valid_y = create_sequential_data(valid.values, lookback)
    # 构建LSTM模型
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(hidden_size, return_sequences=True, input_shape=(valid_X.shape[1:])))
    model.add(layers.LSTM(hidden_size // 2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    # 模型编译与训练
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='mse')
    model.fit(
        train_X, train_y,
        epochs=epoch,
        batch_size=batch_size,
        validation_data=(valid_X, valid_y),
        verbose=1
    )
    # 模型预测与评估
    model_name = 'LSTM'
    pred = model.predict(valid_X).flatten()
    r2_eval = r2(valid_y * scale, pred * scale)
    rmse_eval = rmse(valid_y * scale, pred * scale)
    mape_eval = mape(valid_y * scale, pred * scale)
    print(f"模型名称: {model_name}, R2: {r2_eval:.4f}, RMSE: {rmse_eval:.4f}, MAPE: {mape_eval:.2f}%")
    # 对未来进行预测
    date_range = pd.date_range(start=pred_begin, end=pred_end, freq='D')
    n_pred_day = len(date_range)
    lstm_data_tail = lstm_data[-lookback:]
    future_date = pd.read_csv(future_file, parse_dates=['dt'], index_col=[0])
    future_date.insert(0, 'y', np.nan)
    df_future = pd.concat([lstm_data_tail, future_date]).reset_index(drop=True)
    future_pred = []
    for i in range(lookback, len(df_future)):
        X = [df_future.values[i - lookback: i, :]]
        ans = model.predict(np.array(X)).flatten().item()
        future_pred.append(ans)
        df_future.loc[i, 'y'] = ans
    future_pred = np.array(future_pred[-n_pred_day:])

    return model, valid_y*scale, pred*scale, future_pred*scale


# 定义主函数
def main():
    # 启动spark环境
    spark = build_saprk()
    # 1 数据探索与分析
    # 1.1 读取城市的销售额数据
    df_city = read_hive_to_df(spark, CONFIG.HIVE_TABLE, cols=['dt', 'city', 'sale_amt'])
    df_city['sale_amt'] = df_city['sale_amt'].astype(np.float64)
    df_city['date'] = pd.to_datetime(df_city['dt'], format='%Y%m%d')
    df_city = df_city.set_index('date')
    sale_series = df_city.query(f'city == {CONFIG.TARGET_CITY}')['sale_amt']
    # 1.2 绘制目标城市的销售额趋势图
    plot_series_cruve(sale_series, CONFIG.TARGET_CITY, 'Sales')
    # 1.3 计算城市相似度
    df_train_city = df_city[df_city.index < CONFIG.VALID_BEGIN]
    similar_cities = find_similar_city(df_train_city, CONFIG.TARGET_CITY, k=CONFIG.SIMILAR_K)
    # 1.4 绘制TopK个相似城市的销售额趋势图
    plot_similar_series(df_train_city, CONFIG.TARGET_CITY, similar_cities)
    # 2 数据预处理
    # 2.1 读取日期表的相关字段
    date_cols = ['date', 'day_of_week', 'day_of_month', 'day_of_year', 'week_of_month',
                 'week_of_year', 'month', 'quarter', 'is_weekend', 'is_work_day', 'is_holiday']
    datekey_begin = datekey_to_date(CONFIG.TRAIN_BEGIN)
    datekey_end = datekey_to_date(CONFIG.VALID_END)
    date_condition = f"date between '{datekey_begin}' and '{datekey_end}'"
    df_date = read_hive_to_df(spark, CONFIG.DATE_TABLE, date_cols, date_condition)
    df_date['date'] = pd.to_datetime(df_date['date'])
    df_date['dt'] = df_date['date'].dt.strftime('%Y%m%d')
    df_date.drop('date', aixs=1, inplace=True)
    # 2.2 获取TopK个相似城市的销售额信息
    similar_cities.append(CONFIG.TARGET_CITY)
    df_city_topk = df_city[df_city.city.isin(similar_cities)]
    df_city_topk = df_city_topk.pivot(index='dt', columns='city', values='sale_amt')
    # 2.3 对缺失销售额进行填充
    df_city_topk = df_city_topk.fillna(df_city_topk.mean())
    # 2.4 拼接日期表字段
    df_merge = pd.merge(df_city_topk, df_date, left_index=True, right_on='dt')
    # 3 模型训练和预测
    # 3.1 模型定义、训练和评估
    model, valid_y, valid_pred, future_pred = lstm_pipline(
        df_merge, df_date, CONFIG.TARGET_CITY, 
        CONFIG.VALID_BEGIN, CONFIG.PRED_BEGIN, CONFIG.PRED_END,
        CONFIG.SCALE_ORDER, CONFIG.FUTURE_DATE_FILE, CONFIG.LOOKBACK,
        CONFIG.EPOCH, CONFIG.BATCH_SIZE, CONFIG.HIDDEN_SIZE
    )
    # 3.2 绘制验证集误差曲线图
    plot_pred_error_cruve(CONFIG.VALID_BEGIN, CONFIG.VALID_END, valid_y, valid_pred)
    # 3.3 绘制预测未来销售额曲线图并打印
    plot_series_cruve(pd.Series(future_pred), CONFIG.TARGET_CITY, 'Future Sales')
    for i in future_pred:
        print(round(i, 6), end=',\n')

    return None


# 主程序入口
if __name__ == "__main__":
    main()

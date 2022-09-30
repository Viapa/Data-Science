# 计算uplift最佳的差值系数
def CalBestCoef(table_name, alpha_range=(0.0, 1.0), step=0.05):
    """
    用于计算uplift模型中，score1 - alpha * score0 中的最佳alpha值（基于auc表现）
    :param table_name: 利用SQL开发处理后的得分预测hive表，包含[user_id, label, score1, score0]字段
    :param alpha_range: 自定义alpha搜索范围
    :return:
    """
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    score_dict = {}
    for alpha in np.arange(alpha_range[0], alpha_range[1], step):
        data = spark.sql(
            f"""
            select user_id, label, (score1 - {alpha} * score0) as uplift_score
            from {table_name}
            """
        )
        Evaluator = BinaryClassificationEvaluator(labelCol="label",
                                                  rawPredictionCol="uplift_score",
                                                  metricName='areaUnderROC')
        auc = Evaluator.evaluate(data)
        score_dict[alpha] = auc

    res_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
    return res_dict


# uplift评估指标1-分位数图表
def EvalByDecilesGraph(exp_df, ctrl_df, k=10):
    """
    用于计算Uplift-decile-charts，通过观察柱状图正负情况，获得样本处于正效果的比例
    :param exp_df: 实验组的dataframe，包含user_id, uplift_score, label
    :param ctrl_df: 对照组的dataframe，包含user_id, uplift_score, label
    :param k: 按等比例划分成的组数bins
    :return: 包含bins和mean(uplift)的dataframe表
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    exp_df_desc = exp_df.sort_values(by='uplift_score', ascending=False)
    exp_score = exp_df_desc.uplift_score.values
    exp_group_num = len(exp_score) // k
    exp_score_arr = np.zeros(k, dtype=np.float32)
    idx = 0
    for i in range(k):
        group_mean = np.mean(exp_score[idx: idx + exp_group_num])
        exp_score_arr[i] = group_mean
        idx += exp_group_num + 1

    ctrl_df_desc = ctrl_df.sort_values(by='uplift_score', ascending=False)
    ctrl_score = ctrl_df_desc.uplift_score.values
    ctrl_group_num = len(ctrl_score) // k
    ctrl_score_arr = np.zeros(k, dtype=np.float32)
    idx = 0
    for i in range(k):
        group_mean = np.mean(ctrl_score[idx: idx + ctrl_group_num])
        ctrl_score_arr[i] = group_mean
        idx += ctrl_group_num + 1

    diff_mean_score = exp_score_arr - ctrl_score_arr

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(k), diff_mean_score, width=0.5)
    plt.title("S-model Uplift Decile approach")
    plt.xlabel("Bins")
    plt.ylabel("Mean of Uplift-Score")
    plt.show()

    return pd.DataFrame(diff_mean_score, columns=['mean_uplift'])
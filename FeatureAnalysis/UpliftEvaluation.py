import numpy as np
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 计算uplift最佳的差值系数
def CalBestCoef(table_name, alpha_range=(0.0, 1.0), step=0.05):
    """
    用于计算uplift模型中，score1 - alpha * score0 中的最佳alpha值（基于auc表现）
    :param table_name: 利用SQL开发处理后的得分预测hive表，包含[user_id, label, score1, score0]字段
    :param alpha_range: 自定义alpha搜索范围
    :return:
    """
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



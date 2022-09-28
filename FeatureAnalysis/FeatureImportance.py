import numpy as np
import pandas as pd

# 计算TopK个重要特征
def CalTopkFeatures(ls, k=10, printf=False):
    """
    用于计算xgb模型前k个重要性特征并返回dataframe
    :param ls: 输入xgb的特征重要度json列表
    :param k: 选取前k个重要度特征
    :param printf: 是否进行详细打印
    :return: 前k个重要特征的dataframe信息
    """
    feature_length = len(ls)
    topk_list = []
    for idx, row in enumerate(ls[: k]):
        feature_name = row["feature"]
        weight = row['weight']
        topk_list.append((idx + 1, feature_name, weight))
    topk_df = pd.DataFrame(topk_list, columns=['rank', 'name', 'weight'])

    if printf:
        print("Total feature number is: ", feature_length)
        for idx, row in topk_df.iterrows():
            print(row[0], row[1], end=',\n')

    return topk_df


# 计算某个特征的重要度
def CalOneFeature(ls, target_name):
    """
    用于计算xgb模型某个特征的重要性排名和比例
    :param ls: 输入xgb的特征重要度json列表
    :param target_name: 需要计算重要度的特征名
    :return: 无
    """
    feature_length = len(ls)
    is_contain = False
    for idx, row in enumerate(ls):
        feature_name = row["feature"]
        weight = row['weight']
        if feature_name == target_name:
            is_contain = True
            rank = idx + 1
            ratio = rank / feature_length * 100
            print("特征 {:s} 的重要性排序为 {:d} , 权重为 {:d} , 在 {:d} 个全部特征中的比例为 {:.2f}% .".format(target_name, rank, weight, feature_length, ratio))
            break
    if not is_contain:
        print(f"This feature: '{target_name}' is not in the feature list, which means its importance may be zero!")
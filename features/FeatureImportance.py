import numpy as np
import pandas as pd

# 计算TopK个重要特征
def calTopkFeatures(ls, k=10, printf=False):
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
def calOneFeature(ls, target_name):
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
            print("特征 '{:s}' 的重要性排序为 {:d} , 权重为 {:d} , 在 {:d} 个全部特征中的比例为 {:.2f}% .".format(target_name, rank, weight, feature_length, ratio))
            break
    if not is_contain:
        print(f"输入的特征: '{target_name}' 不在特征重要度列表中, 这可能意味着该特征的重要度可能为零!（没有在建树中进行切分）")


# 计算某批特征集的重要度排序
def calBatchFeatures(ls, target_list):
    """
    用于计算某个特征集内特征的重要度
    :param ls: 输入xgb的特征重要度json列表
    :param target_list: 需要计算重要度的特征集合
    :return: 空
    """
    target_lenth = len(target_list)
    contain_list = []
    print("输入特征集总数为: ", target_lenth)
    for idx, row in enumerate(ls):
        feature_name = row["feature"]
        weight = row['weight']
        rank = idx + 1
        if feature_name in target_list:
            print(f"特征名: '{feature_name}' , 排序: {rank} , 权重: {weight} .")
            contain_list.append(feature_name)

    zero_set = set(target_list) - set(contain_list)
    for i in zero_set:
        print(f"特征名: '{i}' , 排序: -1 , 权重: 0 .")
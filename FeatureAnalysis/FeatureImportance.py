import numpy as np
import pandas as pd


def CalTopkFeatures(ls, k=10, printf=False):
    feature_length = len(ls)
    topk_list = []
    for idx, row in enumerate(ls[: k]):
        feature_name = row["feature"]
        weight = row['weight']
        topk_list.append((idx + 1, feature_name, weight))
    topk_df = pd.DataFrame(topk_list, columns=['rank', 'name', 'weight'])

    if printf:
        for idx, row in topk_df.iterrows():
            print(row[0], row[1], end=',\n')
    return topk_df

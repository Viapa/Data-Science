# 基于AUC的alpha选取
def CalAlphaByAUC(table_name, left_dt, right_dt, alpha_range=(0.0, 1.05), step=0.05):
    """
    用于计算uplift模型中，score1 - alpha * score0 中的最佳alpha值（基于auc表现）
    :param table_name: 利用SQL开发处理后的得分预测hive表，至少包含[user_id, label, score1, score0]字段 [str]
    :param left_dt: 表开始日期 [str]
    :param right_dt: 表结束日期 [str]
    :param alpha_range: 自定义alpha搜索范围 [tuple]
    :param step: alpha值搜索步长 [float]
    :return: 不同alpha取值的结果字典 [dict[float, float]]
    """
    score_dict = dict()
    for alpha in tqdm(np.arange(alpha_range[0], alpha_range[1], step)):
        data = spark.sql(
            f"""
            select user_id, label, (score1 - {alpha} * score0) as uplift_score
            from {table_name}
            where dt between {left_dt} and {right_dt}
            """
        )

        Evaluator = BinaryClassificationEvaluator(labelCol="label",
                                                  rawPredictionCol="uplift_score",
                                                  metricName='areaUnderROC'
                                                  )
        auc = Evaluator.evaluate(data)
        score_dict[alpha] = auc

    res_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
    return res_dict

# 基于Qini面积的alpha选取
def CalAlphaByQini(df, alpha_range=(0.0, 1.1), step=0.1):
    """
    用于计算uplift模型中，score1 - alpha * score0 中的最佳alpha值（基于Qini曲线面积）
    :param df: 包含 user_id, label, groupid, score1, score0 的dataframe
    :param alpha_range: 自定义alpha搜索范围 [tuple]
    :param step: alpha值搜索步长 [float]
    :return: 不同alpha取值的Qini面积结果字典 [dict[float, float]]
    """
    res = dict()
    for alpha in np.arange(alpha_range[0], alpha_range[1], step):
        # 样本排序
        df_ = df.copy()
        df_['uplift'] = df_.apply(lambda x: x['score1'] - alpha * x['score0'], axis=1)
        df_sort = df_.sort_values(by='uplift', ascending=False)
        # 开始计算
        n = len(df_sort)
        # 统计累计到当前样本的 Cumulative actual gain
        sample_uplift = []
        # 不适用iloc方法，太耗时
        N_t_x, N_c_x = 0, 0
        R_t_x, R_c_x = 0, 0
        # 遍历一次即可（速率：50w / min）
        for idx, row in tqdm(df_sort[["label", "groupid"]].iterrows()):
            group_id = row["groupid"]
            label = row["label"]
            if group_id == 1:
                N_t_x += 1
                if label == 1:
                    R_t_x += 1
            elif group_id == 0:
                N_c_x += 1
                if label == 1:
                    R_c_x += 1
            # 值为零的判断
            if N_c_x == 0:
                cumulative_actual_gain = R_t_x
                sample_uplift.append(cumulative_actual_gain)
            else:
                cumulative_actual_gain = R_t_x - R_c_x * (N_t_x / N_c_x)
                sample_uplift.append(cumulative_actual_gain)

        # 计算曲线面积
        area_cruve = np.trapz(sample_uplift) / 1e6
        # 随机算法面积
        area_random = n * sample_uplift[-1] / 2 / 1e6
        # 面积差值,即为auuc
        auuc = area_cruve - area_random
        # 写入字典
        res[alpha] = auuc
        print("alpha: {:.1f}, qini cruve: {:.2f}".format(alpha, auuc))

    return res
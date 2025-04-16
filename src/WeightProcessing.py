import pandas as pd

def merge_and_preprocess_weights(entropy_df, ahp_df):
    """
    合并熵权法和AHP权重，并进行必要的预处理。

    参数:
    entropy_df (DataFrame): 包含熵权法数据的DataFrame
    ahp_df (DataFrame): 包含AHP权重数据的DataFrame

    返回:
    DataFrame: 合并后的权重DataFrame
    """
    # 重命名权重列
    entropy_df = entropy_df.rename(columns={"权重": "entropy_weight"})
    ahp_df = ahp_df.rename(columns={"权重": "ahp_weight"})

    # 合并熵权法和AHP权重（左连接，保留所有熵权法指标）
    merged_weights = pd.merge(
        entropy_df[["指标code", "上级指标", "entropy_weight"]],
        ahp_df[["指标code", "ahp_weight"]],
        on="指标code",
        how="left"
    )

    # 去掉 '%' 符号，并转换为浮动类型
    merged_weights["ahp_weight"] = merged_weights["ahp_weight"].str.replace('%', '').astype(float) / 100

    # 处理缺失值：如果某指标在AHP中没有权重，则设为1
    merged_weights["ahp_weight"] = merged_weights["ahp_weight"].fillna(1)

    return merged_weights


def calculate_combined_and_normalized_weights(merged_weights, a1=0.5):
    """
    计算组合权重，并按父指标分组归一化（确保每个父指标下权重和为1）。

    参数:
    merged_weights (DataFrame): 包含合并后的权重数据的DataFrame
    a1 (float): AHP权重的占比，默认0.5

    返回:
    DataFrame: 计算后包含组合权重和归一化权重的DataFrame
    """
    # 计算组合权重
    merged_weights["combined_weight"] = (
            a1 * merged_weights["ahp_weight"] +
            (1 - a1) * merged_weights["entropy_weight"]
    )

    # 按父指标分组归一化（确保每个父指标下权重和为1）
    def normalize_weights(group):
        total = group["combined_weight"].sum()
        if total > 0:
            return group["combined_weight"] / total
        else:
            # 如果总权重为0，则平均分配
            return 1 / len(group)

    merged_weights["normalized_weight"] = merged_weights.groupby("上级指标")["combined_weight"].transform(
        lambda x: x / x.sum()
    )

    return merged_weights


def calculate_scores(merged_weights, last_df):
    """
    计算上一级指标得分，并返回只包含证券编码、证券名称以及[parent]列的DataFrame。

    参数:
    merged_weights (DataFrame): 包含合并后并归一化的权重数据
    score4_df (DataFrame): 上一级指标得分的DataFrame，包含证券编码、证券名称以及指标得分

    返回:
    DataFrame: 计算后的上一级指标得分DataFrame，只包含证券编码、证券名称和[parent]列
    """
    # 获取所有唯一的上一级指标名称
    parent_indicators = merged_weights["上级指标"].unique()


    # 遍历每个上一级指标，计算其得分
    for parent in parent_indicators:
        # 获取该父指标下的所有子指标及其归一化权重
        sub_weights = merged_weights[merged_weights["上级指标"] == parent]

        # 初始化得分列
        # 方法1：明确创建副本（推荐）
        last_df = last_df.copy()
        last_df[parent] = 0.0

        # 累加子指标得分
        for _, row in sub_weights.iterrows():
            code = row["指标code"]
            weight = row["normalized_weight"]

            if code in last_df.columns:
                last_df.loc[:, parent] += last_df[code] * weight
            else:
                print(f"警告: 指标 {code} 不存在于 score4_df 中，已跳过")

    # 只保留证券编码、证券名称和[parent]列
    parent_columns = [parent for parent in parent_indicators]
    result_columns = ["证券编码", "证券名称"] + parent_columns
    last_df = last_df[result_columns]

    return last_df


def merge_scores(last_df, current_df):
    """
    合并上一级得分数据。

    参数:
    score4_df (DataFrame): 计算后的上一级得分DataFrame
    score3_df (DataFrame): 原有的上一级得分DataFrame

    返回:
    DataFrame: 合并后的最终结果
    """
    # 提取最终结果列（证券编码、名称 + 所有三级指标）
    result_columns = ["证券编码", "证券名称"] + list(last_df.columns.difference(["证券编码", "证券名称"]))
    final_result = last_df[result_columns]

    # 与原有的三级得分数据合并（保留所有行，填充缺失值为0）
    merged_result = pd.merge(
        final_result,
        current_df,
        on=["证券编码", "证券名称"],
        how="outer"
    ).fillna(0)

    return merged_result
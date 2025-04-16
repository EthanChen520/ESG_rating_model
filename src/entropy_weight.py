import pandas as pd
import numpy as np


# 定义标准化函数（含异常处理）
def min_max_normalize(df, verbose):
    normalized_df = df.copy()
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max == col_min:
            normalized_df[col] = 0.5  # 处理常数列
            if verbose:
                print(f"警告: 列 {col} 为常数值，已标准化为0.5")
        else:
            normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
    return normalized_df


# 定义熵值计算（含异常处理）
def calculate_entropy(df_normalized):
    n, m = df_normalized.shape
    if n <= 1:
        raise ValueError("样本数量不足，无法计算熵值")

    # 处理零值
    p = df_normalized / df_normalized.sum(axis=0).replace(0, 1e-10)
    p = p.replace(0, 1e-10)

    entropy = - (1 / np.log(n)) * (p * np.log(p)).sum(axis=0)
    return entropy.replace(np.nan, 1)  # 处理异常熵值


def calculate_weights(E_j):
    """计算权重，包含异常处理"""
    # 计算差异系数
    diff_coefficient = 1 - E_j

    # 处理所有差异系数和为0的情况（例如所有熵值都为1）
    if diff_coefficient.sum() == 0:
        print("警告：所有指标的熵值均为1，将自动平均分配权重")
        return pd.Series([1 / len(E_j)] * len(E_j), index=E_j.index)

    # 正常计算权重
    weights = diff_coefficient / diff_coefficient.sum()
    return weights


def calculate_entropy_weights(match_df, df, indicator_level, verbose):
    """
    封装熵权法计算流程
    :param match_df: 指标匹配表
    :param df: 数据表
    :param indicator_level: 需要处理的指标等级
    :param verbose: 是否打印过程信息
    :return: 包含各上级指标权重和熵值的字典
    """
    # 结果容器
    results = {}

    # 类型预处理
    match_df = match_df.copy()
    match_df['指标code'] = match_df['指标code'].astype(str)
    df.columns = df.columns.astype(str)

    # 构建指标映射
    upper_indicator_to_codes = (
        match_df[match_df['指标等级'] == indicator_level]
        .groupby('上级指标')['指标code']
        .apply(list)
        .to_dict()
    )

    # 遍历处理每个上级指标
    for upper_indicator, codes in upper_indicator_to_codes.items():
        if verbose:
            print(f"\n处理上级指标: {upper_indicator}")

        # 获取有效列
        valid_columns = [col for col in codes if col in df.columns]
        if not valid_columns:
            if verbose:
                print("未找到有效对应列")
            continue

        # 提取数据
        selected_df = df[valid_columns].fillna(0)  # 将 NaN 替换为 0

        try:
            # 标准化
            normalized_df = min_max_normalize(selected_df, verbose)

            # 熵值计算
            E_j = calculate_entropy(normalized_df)

            # 权重计算
            weights = calculate_weights(E_j)

            # 存储结果
            results[upper_indicator] = {
                'entropy': E_j,
                'weights': weights,
                'valid_columns': valid_columns
            }

            if verbose:
                print(f"\n熵值:\n{E_j}")
                print(f"\n权重:\n{weights}")

        except Exception as e:
            if verbose:
                print(f"计算失败: {str(e)}")
            results[upper_indicator] = {
                'error': str(e)
            }

    return results


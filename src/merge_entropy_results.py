import pandas as pd
import numpy as np

def merge_entropy_results(results, match_df, indicator_level):
    """
    将熵权法计算结果合并为一个结构化的DataFrame，包含指标等级

    参数:
        results: calculate_entropy_weights函数返回的结果字典
        match_df: 指标匹配表
        indicator_level: 指标等级

    返回:
        包含上级指标、指标代码、指标等级、熵值和权重的DataFrame
    """
    # 确保指标code为字符串类型
    match_df = match_df.copy()
    match_df['指标code'] = match_df['指标code'].astype(str)

    merged_data = []

    for upper_indicator, data in results.items():
        # 跳过出错的结果
        if 'error' in data:
            continue

        # 提取每个指标的信息
        for code in data['valid_columns']:
            merged_data.append({
                '上级指标': upper_indicator,
                '指标code': code,
                '指标等级': indicator_level,
                '熵值': data['entropy'].get(code, np.nan),
                '权重': data['weights'].get(code, np.nan)
            })

    # 创建DataFrame
    result_df = pd.DataFrame(merged_data)

    # 按照 '指标code' 列将 result_df 和 match_df 合并
    result_df = result_df.merge(match_df[['指标code', '支柱']], on='指标code', how='left')

    # 重新排序列
    result_df = result_df[['上级指标', '指标code', '指标等级', '熵值', '权重', '支柱']]

    return result_df
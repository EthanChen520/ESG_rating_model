from skopt.space import Real
from skopt import gp_minimize
import numpy as np
from itertools import product
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from src.WeightProcessing import merge_and_preprocess_weights, calculate_combined_and_normalized_weights, calculate_scores, merge_scores
from src.entropy_weight import calculate_entropy_weights
from src.merge_entropy_results import merge_entropy_results


# 读取指标表
match_df = pd.read_csv('ESG_index.csv') #指标元素表
# 读取熵权法结果（子指标权重）
layer4_entropy_df = pd.read_csv(
    'layer4_entropy_weight_results.csv')
# 读取AHP结果（子指标权重）
ahp_df = pd.read_csv(
    'ahp_analysis_results.csv')
# 读取四级指标得分数据（叶子节点）
ESG4_df = pd.read_csv(
    'ESG4.csv')
# 读取三级指标得分数据（父节点）
ESG3_incomplete_df = pd.read_csv(
    'ESG3_incomplete.csv')
# 读取MSCI 股票的结果
MSCI_df = pd.read_csv(
    'MSCI.csv')


def process_weights_and_scores(entropy_df, ahp_df, ESG_last_df, a1_E=None, a1_S=None, a1_G=None, is_pillar_based=False):
    # 如果按支柱进行处理
    if is_pillar_based:
        # 获取 '支柱' 列中的唯一值
        unique_pillars = entropy_df['支柱'].unique()

        # 创建一个空的 DataFrame，用于保存每个支柱的结果
        all_ESG_current_dfs = []

        # 为每个支柱设置对应的a1值
        pillar_to_a1 = {
            'E': a1_E,
            'S': a1_S,
            'G': a1_G
        }

        # 遍历每个支柱
        for pillar in unique_pillars:
            # 根据支柱值筛选数据
            pillar_entropy_df = entropy_df[entropy_df['支柱'] == pillar]

            # 获取当前支柱对应的a1值
            current_a1 = pillar_to_a1.get(pillar)  # 获取对应支柱的a1值

            # 合并权重并预处理
            merged_weights = merge_and_preprocess_weights(pillar_entropy_df, ahp_df)

            # 计算组合权重并归一化，传入对应的a1值
            merged_weights = calculate_combined_and_normalized_weights(merged_weights, current_a1)

            # 计算三级得分
            ESG_current_df = calculate_scores(merged_weights, ESG_last_df)

            # 删除 '支柱' 列，不保留支柱
            ESG_current_df = ESG_current_df.drop(columns=['支柱'], errors='ignore')  # 'errors' 处理如果没有支柱列

            # 将每个支柱的结果添加到列表中
            all_ESG_current_dfs.append(ESG_current_df)

        # 合并所有支柱的结果，按照证券编码和证券名称匹配
        final_ESG_df = all_ESG_current_dfs[0]  # 从第一个支柱开始

        # 从第二个支柱开始依次与 final_ESG_df 合并
        for pillar_df in all_ESG_current_dfs[1:]:
            final_ESG_df = pd.merge(final_ESG_df, pillar_df, on=["证券编码", "证券名称"], how="outer",
                                    suffixes=('_' + pillar, '_new'))

        return final_ESG_df

    else:
        # 不按支柱进行处理
        # 合并权重并预处理
        merged_weights = merge_and_preprocess_weights(entropy_df, ahp_df)

        # 计算组合权重并归一化
        merged_weights = calculate_combined_and_normalized_weights(merged_weights, a1_E)  # 使用统一的 a1

        # 计算三级得分
        ESG_current_df = calculate_scores(merged_weights, ESG_last_df)

        return ESG_current_df


# 目标函数：根据给定的 alpha3 和 alpha2 值，计算相关性并返回负的相关性（因为优化器是最小化目标函数）
def objective(params):
    alpha4_E, alpha4_S, alpha4_G, alpha3_E, alpha3_S, alpha3_G, alpha2_E, alpha2_S, alpha2_G = params

    # 打印当前 alpha 组合
    print(f"\n当前 alpha 组合:")
    print(f"alpha4_E={alpha4_E:.3f}, alpha4_S={alpha4_S:.3f}, alpha4_G={alpha4_G:.3f}")
    print(f"alpha3_E={alpha3_E:.3f}, alpha3_S={alpha3_S:.3f}, alpha3_G={alpha3_G:.3f}")
    print(f"alpha2_E={alpha2_E:.3f}, alpha2_S={alpha2_S:.3f}, alpha2_G={alpha2_G:.3f}")

    # 1. ESG三级得分
    ESG3_incomplete_2_df = process_weights_and_scores(layer4_entropy_df, ahp_df, ESG4_df, alpha4_E, alpha4_S, alpha4_G, is_pillar_based=True)
    ESG3_df = merge_scores(ESG3_incomplete_2_df, ESG3_incomplete_df)

    # 2. 计算三级到二级的entropy_weight并返回DataFrame
    ESG3_entropy_weight = calculate_entropy_weights(match_df=match_df, df=ESG3_df, indicator_level=3, verbose=False)
    layer3_entropy_df = merge_entropy_results(results=ESG3_entropy_weight, match_df=match_df, indicator_level=3)

    # 计算第二级ESG数据分数
    ESG2_df = process_weights_and_scores(layer3_entropy_df, ahp_df, ESG3_df, alpha3_E, alpha3_S, alpha3_G, is_pillar_based=True)

    # 计算二级到一级的entropy_weight并返回DataFrame
    ESG2_entropy_weight = calculate_entropy_weights(match_df=match_df, df=ESG2_df, indicator_level=2, verbose=False)
    layer2_entropy_df = merge_entropy_results(results=ESG2_entropy_weight, match_df=match_df, indicator_level=2)

    # 计算ESG第一级数据分数
    ESG1_df = process_weights_and_scores(layer2_entropy_df, ahp_df, ESG2_df, alpha2_E, alpha2_S, alpha2_G, is_pillar_based=True)

    # 合并MSCI_df和ESG1_df，按证券编码和证券名称匹配
    merged_df = pd.merge(MSCI_df, ESG1_df, on=["证券编码", "证券名称"], how="inner", suffixes=('_msci', '_own'))

    # 计算相关性
    corr_E, _ = spearmanr(merged_df['E_msci'], merged_df['E_own'])
    corr_S, _ = spearmanr(merged_df['S_msci'], merged_df['S_own'])
    corr_G, _ = spearmanr(merged_df['G_msci'], merged_df['G_own'])

    # 计算三个相关系数的平均值
    avg_corr = np.mean([corr_E, corr_S, corr_G])

    # 返回负的平均相关系数（贝叶斯优化最小化目标函数，所以我们返回负值）
    return -avg_corr

# 定义超参数的搜索空间
space = [
    Real(0.0, 1.0, name='alpha4_E'),  # alpha4_E 范围
    Real(0.0, 1.0, name='alpha4_S'),  # alpha4_S 范围
    Real(0.0, 1.0, name='alpha4_G'),  # alpha4_G 范围
    Real(0.0, 1.0, name='alpha3_E'),  # alpha3_E 范围
    Real(0.0, 1.0, name='alpha3_S'),  # alpha3_S 范围
    Real(0.0, 1.0, name='alpha3_G'),  # alpha3_G 范围
    Real(0.0, 1.0, name='alpha2_E'),  # alpha2_E 范围
    Real(0.0, 1.0, name='alpha2_S'),  # alpha2_S 范围
    Real(0.0, 1.0, name='alpha2_G')   # alpha2_G 范围
]

# 使用贝叶斯优化进行超参数搜索
result = gp_minimize(objective, space, n_calls=50, random_state=42)

# 最优的超参数组合
best_alpha_combo = result.x
best_corr = -result.fun  # 因为我们最小化的是负的相关系数，所以需要取负值

# === 输出最优结果 ===
print(f"最优alpha组合：第四层α_E={best_alpha_combo[0]}, α_S={best_alpha_combo[1]}, α_G={best_alpha_combo[2]}")
print(f"最优alpha组合：第三层α_E={best_alpha_combo[3]}, α_S={best_alpha_combo[4]}, α_G={best_alpha_combo[5]}")
print(f"最优alpha组合：第二层α_E={best_alpha_combo[6]}, α_S={best_alpha_combo[7]}, α_G={best_alpha_combo[8]}")
print(f"最大斯皮尔曼相关系数：{best_corr}")






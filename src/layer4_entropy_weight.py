import pandas as pd
import os
from entropy_weight import calculate_entropy_weights
from merge_entropy_results import merge_entropy_results

match_df = pd.read_csv('ESG_index.csv') # 指标元素表
#分数矩阵
df = pd.read_csv('ESG4.csv') # 第四层分数
output_path = 'output_path' # 输出路径

# 第四层熵权法权重
results = calculate_entropy_weights(
    match_df=match_df,
    df=df,
    indicator_level=4,  # 可以修改指标等级
    verbose=False        # 关闭打印可设为False
)

# 包含上级指标、指标代码、指标等级、熵值和权重的DataFrame
merged_results = merge_entropy_results(
    results=results,
    match_df=match_df,
    indicator_level=4 # 传入指标等级
)

# 输出到指定文件路径
output_file = os.path.join(output_path, 'layer4_entropy_weight_results.csv')
merged_results.to_csv(output_file, index=False, encoding='utf_8_sig')
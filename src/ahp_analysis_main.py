import os
import numpy as np
import pandas as pd

# 数据文件夹路径
data_path = r'/matrix_data/data'
ESG_index_path = r'/raw_data/ESG_index.csv'

# 读取三个文件
ESG_df = pd.read_csv(ESG_index_path)



# 用于存储结果的列表
results = []


# 计算AHP的函数
def calculate_ahp(matrix, columns, filename):
    # 去掉文件名的扩展名（.csv）
    filename_without_extension = os.path.splitext(filename)[0]

    # 计算判断矩阵的特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(matrix)

    # 转换为实数（因理论上的判断矩阵特征值为实数）
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # 选择最大特征值对应的特征向量
    max_eigval_index = np.argmax(eigvals)
    max_eigvec = eigvecs[:, max_eigval_index]

    # 确保特征向量所有分量为正数（取绝对值）
    max_eigvec = np.abs(max_eigvec)

    # 归一化特征向量
    norm_max_eigvec = max_eigvec / np.sum(max_eigvec)

    # 计算一致性指标CI
    n = matrix.shape[0]
    ci = (eigvals[max_eigval_index] - n) / (n - 1) if n > 1 else 0

    # 处理n=1或n=2的特殊情况
    if n == 1:
        cr = 0.0
    elif n == 2:
        cr = 0.0
    else:
        # 标准RI值表（n=1~9）
        ri = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45]
        ri_index = n - 1  # RI表从n=1开始对应索引0
        if ri_index >= len(ri):
            ri_value = ri[-1]
        else:
            ri_value = ri[ri_index]
        cr = ci / ri_value if ri_value != 0 else 0

    # 输出结果
    weights = norm_max_eigvec
    weights_percent = [weight * 100 for weight in weights]

    # 根据去掉.csv后的文件名长度设置指标等级
    filename_without_extension_length = len(filename_without_extension)

    if filename_without_extension == "ESG":
        level = 0
    elif filename_without_extension_length == 1:
        level = 1
    elif 1 < filename_without_extension_length < 4:
        level = 2
    else:
        level = 3

    # 收集结果
    for i, weight in enumerate(weights_percent):
        result = {
            '上级指标': filename_without_extension,  # 使用去掉扩展名的文件名
            '一致性比率': cr,
            '指标名称': columns[i],
            '权重': f"{weight:.2f}%",
            '指标等级': level
        }
        results.append(result)


# 遍历文件夹并对每个CSV文件进行AHP分析
def analyze_all_files(data_path):
    # 使用os.walk递归遍历文件夹及子文件夹
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                print(f"\n正在处理文件: {file_path}")

                # 读取CSV文件并转换为numpy矩阵
                df = pd.read_csv(file_path, index_col=0)
                matrix = df.values

                # 检查矩阵是否为方阵
                if matrix.shape[0] != matrix.shape[1]:
                    print(f"错误：文件 {filename} 不是方阵，跳过处理")
                    continue

                # 调用AHP分析函数
                calculate_ahp(matrix, df.columns, filename)


# 执行文件分析
analyze_all_files(data_path)

# 将结果转换为DataFrame
df_results = pd.DataFrame(results)

# 进行左连接
merged_df = pd.merge(
    df_results,
    ESG_df[["指标名称", "指标code"]],  # 只保留 "指标code" 列
    left_on="指标名称",  # df_results 中的指标名称列
    right_on="指标名称",  # combined_df 中的指标名称列
    how="left"  # 保留 df_results 的所有行
)

# 填充指标名称为 'E'、'G'、'S' 的指标code为空的行
merged_df.loc[merged_df['指标名称'] == 'E', '指标code'] = 'E'
merged_df.loc[merged_df['指标名称'] == 'G', '指标code'] = 'G'
merged_df.loc[merged_df['指标名称'] == 'S', '指标code'] = 'S'


# 保存为CSV文件
output_file = r'/matrix_data/ahp_analysis_results.csv'
merged_df.to_csv(output_file, index=False)

print(f"\n结果已保存为: {output_file}")
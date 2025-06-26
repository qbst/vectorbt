import numpy as np
import pandas as pd
from vectorbt.base.array_wrapper import ArrayWrapper

# 准备测试数据
print("=== 准备测试数据 ===")
index = pd.Index(['2024-01-01', '2024-01-02', '2024-01-03'], name='date')
columns = pd.Index(['AAPL', 'GOOGL', 'MSFT'], name='symbol')

# 创建基础ArrayWrapper
wrapper = ArrayWrapper(
    index=index,
    columns=columns,
    ndim=2
)

print(f"原始包装器形状: {wrapper.shape}")
print(f"原始索引: {wrapper.index}")
print(f"原始列: {wrapper.columns}")
print()

# ============================================================================
# 示例1: 基本行列索引 - 选择前2行和前2列
# ============================================================================
print("=== 示例1: 基本行列索引 (选择前2行前2列) ===")

def indexing_func_1(x):
    """选择前2行，前2列"""
    return x.iloc[:2, :2]

result = wrapper.indexing_func_meta(indexing_func_1)
new_wrapper, idx_idxs, col_idxs, ungrouped_col_idxs = result

print(f"新包装器形状: {new_wrapper.shape}")
print(f"新索引: {new_wrapper.index}")
print(f"新列: {new_wrapper.columns}")
print(f"行索引数组: {idx_idxs}")
print(f"列索引数组: {col_idxs}")
print(f"未分组列索引数组: {ungrouped_col_idxs}")
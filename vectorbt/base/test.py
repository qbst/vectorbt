import pandas as pd
import numpy as np
# 假设 align_indexes 位于 vectorbt.base.index_fns 模块
from index_fns import align_indexes

# --- 1. 创建一些具有不同索引的 Pandas Series ---
index_s1 = pd.Index(['a', 'b', 'c', 'd', 'e'], name='idx_s1')
s1 = pd.Series([10, 20, 30, 40, 50], index=index_s1)

# s2 的索引与 s1 有重叠，但顺序不同，且包含 s1 没有的元素 'f'
index_s2 = pd.Index(['c', 'e', 'a', 'f', 'b'], name='idx_s2')
s2 = pd.Series([100, 200, 300, 400, 500], index=index_s2)

# s3 的索引与 s1, s2 也有重叠，且包含它们都没有的元素 'g'
index_s3 = pd.Index(['d', 'b', 'g', 'c'], name='idx_s3')
s3 = pd.Series([1000, 2000, 3000, 4000], index=index_s3)

print("--- 原始 Series ---")
print("Series 1 (s1):")
print(s1)
print("\nSeries 2 (s2):")
print(s2)
print("\nSeries 3 (s3):")
print(s3)

# --- 2. 提取这些 Series 的索引 ---
indexes_to_align = [s1.index, s2.index, s3.index]
print("\n--- 原始索引列表 ---")
for i, idx in enumerate(indexes_to_align):
    print(f"Index {i+1} ({idx.name}): {list(idx)}")

# --- 3. 调用 align_indexes 函数 ---
# 它返回一个元组，其中每个元素是一个 NumPy 整数数组。
# 这些整数数组是对应原始索引的 .iloc 定位符。
aligned_iloc_positions_tuple = align_indexes(indexes_to_align)

print("\n--- align_indexes 函数的输出 ---")
print("返回的整数位置元组 (用于 .iloc):")
for i, positions in enumerate(aligned_iloc_positions_tuple):
    original_index_name = indexes_to_align[i].name
    # 使用这些位置从原始索引中提取对应的标签，以验证它们确实指向共同的标签
    labels_from_iloc = list(indexes_to_align[i][positions])
    print(f"  对于索引 '{original_index_name}': 位置数组 = {positions}, 对应的标签 = {labels_from_iloc}")

# --- 4. 使用返回的整数位置来对齐原始 Series ---
# 通过 .iloc 和返回的位置数组，我们可以从每个原始 Series 中提取数据
# 使得提取出的新 Series 具有相同的索引（即原始索引的交集）
s1_aligned = s1.iloc[aligned_iloc_positions_tuple[0]]
s2_aligned = s2.iloc[aligned_iloc_positions_tuple[1]]
s3_aligned = s3.iloc[aligned_iloc_positions_tuple[2]]

print("\n--- 对齐后的 Series ---")
print("对齐后的 Series 1 (s1_aligned):")
print(s1_aligned)
print("\n对齐后的 Series 2 (s2_aligned):")
print(s2_aligned)
print("\n对齐后的 Series 3 (s3_aligned):")

# --- 5. 验证对齐结果 ---
# 对齐后的 Series 应该具有完全相同的索引
print("\n--- 验证对齐后索引的一致性 ---")
print(f"s1_aligned.index: {list(s1_aligned.index)}")
print(f"s2_aligned.index: {list(s2_aligned.index)}")
print(f"s3_aligned.index: {list(s3_aligned.index)}")

are_indexes_equal = (s1_aligned.index.equals(s2_aligned.index) and
                     s2_aligned.index.equals(s3_aligned.index))

print(f"\n所有对齐后的 Series 是否具有相同的索引? {are_indexes_equal}")

if are_indexes_equal:
    print("\n结论:")
    print("`align_indexes` 函数成功地找到了所有输入索引的公共标签（交集），")
    print(f"这个公共索引是: {list(s1_aligned.index)}")
    print("并为每个原始索引提供了整数位置，以便通过 .iloc 选取这些公共元素。")
    print("这使得原始数据可以在这些共同的索引点上进行比较或合并。")
else:
    print("\n结论: 对齐失败或行为与预期不符。")

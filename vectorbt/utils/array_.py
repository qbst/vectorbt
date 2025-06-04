# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

import numpy as np  
from numba import njit 

from vectorbt import _typing as tp  


def is_sorted(a: tp.Array1d) -> np.bool_:
    """
    检查一维数组是否已按升序排列
    
    参数：
        a (tp.Array1d): 待检查的一维NumPy数组
        
    返回：
        np.bool_: 如果数组已排序返回True，否则返回False
    """
    return np.all(a[:-1] <= a[1:])  


@njit(cache=True)  
def is_sorted_nb(a: tp.Array1d) -> bool:
    """
    检查一维数组是否已按升序排列（Numba编译版本）
    
    参数：
        a (tp.Array1d): 待检查的一维NumPy数组
        
    返回：
        bool: 如果数组已排序返回True，否则返回False
    """
    for i in range(a.size - 1):  
        if a[i + 1] < a[i]: 
            return False  
    return True  


@njit(cache=True)  
def insert_argsort_nb(A: tp.Array1d, I: tp.Array1d) -> None:
    """
    原地插入排序，时间复杂度O(n²)，空间复杂度O(1)
    
    参数：
        A (tp.Array1d): 待排序的一维数组，会被原地修改
        I (tp.Array1d): 对应的索引数组，会被原地修改以反映排序后的索引
        
    返回：
        None: 原地操作，直接修改输入数组
        
    注意事项：
        - NaN值会被排序到数组末尾
    """
    for j in range(1, len(A)):  
        A_j = A[j] 
        I_j = I[j]  
        i = j - 1  
        
        while i >= 0 and (A[i] > A_j or np.isnan(A[i])):
            A[i + 1] = A[i]
            I[i + 1] = I[i]  
            i = i - 1  
            
        A[i + 1] = A_j  
        I[i + 1] = I_j  


def get_ranges_arr(starts: tp.ArrayLike, ends: tp.ArrayLike) -> tp.Array1d:
    """
    基于起始和结束索引构建连续范围数组
    
    参数：
        starts (tp.ArrayLike): 起始索引，可以是标量、列表或一维数组
        ends (tp.ArrayLike): 结束索引，可以是标量、列表或一维数组
        
    返回：
        tp.Array1d: 包含所有指定范围内整数的一维数组
        
    示例：
        >>> get_ranges_arr([1, 5], [4, 7])
        array([1, 2, 3, 5, 6])
    """
    # 以 starts = [1, 5], ends = [4, 7] 为例
    
    starts_arr = np.asarray(starts)  
    if starts_arr.ndim == 0:  # 如果是标量，转换为1维数组
        starts_arr = np.array([starts_arr])
        
    ends_arr = np.asarray(ends)  
    if ends_arr.ndim == 0:
        ends_arr = np.array([ends_arr])
        
    # 使用广播使两个数组形状一致
    starts_arr, end = np.broadcast_arrays(starts_arr, ends_arr)  
    # 计算每个范围的长度（元素个数）
    counts = ends_arr - starts_arr  # [3, 2]
    # 计算长度的累积和，用于确定每个范围在结果数组中的位置
    counts_csum = counts.cumsum()  # [3, 5]，减去1就是每个范围的结束值所在索引
    # 创建标识数组，长度为所有范围的总长度
    id_arr = np.ones(counts_csum[-1], dtype=int)  # [1, 1, 1, 1, 1]
    # 设置第一个元素为第一个范围的起始值
    id_arr[0] = starts_arr[0]  # [1, 1, 1, 1, 1]
    # 在每个新范围的起始位置设置跳跃值
    id_arr[counts_csum[:-1]] = starts_arr[1:] - ends_arr[:-1] + 1  # [1, 1, 1, 2, 1]
    # 对标识数组进行累积和操作，生成最终的范围数组
    return id_arr.cumsum()  # [1, 2, 3, 5, 6]


@njit(cache=True)  # 使用Numba JIT编译以获得最优性能
def uniform_summing_to_one_nb(n: int) -> tp.Array1d:
    """
    生成和为1的均匀随机浮点数数组
    
    参数：
        n (int): 要生成的随机数个数，必须大于0
        
    返回：
        tp.Array1d: 长度为n的一维数组，所有元素和为1
    """
    rand_floats = np.empty(n + 1, dtype=np.float64) 
    rand_floats[0] = 0.  
    rand_floats[1] = 1.  
    # 在(0,1)区间内生成n-1个随机点
    rand_floats[2:] = np.random.uniform(0, 1, n - 1)  # 例如 n=4，则rand_floats = [0, 1, 0.23, 0.45, 0.32]
    rand_floats = np.sort(rand_floats)  # [0, 0.23, 0.32, 0.45, 1]
    rand_floats = rand_floats[1:] - rand_floats[:-1]  # [0.23, 0.09, 0.13, 0.65]
    return rand_floats  # 返回和为1的随机数数组


def renormalize(a: tp.MaybeArray[float], from_range: tp.Tuple[float, float],
                to_range: tp.Tuple[float, float]) -> tp.MaybeArray[float]:
    """
    将数值从一个范围线性重新映射到另一个范围
    
    参数：
        a (tp.MaybeArray[float]): 输入数值，可以是标量或数组
        from_range (tp.Tuple[float, float]): 源范围，格式为(最小值, 最大值)
        to_range (tp.Tuple[float, float]): 目标范围，格式为(最小值, 最大值)
        
    返回：
        tp.MaybeArray[float]: 重新映射后的数值，类型与输入保持一致
        
        
    示例：
        >>> renormalize([0, 25, 50, 75, 100], (0, 100), (0, 1))
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    from_delta = from_range[1] - from_range[0]  
    to_delta = to_range[1] - to_range[0]  
    return (to_delta * (a - from_range[0]) / from_delta) + to_range[0]


renormalize_nb = njit(cache=True)(renormalize)
"""
创建renormalize函数的Numba编译版本
"""

def min_rel_rescale(a: tp.Array, to_range: tp.Tuple[float, float]) -> tp.Array:
    """
    以 to_range[0] 为头基准，进行线性比例缩小：
        如果 np.min(a) / np.max(a) < to_range[0] / to_range[1]，保持比例不变
        否则，将 a 以 to_range[0] / to_range[1] 比例进行缩小
    """
    a_min = np.min(a)  
    a_max = np.max(a)  
    
    if a_max - a_min == 0:  # 如果所有元素都相等，返回全为目标最小值的数组
        return np.full(a.shape, to_range[0])  
    
    from_range = (a_min, a_max)
    
    from_range_ratio = np.inf  
    if a_min != 0:  
        from_range_ratio = a_max / a_min  
    
    to_range_ratio = to_range[1] / to_range[0]  
    
    if from_range_ratio < to_range_ratio: 
        to_range = (to_range[0], to_range[0] * from_range_ratio)
    
    return renormalize(a, from_range, to_range)


def max_rel_rescale(a: tp.Array, to_range: tp.Tuple[float, float]) -> tp.Array:
    a_min = np.min(a)  
    a_max = np.max(a)  
    
    if a_max - a_min == 0:  
        return np.full(a.shape, to_range[1])  
    
    from_range = (a_min, a_max)  
    
    from_range_ratio = np.inf  
    if a_min != 0:  
        from_range_ratio = a_max / a_min  
    
    to_range_ratio = to_range[1] / to_range[0]  
    
    if from_range_ratio < to_range_ratio:  
        to_range = (to_range[1] / from_range_ratio, to_range[1])
    
    return renormalize(a, from_range, to_range)  


@njit(cache=True)  # 使用Numba JIT编译以获得最优性能
def rescale_float_to_int_nb(floats: tp.Array, 
                            int_range: tp.Tuple[float, float], 
                            total: float) -> tp.Array:
    """
    将浮点数数组缩放为整数数组，同时保持总和为 total
    
    参数：
        floats (tp.Array): 输入的浮点数数组，通常已经归一化到[0,1]
        int_range (tp.Tuple[float, float]): 整数范围，格式为(最小值, 最大值)
        total (float): 目标总和，转换后的整数数组总和应等于此值
        
    返回：
        tp.Array: 整数数组，总和等于指定的total值
    """
    # 将浮点数组重新映射到整数范围，并使用floor获得初始整数分配
    ints = np.floor(renormalize_nb(floats, [0., 1.], int_range)) 
    
    leftover = int(total - ints.sum())  
    
    # 随机分配余量，确保总和准确
    for i in range(leftover): 
        ints[np.random.choice(len(ints))] += 1  # 随机选择一个元素并增加1
        
    return ints  

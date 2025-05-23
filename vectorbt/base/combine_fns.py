# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT BASE MODULE: COMBINE FUNCTIONS
================================================================================

文件作用概述：
本文件是vectorbt库中专门用于数组组合和批量函数应用的高性能计算引擎。该模块是vectorbt
实现大规模参数扫描、批量回测和分布式计算的核心基础设施，为量化交易中常见的"暴力搜索"
超参数空间提供了强大的计算支持。

核心设计理念：
1. **批量计算优化**：通过apply_and_concat系列函数，将单次函数调用扩展为批量并行调用，
   然后高效地将结果连接成单一数组，避免循环开销和内存碎片化。

2. **多层性能优化**：提供三个性能层次的实现：
   - Python原生版本：功能完整，支持进度条和复杂参数传递
   - Numba编译版本：极致性能，适用于数值密集型计算
   - Ray分布式版本：利用多核/多机资源，处理超大规模计算任务

3. **灵活的结果组织**：支持单输出和多输出函数的批量处理，自动处理结果的维度标准化
   和列向连接，确保输出格式的一致性和可预测性。

4. **组合与聚合模式**：提供combine_and_concat等函数，实现一对多的组合计算模式，
   以及combine_multiple等函数实现链式组合计算。

主要功能模块：
- **单输出批量处理**：apply_and_concat_one系列，用于批量应用返回单个数组的函数
- **多输出批量处理**：apply_and_concat_multiple系列，用于批量应用返回多个数组的函数  
- **组合计算模式**：combine_and_concat系列，实现基准对象与多个目标对象的组合计算
- **分布式计算支持**：ray_apply及其衍生函数，提供基于Ray的分布式并行计算
- **性能优化工具**：Numba编译版本，为数值计算提供接近C语言的执行速度

典型应用场景：
- **策略参数优化**：对数百/数千个参数组合进行批量回测，寻找最优参数
- **技术指标批量计算**：同时计算多个周期的移动平均线、RSI、MACD等指标
- **多资产组合分析**：并行计算不同股票组合的风险收益特征  
- **敏感性分析**：评估策略对不同市场条件或参数变化的敏感程度
- **蒙特卡洛模拟**：进行大规模随机模拟，评估策略的统计特性

技术特点：
- 深度集成Numba JIT编译，核心循环达到接近C语言的执行速度
- 原生支持Ray分布式计算框架，轻松扩展到多机集群
- 智能的内存管理，通过预分配避免动态内存分配的开销
- 完整的进度监控和任务管理，支持长时间运行的大型计算任务
- 与numpy broadcasting完美集成，自动处理维度转换和数组对齐

与其他模块的协作：
- 使用reshape_fns模块的to_2d等函数确保输出数组格式的一致性
- 与column_grouper协作处理分组计算和聚合操作
- 为vectorbt高层API（如指标计算、策略回测）提供底层计算引擎
- 作为vectorbt性能优化的核心组件，支撑整个库的高性能特性
"""

import numpy as np
from numba import njit
from tqdm.auto import tqdm

from vectorbt import _typing as tp
from vectorbt.base import reshape_fns


def apply_and_concat_none(n: int,
                          apply_func: tp.Callable, 
                          *args,
                          show_progress: bool = False,
                          tqdm_kwargs: tp.KwargsLike = None,
                          **kwargs) -> None:
    """
    对范围[0, n)内的每个整数值依次调用指定的应用函数，主要用于执行原地操作
    （如文件写入、数据库更新、图像保存等）而不需要收集返回值的场景。
    
    Args:
        n (int): 循环迭代的总次数，函数将对0到n-1的每个整数值调用apply_func
        apply_func (tp.Callable): 要应用的可调用函数，该函数必须接受以下参数：
                                 - 第一个参数：当前迭代的索引值i (int)
                                 - *args：位置参数
                                 - **kwargs：关键字参数
        *args: 传递给apply_func的额外位置参数
        show_progress (bool, optional): 是否显示进度条。默认为False
                                      - True: 显示tqdm进度条
                                      - False: 不显示进度条
        tqdm_kwargs (tp.KwargsLike, optional): 传递给tqdm进度条的配置参数字典
                                             常用参数如desc（描述文本）、unit（单位）等
                                             默认为None，内部会初始化为空字典
        **kwargs: 传递给apply_func的额外关键字参数
    
    Returns:
        None: 该函数不返回任何值，主要用于执行副作用操作
    
    Examples:
        >>> # 示例1：批量保存文件
        >>> def save_file(i, data_list, output_dir):
        ...     filename = f"{output_dir}/file_{i}.txt"
        ...     with open(filename, 'w') as f:
        ...         f.write(data_list[i])
        >>> 
        >>> data = ["content1", "content2", "content3"]
        >>> apply_and_concat_none(3, save_file, data, "/tmp", 
        ...                      show_progress=True,
        ...                      tqdm_kwargs={"desc": "保存文件"})
    """
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    for i in tqdm(range(n), disable=not show_progress, **tqdm_kwargs):
        apply_func(i, *args, **kwargs)


@njit  # Numba即时编译装饰器，将Python函数编译为高性能的机器码
def apply_and_concat_none_nb(n: int, apply_func_nb: tp.Callable, *args) -> None:
    """
    apply_and_concat_none函数的Numba编译版本，用于高性能的批量函数应用操作
    由于Numba的限制，该函数简化了参数接口，去除了进度条显示和关键字参数支持。
    
    Args:
        n (int): 循环迭代的总次数，函数将对0到n-1的每个整数值调用apply_func_nb
               必须是正整数，表示要执行的批量操作数量
        apply_func_nb (tp.Callable): 要应用的Numba编译函数，该函数必须满足以下要求：
                                   - 必须使用@njit装饰器进行Numba编译
                                   - 第一个参数：当前迭代的索引值i (int)
                                   - 后续参数：*args解包的位置参数
                                   - 不支持关键字参数(**kwargs)
                                   - 函数内部只能使用Numba支持的操作和数据类型
        *args: 传递给apply_func_nb的额外位置参数，必须是Numba兼容的数据类型
              包括NumPy数组、标量数值、元组等，不支持Python对象、字典、列表等
    
    Returns:
        None: 该函数不返回任何值，主要用于执行副作用操作（如原地数组修改）
    
    Examples:
        >>> import numpy as np
        >>> from numba import njit
        >>> 
        >>> # 示例1：批量原地数组操作
        >>> @njit
        ... def multiply_inplace(i, arr, multiplier):
        ...     arr[i] *= multiplier  # 原地修改数组元素
        >>> 
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> apply_and_concat_none_nb(len(data), multiply_inplace, data, 2.0)
        >>> # 结果：data变为[2.0, 4.0, 6.0, 8.0, 10.0]
    """
    for i in range(n):
        apply_func_nb(i, *args)


def apply_and_concat_one(n: int,
                         apply_func: tp.Callable, *args,
                         show_progress: bool = False,
                         tqdm_kwargs: tp.KwargsLike = None,
                         **kwargs) -> tp.Array2d:
    """
    批量应用函数并按列连接结果的核心工具函数
    
    该函数对范围[0, n)内的每个整数值依次调用指定的应用函数，收集每次调用的结果，
    并将所有结果沿着轴1（列方向）连接成一个二维NumPy数组。这是vectorbt库中
    用于批量处理和超参数网格搜索的核心函数之一，特别适用于量化交易中的批量回测、
    特征工程和策略优化等场景。
    
    Args:
        n (int): 循环迭代的总次数，函数将对0到n-1的每个整数值调用apply_func
               必须是正整数，表示要执行的批量操作数量
               
        apply_func (tp.Callable): 要应用的可调用函数，该函数必须满足以下要求：
                                 - 第一个参数：当前迭代的索引值i (int)
                                 - 后续参数：*args解包的位置参数
                                 - 关键字参数：**kwargs解包的关键字参数
                                 - 返回值：必须是1维或2维的NumPy数组
                                 
        *args: 传递给apply_func的额外位置参数
              可以是任何Python对象，常见类型包括：
              - NumPy数组（如价格数据、成交量数据）
              - 标量参数（如技术指标的周期参数）
              - 配置对象（如策略参数字典）
              
        show_progress (bool, optional): 是否显示进度条。默认为False
                                      - True: 显示tqdm进度条，适用于长时间运行的批量计算
                                      - False: 不显示进度条，适用于快速计算或自动化脚本
                                      
        tqdm_kwargs (tp.KwargsLike, optional): 传递给tqdm进度条的配置参数字典
                                             常用参数包括：
                                             - desc: 进度条描述文本（如"计算技术指标"）
                                             - unit: 进度单位（如"股票", "策略", "参数组合"）
                                             - ncols: 进度条显示宽度
                                             - leave: 是否保留完成后的进度条
                                             默认为None，内部会初始化为空字典
                                             
        **kwargs: 传递给apply_func的额外关键字参数
                 可以包含策略参数、计算配置、数据处理选项等
    
    Returns:
        tp.Array2d: 二维NumPy数组，形状为(rows, n * cols)
                   其中rows是apply_func返回数组的行数，cols是返回数组的列数
                   每次apply_func调用的结果按列顺序连接在一起
                   
                   数组结构说明：
                   - 如果apply_func返回1维数组[a1, a2, a3]，结果形状为(3, n)
                   - 如果apply_func返回2维数组shape为(m, k)，结果形状为(m, n*k)
                   - 第i次调用的结果存储在列索引[i*k:(i+1)*k]的位置
    Examples:
        >>> import numpy as np
        >>> 
        >>> # 示例1：批量计算不同周期的简单移动平均线
        >>> def calculate_sma(i, prices, periods):
        ...     period = periods[i]
        ...     # 计算简单移动平均线
        ...     sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        ...     return sma.reshape(-1, 1)  # 返回列向量
        >>> 
        >>> prices = np.random.rand(100) * 100  # 模拟价格数据
        >>> periods = [5, 10, 20, 50]  # 不同的移动平均周期
        >>> results = apply_and_concat_one(
        ...     len(periods), 
        ...     calculate_sma, 
        ...     prices, 
        ...     periods,
        ...     show_progress=True,
        ...     tqdm_kwargs={"desc": "计算SMA", "unit": "周期"}
        ... )
        >>> # 结果：shape为(96, 4)，每列对应一个周期的SMA值
    """
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    
    outputs = []
    
    for i in tqdm(range(n), disable=not show_progress, **tqdm_kwargs):
        # reshape_fns.to_2d(): 确保apply_func返回的数组被转换为2维格式，来确保后续np.column_stack操作的一致性
        outputs.append(reshape_fns.to_2d(apply_func(i, *args, **kwargs)))
    
    # 使用NumPy的column_stack函数将所有输出数组按列连接
    # column_stack会将输入的数组序列沿着轴1（列方向）堆叠
    # 例如：[array([[1], [2]]), array([[3], [4]])] -> array([[1, 3], [2, 4]])
    return np.column_stack(outputs)

@njit
def to_2d_one_nb(a: tp.Array) -> tp.Array2d:
    """
    a的维度：大于1时，直接返回a
            0维 () -> 1维数组 (1,) -> 2维数组 (1, 1)  
            1维 (n,) -> 2维数组 (n, 1)
    """
    if a.ndim > 1:  
        return a  

    return np.expand_dims(a, axis=1)


@njit
def apply_and_concat_one_nb(n: int, apply_func_nb: tp.Callable, *args) -> tp.Array2d:
    """
    apply_and_concat_one函数的Numba高性能编译版本，用于批量应用函数并按列连接结果
    
    该函数是vectorbt库中批量处理和超参数优化的核心工具之一的高性能版本，专门用于
    量化交易中的大规模计算场景（如批量回测、参数扫描、技术指标批量计算等）。
    相比Python版本，Numba编译版本能提供10-100倍的性能提升。
    
    函数工作流程：
    1. 首先调用apply_func_nb(0, *args)获取第一个结果作为模板，确定输出数组的行数和数据类型
    2. 基于模板预分配一个大的二维数组，列数为n * 单次输出列数
    3. 循环n次，每次调用apply_func_nb(i, *args)，将结果按列顺序存储到预分配数组中
    4. 返回最终的连接结果
    
    Args:
        n (int): 循环迭代的总次数，必须是正整数
               表示要执行的批量操作数量，对应不同的参数组合或数据分片
               典型值范围：1-10000（取决于计算复杂度和内存限制）
               
        apply_func_nb (tp.Callable): 要批量应用的Numba编译函数，必须满足以下要求：
                                   - 必须使用@njit装饰器进行Numba编译
                                   - 函数签名：apply_func_nb(i: int, *args) -> array_like
                                   - 第一个参数i：当前迭代索引（0到n-1）
                                   - 返回值：NumPy兼容数组（1维、2维或更高维）
                                   - 每次调用返回的数组形状必须一致（除了可能的列数）
                                   - 只支持Numba兼容的数据类型和操作
                                   
        *args: 传递给apply_func_nb的额外位置参数，必须是Numba兼容类型：
              - NumPy数组：np.ndarray（任意维度和数值类型）
              - 标量数值：int、float、complex及其NumPy对应类型
              - 元组：包含上述类型的元组（tuple）
              - 不支持：Python列表、字典、自定义对象、字符串等
    
    Returns:
        tp.Array2d: 二维NumPy数组，包含所有batch结果的列向连接
                   - 形状：(rows, n * cols_per_batch)
                   - rows: apply_func_nb单次返回结果的行数
                   - n: 批次数量
                   - cols_per_batch: apply_func_nb单次返回结果的列数
                   - 数据类型：与apply_func_nb返回结果保持一致
                   
                   数据排列方式：
                   - 第1次调用结果存储在列[0:cols_per_batch]
                   - 第2次调用结果存储在列[cols_per_batch:2*cols_per_batch]
                   - 第i次调用结果存储在列[i*cols_per_batch:(i+1)*cols_per_batch]
    """
    # 步骤1：获取第一次函数调用的结果作为输出模板
    # 这一步的目的是确定每次apply_func_nb调用返回结果的维度和数据类型
    # 以便后续预分配足够大小的输出数组，避免动态扩展带来的性能损失
    output_0 = to_2d_one_nb(apply_func_nb(0, *args))
    
    # 步骤2：基于第一次调用结果预分配最终输出数组
    # 数组维度设计：
    # - 行数：保持与单次调用结果相同（output_0.shape[0]）
    # - 列数：n次调用结果按列连接，所以是 n * 单次结果列数（n * output_0.shape[1]）
    # - 数据类型：保持与单次调用结果相同（output_0.dtype）
    # 使用np.empty而非np.zeros是因为所有位置都会被后续赋值覆盖，无需初始化为0
    output = np.empty((output_0.shape[0], n * output_0.shape[1]), dtype=output_0.dtype)
    
    # 步骤3：主循环 - 批量执行apply_func_nb并将结果存储到预分配数组中
    # 循环范围：[0, 1, 2, ..., n-1]，每个i对应一次函数调用
    for i in range(n):
        
        # 步骤3.1：处理第一次迭代的特殊情况
        # 由于我们已经在步骤1中计算了apply_func_nb(0, *args)的结果，
        # 为了避免重复计算，直接复用之前的结果output_0
        if i == 0:
            outputs_i = output_0  # 复用第一次调用的结果，避免重复计算提高效率
        else:
            # 步骤3.2：对于其他迭代，重新调用apply_func_nb计算当前索引i的结果
            # to_2d_one_nb确保返回结果是标准的2维数组格式，便于后续的数组切片赋值操作
            outputs_i = to_2d_one_nb(apply_func_nb(i, *args))
        
        # 步骤3.3：将当前迭代的结果存储到输出数组的对应列位置
        # 列索引计算逻辑：
        # - 起始列：i * outputs_i.shape[1]（第i个结果块的起始位置）
        # - 结束列：(i + 1) * outputs_i.shape[1]（第i个结果块的结束位置，不包含）
        # - 例如：第0个结果存储在列[0:cols], 第1个结果存储在列[cols:2*cols]
        # 使用切片赋值实现高效的内存拷贝，避免循环赋值的开销
        output[:, i * outputs_i.shape[1]:(i + 1) * outputs_i.shape[1]] = outputs_i
    
    return output


def apply_and_concat_multiple(n: int,
                              apply_func: tp.Callable, *args,
                              show_progress: bool = False,
                              tqdm_kwargs: tp.KwargsLike = None,
                              **kwargs) -> tp.List[tp.Array2d]:
    """
    批量应用函数并按组进行列连接的多输出版本核心工具函数
    
    该函数是vectorbt库中用于处理多输出批量计算的核心工具，与apply_and_concat_one的区别在于：
    apply_func每次调用返回多个数组（而非单个数组），函数会将相同位置的数组分别进行列连接。
    这在量化交易中特别有用，例如同时计算多个技术指标、多个策略的信号和收益等场景。
    
    函数工作流程：
    1. 对范围[0, n)内的每个整数i，调用apply_func(i, *args, **kwargs)
    2. 每次调用返回多个数组（如元组形式：(array1, array2, array3)）
    3. 将每个数组转换为2维格式，收集所有n次调用的结果
    4. 将相同位置的数组进行列连接（第一个位置的所有数组连接，第二个位置的所有数组连接...）
    5. 返回连接后的数组列表，每个元素对应原始输出的一个位置
    
    Args:
        n (int): 循环迭代的总次数，必须是正整数
               表示要执行的批量操作数量，对应不同的参数组合或数据分片
               典型应用：不同时间窗口、不同股票、不同策略参数的组合数量
               
        apply_func (tp.Callable): 要批量应用的可调用函数，必须满足以下要求：
                                 - 函数签名：apply_func(i: int, *args, **kwargs) -> Tuple[Array, Array, ...]
                                 - 第一个参数i：当前迭代索引（0到n-1）
                                 - 返回值：多个NumPy数组的可迭代对象（如元组、列表）
                                 - 每次调用返回的数组数量必须一致
                                 - 每个位置的数组行数必须一致（列数可以不同）
                                 
        *args: 传递给apply_func的额外位置参数
              可以是任何Python对象，常见类型包括：
              - NumPy数组（如价格数据、成交量数据、基准数据）
              - 策略配置对象、参数列表等
              
        show_progress (bool, optional): 是否显示进度条。默认为False
                                      - True: 显示tqdm进度条，适用于长时间运行的批量计算
                                      - False: 不显示进度条，适用于快速计算
                                      
        tqdm_kwargs (tp.KwargsLike, optional): 传递给tqdm进度条的配置参数字典
                                             常用参数包括：
                                             - desc: 进度条描述文本（如"计算多指标组合"）
                                             - unit: 进度单位（如"参数组合", "股票组合"）
                                             默认为None，内部会初始化为空字典
                                             
        **kwargs: 传递给apply_func的额外关键字参数
                 可以包含策略参数、计算配置、数据处理选项等
    
    Returns:
        tp.List[tp.Array2d]: 数组列表，每个元素是一个二维NumPy数组
                           - 列表长度：等于apply_func返回的数组数量
                           - 每个数组形状：(rows, total_cols)
                           - rows: apply_func单次返回中对应位置数组的行数
                           - total_cols: n次调用中对应位置数组的列数总和
                           
                           数据组织方式：
                           - 第k个数组：包含所有n次调用中第k个位置的数组按列连接的结果
                           - 列顺序：[第0次调用第k个数组 | 第1次调用第k个数组 | ... | 第n-1次调用第k个数组]
    """
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    
    outputs = []
    
    for i in tqdm(range(n), disable=not show_progress, **tqdm_kwargs):
        
        # 调用apply_func获取当前迭代的多个输出数组
        # 返回结果应该是多个数组的可迭代对象（如元组、列表）
        current_outputs = apply_func(i, *args, **kwargs)
        
        # 标准化当前迭代的所有输出数组为2维格式并转换为元组添加到 outputs
        outputs.append(tuple(map(reshape_fns.to_2d, current_outputs)))
    
    # 步骤4：重组数据结构并执行分组列连接
    # 这是整个函数最关键的数据重组步骤，将"按迭代分组"转换为"按输出位置分组"
    
    # outputs结构：[(第0次: 位置0数组, 位置1数组, ...), (第1次: 位置0数组, 位置1数组, ...), ...]
    # zip(*outputs)后：[(所有第0个位置数组), (所有第1个位置数组), ...]
    # 这样相同位置的数组就被分组到一起了
    transposed_outputs = list(zip(*outputs))
    
    # 对每组相同位置的数组执行列连接，然后转换为列表返回
    return list(map(np.column_stack, transposed_outputs))


@njit
def to_2d_multiple_nb(a: tp.Iterable[tp.Array]) -> tp.List[tp.Array2d]:
    """
    批量将多个数组转换为二维数组格式的Numba编译版本
    
    该函数是to_2d_one_nb的批量处理版本，专门用于处理包含多个NumPy数组的可迭代对象，
    将其中的每个数组统一转换为二维数组格式。这在vectorbt库的批量数据处理和
    多输出函数结果标准化中起到关键作用，确保所有数组具有一致的维度结构。
    
    函数工作原理：
    1. 遍历输入的可迭代对象中的每个数组
    2. 对每个数组调用to_2d_one_nb函数进行维度转换：
       - 0维标量 () -> 2维数组 (1, 1)
       - 1维数组 (n,) -> 2维数组 (n, 1)  
       - 2维及以上数组 -> 保持原样
    3. 将所有转换后的2维数组收集到列表中返回
    
    Args:
        a (tp.Iterable[tp.Array]): 包含多个NumPy数组的可迭代对象
                                 可以是以下类型：
                                 - tuple: 元组形式的数组集合，如 (array1, array2, array3)
                                 - list: 列表形式的数组集合，如 [array1, array2, array3]  
                                 - 其他可迭代对象: 任何支持迭代的数组容器
                                 
                                 重要约束条件：
                                 - 必须是严格同质的（strictly homogeneous），即所有数组必须具有
                                   相同的数据类型（dtype），这是Numba编译的要求
                                 - 每个元素必须是NumPy兼容的数组类型
                                 - 数组的形状可以不同，但数据类型必须一致
    
    Returns:
        tp.List[tp.Array2d]: 转换后的二维数组列表
                           - 列表长度等于输入可迭代对象的元素数量
                           - 每个元素都是二维NumPy数组（shape为(rows, cols)）
                           - 保持原始数组的数据类型不变
                           - 列表中数组的顺序与输入顺序保持一致
    """
    lst = list()
    
    for _a in a:
        lst.append(to_2d_one_nb(_a))
        
    return lst


@njit
def apply_and_concat_multiple_nb(n: int, apply_func_nb: tp.Callable, *args) -> tp.List[tp.Array2d]:
    """
    apply_and_concat_multiple函数的Numba高性能编译版本，用于批量应用多输出函数并按组进行列连接
    
    函数工作原理：
    1. 调用apply_func_nb(0, *args)获取第一个结果作为模板，确定输出数组的数量、维度和数据类型
    2. 基于模板为每个输出位置预分配大的二维数组，避免动态扩展的性能损失
    3. 循环n次，每次调用apply_func_nb(i, *args)获取多个输出数组
    4. 将每次调用中相同位置的数组按列顺序存储到对应的预分配数组中
    5. 返回包含所有连接结果的数组列表
    
    Args:
        n (int): 循环迭代的总次数，必须是正整数
               表示要执行的批量操作数量，对应不同的参数组合或数据分片
               
        apply_func_nb (tp.Callable): 要批量应用的Numba编译函数，必须满足以下严格要求：
                                   - 必须使用@njit装饰器进行Numba编译
                                   - 函数签名：apply_func_nb(i: int, *args) -> Tuple[Array, Array, ...]
                                   - 第一个参数i：当前迭代索引（0到n-1）
                                   - 返回值：多个NumPy数组的严格同质化元组（所有数组dtype必须相同）
                                   - 每次调用返回的数组数量必须完全一致
                                   - 每个位置的数组行数必须一致（列数可以不同）
                                   - 只支持Numba兼容的数据类型和操作
                                   
        *args: 传递给apply_func_nb的额外位置参数，必须是Numba兼容类型：
              - NumPy数组：np.ndarray（任意维度和数值类型）
              - 标量数值：int、float、complex及其NumPy对应类型
              - 元组：包含上述类型的元组（tuple）
              - 不支持：Python列表、字典、自定义对象、字符串等
    
    Returns:
        tp.List[tp.Array2d]: 包含所有连接结果的二维数组列表
                           - 列表长度：等于apply_func_nb返回的数组数量
                           - 每个数组形状：(rows, total_cols)
                           - rows: apply_func_nb单次返回中对应位置数组的行数
                           - total_cols: n次调用中对应位置数组的列数总和
                           
                           数据组织方式：
                           - 第k个数组：包含所有n次调用中第k个位置的数组按列连接的结果
                           - 列顺序：[第0次调用第k个数组 | 第1次调用第k个数组 | ... | 第n-1次调用第k个数组]
    """
    outputs = list()
    
    # 获取第一次函数调用的结果作为模板，这一步的目的是确定输出数组的数量、每个数组的维度信息和数据类型
    outputs_0 = to_2d_multiple_nb(apply_func_nb(0, *args))
    
    # 为每个输出位置预分配内存，避免动态扩展的开销
    for j in range(len(outputs_0)):
        # outputs_0[j].shape[0] * outputs_0[j].shape[1]
        outputs.append(np.empty((outputs_0[j].shape[0], n * outputs_0[j].shape[1]), dtype=outputs_0[j].dtype))
    
    for i in range(n):
        if i == 0:
            outputs_i = outputs_0  
        else:
            outputs_i = to_2d_multiple_nb(apply_func_nb(i, *args))
        
        for j in range(len(outputs_i)):
            outputs[j][:, i * outputs_i[j].shape[1]:(i + 1) * outputs_i[j].shape[1]] = outputs_i[j]
    
    return outputs


def select_and_combine(i: int,
                       obj: tp.Any,
                       others: tp.Sequence,
                       combine_func: tp.Callable,
                       *args, **kwargs) -> tp.AnyArray:
    return combine_func(obj, others[i], *args, **kwargs)


def combine_and_concat(obj: tp.Any,
                       others: tp.Sequence,
                       combine_func: tp.Callable,
                       *args, **kwargs) -> tp.Array2d:
    """
    将一个基准对象obj与多个目标对象others逐一组合来调用combine_func，然后将所有组合结果按列连接成一个二维数组
    
    函数工作流程：
    1. 使用apply_and_concat_one作为底层引擎进行批量处理
    2. 对others序列中的每个元素（索引0到len(others)-1），调用select_and_combine
    3. select_and_combine函数选择others[i]并使用combine_func与obj组合
    4. 收集所有组合结果并按列连接成二维数组返回
    
    Args:
        obj (tp.Any): 要进行组合的基准对象（参考对象）
                     
        others (tp.Sequence): 包含多个待比较对象的序列
                             序列中的每个元素都将与obj进行组合比较
                             
        combine_func (tp.Callable): 用于组合obj和others中每个元素的函数
                                  函数必须支持以下调用方式：
                                  combine_func(obj, other_element, *args, **kwargs)
                                  
                                  量化交易中常用的组合函数：
                                  - 收益比较：lambda base, strategy: strategy - base
                                  - 相关性计算：numpy.corrcoef
                                  - 比率计算：lambda x, y: y / x
                                  - 性能指标：sharpe_ratio, max_drawdown等
                                  - 数组运算：numpy.add, numpy.multiply等
                                  
        *args: 传递给combine_func的额外位置参数
              这些参数在每次调用combine_func时都会传递
              例如：计算参数、配置选项等
              
        **kwargs: 传递给combine_func的额外关键字参数
                 可以包含：
                 - 计算配置：如window_size、method等
                 - 显示选项：如show_progress=True
                 - tqdm配置：如tqdm_kwargs={"desc": "组合分析"}
    
    Returns:
        tp.Array2d: 二维NumPy数组，包含所有组合结果的列连接
                   - 形状：(rows, len(others) * cols_per_result)
                   - rows：单次组合结果的行数
                   - len(others)：组合的总数（others序列长度）
                   - cols_per_result：单次组合结果的列数
                   
                   数据排列方式：
                   - 第0列组：obj与others[0]的组合结果
                   - 第1列组：obj与others[1]的组合结果
                   - ...
                   - 第n列组：obj与others[n-1]的组合结果
    """
    return apply_and_concat_one(len(others), select_and_combine, obj, others, combine_func, *args, **kwargs)


@njit
def select_and_combine_nb(i: int, obj: tp.Any, others: tp.Sequence, combine_func_nb: tp.Callable, *args) -> tp.Array:
    return combine_func_nb(obj, others[i], *args)


@njit
def combine_and_concat_nb(obj: tp.Any, others: tp.Sequence, combine_func_nb: tp.Callable, *args) -> tp.Array2d:
    return apply_and_concat_one_nb(len(others), select_and_combine_nb, obj, others, combine_func_nb, *args)


def combine_multiple(objs: tp.Sequence, combine_func: tp.Callable, *args, **kwargs) -> tp.AnyArray:
    """
    相当于　combine_func(...combine_func(combine_func(obj₀, obj₁), obj₂)..., objₙ)
    
    函数执行流程：
    1. 将第一个对象objs[0]作为初始结果
    2. 从第二个对象开始，依次与当前结果进行组合：
       - 第1次组合：result = combine_func(objs[0], objs[1], *args, **kwargs)
       - 第2次组合：result = combine_func(result, objs[2], *args, **kwargs)  
       - 第3次组合：result = combine_func(result, objs[3], *args, **kwargs)
       - ...以此类推直到所有对象都被组合
    3. 返回最终组合结果
    
    Args:
        objs (tp.Sequence): 要进行组合的对象序列，必须满足以下要求：
                          - 序列长度至少为1（不能为空序列）
                          - 序列中的对象类型应与combine_func的输入要求兼容
                          - 对象类型可以是：NumPy数组、DataFrame、策略对象、指标计算结果等
        
        combine_func (tp.Callable): 用于组合两个对象的可调用函数，必须满足以下签名：
                                  combine_func(obj1, obj2, *args, **kwargs) -> combined_result
                                  
                                  函数要求：
                                  - 第一个参数：左操作数（当前累积结果或初始对象）
                                  - 第二个参数：右操作数（序列中的下一个对象）
                                  - 返回值：组合后的结果，类型应与输入对象兼容
                                  
                                  常用组合函数示例：
                                  - 数组加法：numpy.add 或 lambda x, y: x + y
                                  - 数组乘法：numpy.multiply 或 lambda x, y: x * y
        
        *args: 传递给combine_func的额外位置参数
              这些参数在每次调用combine_func时都会保持不变
              例如：
              - 数值计算参数：权重、系数、阈值等
              - 配置参数：计算模式、处理选项等
              - 其他辅助数据：参考数组、掩码数组等
        
        **kwargs: 传递给combine_func的额外关键字参数
                 这些参数在每次调用combine_func时都会保持不变
                 例如：
                 - axis: 指定计算轴向（如axis=0表示按列计算）
                 - method: 指定计算方法（如method='linear'）
                 - weights: 权重参数
                 - fillna: 缺失值处理方式
    
    Returns:
        tp.AnyArray: 最终组合得到的单一对象
                    - 数据类型：通常与输入对象类型一致，但也可能根据combine_func而变化
                    - 数据形状：由combine_func的逻辑决定，可能保持原形状或产生新形状
                    - 内容：经过所有对象顺序组合后的最终结果
    """
    result = objs[0]
    
    for i in range(1, len(objs)):
        result = combine_func(result, objs[i], *args, **kwargs)
    
    return result


@njit
def combine_multiple_nb(objs: tp.Sequence, combine_func_nb: tp.Callable, *args) -> tp.Array:
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func_nb(result, objs[i], *args)
    return result


def ray_apply(n: int,
              apply_func: tp.Callable, *args,
              ray_force_init: bool = False,
              ray_func_kwargs: tp.KwargsLike = None,
              ray_init_kwargs: tp.KwargsLike = None,
              ray_shutdown: bool = False,
              **kwargs) -> tp.List[tp.AnyArray]:
    """
    使用Ray分布式计算框架并行执行批量函数应用操作的核心工具函数
    
    该函数是vectorbt库中进行大规模并行计算的核心工具，特别适用于量化交易中的
    超参数扫描、批量回测、分布式特征工程等计算密集型任务。通过Ray框架将
    单一apply_func函数的n次调用分布到多个CPU核心或机器上并行执行，
    显著提升计算性能。
    
    函数工作流程：
    1. 动态导入Ray库并初始化分布式计算环境
    2. 将apply_func转换为Ray远程函数（remote function）
    3. 将大型数据对象存储到Ray共享对象存储中，避免重复序列化传输
    4. 创建n个并行任务，每个任务执行apply_func(i, *args, **kwargs)
    5. 异步提交所有任务并等待完成，收集结果
    6. 可选择性地关闭Ray运行时环境
    
    Args:
        n (int): 并行任务的总数量，必须是正整数
               表示要执行apply_func(i, *args, **kwargs)的次数，其中i从0到n-1
               典型取值范围：10-10000（取决于任务复杂度和集群规模）
               
        apply_func (tp.Callable): 要并行执行的目标函数，必须满足以下要求：
                                 - 函数签名：apply_func(i: int, *args, **kwargs) -> Any
                                 - 第一个参数i：任务索引（0到n-1），用于区分不同的并行任务
                                 - 后续参数：*args和**kwargs传递的共享参数
                                 - 返回值：任意类型的计算结果
                                 - 函数必须是可序列化的（支持pickle），避免使用局部函数或闭包
                                 
                                 量化交易中的典型应用函数：
                                 - 策略回测：backtest_strategy(i, params[i], price_data, **config)
                                 - 技术指标计算：calc_indicators(i, symbols[i], data, period, **opts)
                                 - 风险计算：calc_risk_metrics(i, portfolios[i], benchmark, **kwargs)
                                 
        *args: 传递给apply_func的位置参数，所有并行任务共享这些参数
              这些参数会被存储到Ray对象存储中，避免每次任务执行时重复传输
              常见类型：
              - 大型NumPy数组（如价格矩阵、成交量数据）
              - 配置对象、参数列表
              - 预处理好的数据集
              
        ray_force_init (bool, optional): 是否强制重新初始化Ray运行时环境。默认为False
                                        - True: 如果Ray已经初始化，则先关闭再重新初始化
                                               适用于需要更改Ray配置或清理状态的场景
                                        - False: 如果Ray未初始化才进行初始化，保持现有Ray环境
                                                适用于多次调用ray_apply的场景
                                        
        ray_func_kwargs (tp.KwargsLike, optional): 传递给ray.remote()的配置参数字典
                                                 用于控制远程函数的资源分配和执行特性
                                                 常用配置：
                                                 - num_cpus: 每个任务使用的CPU核心数（如num_cpus=1）
                                                 - num_gpus: 每个任务使用的GPU数量（如num_gpus=0.5）
                                                 - memory: 每个任务的内存限制（如memory=2000*1024*1024）
                                                 - max_retries: 任务失败时的最大重试次数（如max_retries=3）
                                                 默认为None，内部会初始化为空字典
                                                 
        ray_init_kwargs (tp.KwargsLike, optional): 传递给ray.init()的初始化参数字典
                                                 用于配置Ray集群的连接和资源设置
                                                 常用配置：
                                                 - address: Ray集群地址（如"ray://head-node:10001"）
                                                 - num_cpus: 本地Ray实例的CPU核心数
                                                 - num_gpus: 本地Ray实例的GPU数量
                                                 - object_store_memory: 对象存储内存大小
                                                 - log_to_driver: 是否将日志输出到驱动程序
                                                 默认为None，内部会初始化为空字典
                                                 
        ray_shutdown (bool, optional): 是否在任务完成后关闭Ray运行时环境。默认为False
                                      - True: 任务完成后立即关闭Ray，释放所有相关资源
                                             适用于单次大型计算任务
                                      - False: 保持Ray运行，适用于需要多次调用的场景
                                              
        **kwargs: 传递给apply_func的关键字参数，所有并行任务共享这些参数
                 这些参数同样会被存储到Ray对象存储中进行优化传输
                 常见用途：
                 - 计算配置：window_size、method、threshold等
                 - 数据处理选项：fillna、normalize、scaling等
                 - 算法参数：learning_rate、epochs、batch_size等
    
    Returns:
        tp.List[tp.AnyArray]: 所有并行任务结果的列表
                            - 列表长度：等于n（任务数量）
                            - 列表顺序：results[i]对应apply_func(i, *args, **kwargs)的结果
                            - 元素类型：与apply_func的返回值类型一致
                            - 数据结构：保持apply_func原始返回值的结构和类型
                            
                            结果组织方式：
                            - results[0]: apply_func(0, *args, **kwargs)的结果
                            - results[1]: apply_func(1, *args, **kwargs)的结果
                            - ...
                            - results[n-1]: apply_func(n-1, *args, **kwargs)的结果
    """
    # 动态导入Ray分布式计算框架
    # 使用动态导入而非模块级导入是为了避免在未安装Ray的环境中引发ImportError
    # 这种设计模式允许vectorbt在没有Ray的环境中正常运行其他功能
    import ray

    if ray_init_kwargs is None:
        ray_init_kwargs = {}
    
    if ray_func_kwargs is None:
        ray_func_kwargs = {}
    
    # 当ray_force_init为True时，无论Ray当前状态如何都要重新初始化
    if ray_force_init:
        if ray.is_initialized():
            ray.shutdown()
    
    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)
    
    # 将普通Python函数转换为Ray远程函数（Remote Function）
    if len(ray_func_kwargs) > 0:
        apply_func = ray.remote(**ray_func_kwargs)(apply_func)
    else:
        apply_func = ray.remote(apply_func)
    
    # 将共享参数存储到Ray分布式对象存储中
    arg_refs = ()  
    for v in args:
        # 将当前参数v存储到Ray对象存储，获取对象引用并添加到arg_refs元组
        # 这避免了每次远程调用时都序列化和传输大型对象
        arg_refs += (ray.put(v),)
    
    kwarg_refs = {}  
    for k, v in kwargs.items():
        kwarg_refs[k] = ray.put(v)
    
    # 创建并提交所有并行任务到Ray集群
    # 使用列表推导式批量创建n个异步任务，每个任务对应一个不同的索引i
    # apply_func.remote()创建异步任务并立即返回Future对象，不阻塞主线程
    futures = [apply_func.remote(i, *arg_refs, **kwarg_refs) for i in range(n)]
    
    # 等待所有并行任务完成并收集结果
    # ray.get()是阻塞操作，等待所有Future对象完成并返回结果列表
    # Ray会自动处理任务调度、负载均衡、失败重试等分布式计算细节
    # 结果列表的顺序与futures列表的顺序一致，即results[i]对应futures[i]的结果
    results = ray.get(futures)
    
    # 如果设置了ray_shutdown=True，则在任务完成后关闭Ray运行时
    # 这会释放所有分布式资源，包括worker进程、对象存储、网络连接等
    if ray_shutdown:
        ray.shutdown()
    
    return results


def apply_and_concat_one_ray(*args, **kwargs) -> tp.Array2d:
    """
    apply_and_concat_one函数的Ray分布式并行版本，用于大规模批量函数应用和列连接操作
    
    函数工作流程：
    1. 调用ray_apply进行分布式并行计算，获取n个并行任务的结果列表
    2. 将每个结果转换为标准的2维数组格式（使用reshape_fns.to_2d）
    3. 使用NumPy的column_stack将所有结果按列连接成单一的2维数组
    4. 返回连接后的最终结果数组
    
    Args:
        *args: 传递给ray_apply的位置参数，具体包括：
              - n (int): 并行任务数量，对应apply_func的调用次数
              - apply_func (tp.Callable): 要并行执行的目标函数
              - 其他位置参数：传递给apply_func的共享参数
              
        **kwargs: 传递给ray_apply的关键字参数，包括：
                 - ray_force_init (bool): 是否强制重新初始化Ray环境
                 - ray_func_kwargs (dict): Ray远程函数的配置参数
                 - ray_init_kwargs (dict): Ray初始化参数
                 - ray_shutdown (bool): 是否在完成后关闭Ray
                 - 以及传递给apply_func的其他关键字参数
    """
    # 使用Ray分布式框架并行执行批量任务
    # 返回results列表，其中results[i]对应第i次函数调用的结果
    results = ray_apply(*args, **kwargs)
    standardized_results = list(map(reshape_fns.to_2d, results))
    # 将所有标准化结果按列连接成单一的2维数组
    # 例如：[array([[1], [2]]), array([[3], [4]])] -> array([[1, 3], [2, 4]])
    return np.column_stack(standardized_results)


def apply_and_concat_multiple_ray(*args, **kwargs) -> tp.List[tp.Array2d]:
    """
    apply_and_concat_multiple函数的Ray分布式并行版本，用于多输出批量函数的分布式执行和分组连接
    
    函数工作流程：
    1. 调用ray_apply进行分布式并行计算，获取n个并行任务的结果列表
    2. 每个任务返回多个数组的元组或列表（如(array1, array2, array3)）
    3. 重组数据结构，将相同位置的数组分组到一起
    4. 对每组相同位置的数组执行列连接操作
    5. 返回连接后的数组列表，每个元素对应原始输出的一个位置
    
    Args:
        *args: 传递给ray_apply的位置参数，具体包括：
              - n (int): 并行任务数量，对应apply_func的调用次数
              - apply_func (tp.Callable): 要并行执行的多输出目标函数
                该函数必须返回多个数组的可迭代对象（如元组、列表）
                每次调用返回的数组数量必须一致
              - 其他位置参数：传递给apply_func的共享参数
              
        **kwargs: 传递给ray_apply的关键字参数，包括：
                 - ray_force_init (bool): 是否强制重新初始化Ray环境
                 - ray_func_kwargs (dict): Ray远程函数的配置参数
                 - ray_init_kwargs (dict): Ray初始化参数
                 - ray_shutdown (bool): 是否在完成后关闭Ray
                 - 以及传递给apply_func的其他关键字参数
    """
    results = ray_apply(*args, **kwargs)
    # 原始结构：results = [(任务0的输出1, 任务0的输出2, ...), (任务1的输出1, 任务1的输出2, ...), ...]
    # 目标结构：[(所有任务的输出1), (所有任务的输出2), ...]
    transposed_results = list(zip(*results))
    concatenated_results = list(map(np.column_stack, transposed_results))
    return concatenated_results


def combine_and_concat_ray(obj: tp.Any,
                           others: tp.Sequence,
                           combine_func: tp.Callable,
                           *args, **kwargs) -> tp.Array2d:
    return apply_and_concat_one_ray(len(others), 
                                    select_and_combine, 
                                    obj, 
                                    others, 
                                    combine_func, 
                                    *args, 
                                    **kwargs)

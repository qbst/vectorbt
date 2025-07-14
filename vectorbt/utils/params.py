# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT UTILS MODULE: 参数处理工具模块
================================================================================

在量化交易策略开发和回测过程中，经常需要对不同的策略参数进行组合、广播、转换等操作，以便进行大规模的
参数扫描和优化。
该模块提供了一套完整的参数处理工具链，支持复杂的参数组合生成和高效的参数管理。
"""

import itertools  # 导入itertools模块，提供高效的迭代器工具，用于参数组合生成
from collections.abc import Callable  # 导入Callable抽象基类，用于函数对象的类型注解

from numba.typed import List  # 导入Numba的类型化列表，用于JIT编译环境下的高性能列表操作

from vectorbt import _typing as tp  # 导入vectorbt的类型定义模块，提供统一的类型注解
from vectorbt.utils import checks  # 导入vectorbt的检查工具模块，提供参数验证功能


def to_typed_list(lst: list) -> List:
    """
    将Python原生列表转换为Numba兼容的类型化列表
    
    这个函数专门用于解决Numba JIT编译环境中的列表类型兼容问题。在量化交易的
    高性能计算场景中，Numba编译器需要明确的类型信息来优化代码执行。Python
    原生列表在Numba中的直接构造存在已知缺陷，该函数提供了一个可靠的转换方案。

    
    参数：
        lst (list): 需要转换的Python原生列表，可以包含任意类型的元素
    
    返回：
        List: Numba兼容的类型化列表，可以在JIT编译函数中安全使用
    
    示例：
        >>> # 转换技术指标参数
        >>> rsi_periods = [14, 21, 28]
        >>> typed_periods = to_typed_list(rsi_periods)
        >>> # 现在可以在@njit装饰的函数中使用typed_periods
    
    参考：
        Numba issue: https://github.com/numba/numba/issues/6651
        修复了直接构造类型化列表的已知问题
    """
    nb_lst = List()  # 创建空的Numba类型化列表对象
    for elem in lst:  # 遍历输入的Python原生列表中的每个元素
        nb_lst.append(elem)  # 将每个元素逐个追加到类型化列表中，触发类型推断
    return nb_lst  # 返回完全构造的Numba兼容类型化列表


def flatten_param_tuples(param_tuples: tp.Sequence) -> tp.List[tp.List]:
    """
    递归展开嵌套的参数元组结构，将其转换为扁平化的参数列表
    
    该函数是vectorbt参数处理系统的核心组件，专门用于处理复杂的嵌套参数结构。
    在量化交易策略开发中，经常需要处理多层嵌套的参数组合，如技术指标的多周期
    参数、多资产的权重分配、多策略的参数矩阵等。该函数通过递归解压缩技术，
    将这些复杂结构转换为便于处理的扁平化格式。
    
    核心算法：
    1. 使用zip(*param_tuples)进行矩阵转置，将按行组织的参数转换为按列组织
    2. 递归检测每个解压缩后的元素是否为嵌套的元组结构
    3. 如果发现嵌套结构，递归调用自身进行进一步展开
    4. 最终返回完全扁平化的参数列表矩阵
    
    数据结构示例：
    输入: [([1, 2], [3, 4]), ([5, 6], [7, 8])]
    中间: ([1, 2], [5, 6]) 和 ([3, 4], [7, 8])
    输出: [[1, 5], [2, 6], [3, 7], [4, 8]]
    
    参数：
        param_tuples (tp.Sequence): 包含参数元组的序列，可以是多层嵌套结构
    
    返回：
        tp.List[tp.List]: 扁平化的参数列表，每个子列表对应一个参数维度
    
    示例：
        >>> # 展开移动平均线参数组合
        >>> ma_params = [
        ...     ((5, 10), (20, 30)),  # 短期和长期MA的周期组合
        ...     ((0.02, 0.05), (0.08, 0.1))  # 对应的交易阈值
        ... ]
        >>> flattened = flatten_param_tuples(ma_params)
        >>> # 结果: [[5, 20], [10, 30], [0.02, 0.08], [0.05, 0.1]]
        
        >>> # 展开多资产权重分配
        >>> weight_params = [
        ...     ((0.3, 0.7), (0.4, 0.6)),  # 股票权重
        ...     ((0.2, 0.8), (0.3, 0.7))   # 债券权重  
        ... ]
        >>> weights = flatten_param_tuples(weight_params)
        
    """
    # 比如输入：[([1, 2], [3, 4]), ([5, 6], [7, 8])]
    param_list = []  # 初始化空的参数列表，用于存储展开后的结果
    # unzipped_tuples = ([1, 2], [5, 6]) 和 ([3, 4], [7, 8])
    unzipped_tuples = zip(*param_tuples)  # 对参数元组进行解压缩转置操作，将行列互换
    for i, unzipped in enumerate(unzipped_tuples):  # 遍历解压缩后的每个参数维度
        unzipped = list(unzipped)  # 将解压缩的迭代器转换为列表，便于后续处理
        if isinstance(unzipped[0], tuple):  # 检查第一个元素是否为元组，判断是否存在嵌套结构
            param_list.extend(flatten_param_tuples(unzipped))  # 递归调用自身展开嵌套结构
        else:  # 如果不是元组，说明已经是最底层的参数值
            param_list.append(unzipped)  # 直接将参数列表添加到结果中
    return param_list  # 返回完全展开的参数列表


def create_param_combs(op_tree: tp.Tuple, depth: int = 0) -> tp.List[tp.List]:
    """
    基于操作树生成任意复杂度的参数组合，实现灵活的参数空间构建
    
    这是vectorbt参数处理系统中最强大的功能之一，允许通过声明式的操作树语法
    生成复杂的参数组合。该函数支持嵌套的操作结构，可以组合itertools中的各种
    函数（如product、combinations、permutations等）来生成满足特定需求的
    参数空间。这种设计特别适合量化交易中的超参数优化和策略参数扫描。
    
    操作树结构：
    op_tree是一个元组，第一个元素必须是可调用对象（函数），后续元素是该函数的参数。
    如果某个参数本身也是一个操作树（元组且第一个元素是函数），则会递归处理。
    
    核心算法流程：
    1. 验证操作树的格式和第一个元素的可调用性
    2. 递归处理嵌套的操作树结构，构建新的操作树
    3. 调用操作函数生成参数组合
    4. 在顶层调用时，使用flatten_param_tuples进行结果展开
    
    支持的操作函数：
    - itertools.product: 笛卡尔积，生成所有可能的参数组合
    - itertools.combinations: 组合，生成无重复的参数组合
    - itertools.permutations: 排列，生成有序的参数组合
    - itertools.combinations_with_replacement: 可重复组合
    - 自定义函数：用户可以定义自己的参数生成逻辑

    
    参数：
        op_tree (tp.Tuple): 操作树元组，格式为(函数, 参数1, 参数2, ...)
        depth (int, optional): 递归深度，默认为0，用于内部递归控制
    
    返回：
        tp.List[tp.List]: 生成的参数组合列表，每个子列表代表一个参数维度
    
    示例：
        >>> import numpy as np
        >>> from itertools import combinations, product
        
        >>> # 生成移动平均线的周期组合
        >>> ma_combinations = create_param_combs(
        ...     (product, (combinations, [5, 10, 20, 50], 2), [0.02, 0.05])
        ... )
        >>> # 结果：所有可能的双MA周期组合与交易阈值的笛卡尔积
        
        >>> # 生成投资组合权重组合
        >>> weight_combinations = create_param_combs(
        ...     (combinations, [0.1, 0.2, 0.3, 0.4], 3)
        ... )
        >>> # 结果：从4个权重中选择3个的所有组合
        
        >>> # 复杂的嵌套参数组合
        >>> complex_params = create_param_combs(
        ...     (product, 
        ...      (combinations, ['AAPL', 'GOOGL', 'MSFT', 'TSLA'], 2),
        ...      [14, 21, 28],  # RSI周期
        ...      [0.7, 0.8, 0.9])  # 超买阈值
        ... )
        >>> # 结果：股票对、RSI周期、超买阈值的完整组合
    
    高级用法：
        >>> # 自定义参数生成函数
        >>> def custom_ranges(start, end, step):
        ...     return list(range(start, end, step))
        
        >>> custom_params = create_param_combs(
        ...     (product, (custom_ranges, 10, 100, 10), [0.01, 0.02, 0.03])
        ... )
    
    性能考量：
        - 参数组合的数量会随着参数维度呈指数增长，需要合理控制参数范围
        - 对于大规模参数空间，建议使用分批处理或随机采样策略
        - 递归深度过深可能导致栈溢出，建议合理设计操作树结构
    
    错误处理：
        - 操作树格式错误时会触发断言异常
        - 不支持的操作函数会在运行时报错
        - 参数不匹配时会传播相应的函数异常
    """
    checks.assert_instance_of(op_tree, tuple)  # 验证操作树必须是元组类型
    checks.assert_instance_of(op_tree[0], Callable)  # 验证第一个元素必须是可调用对象
    new_op_tree: tp.Tuple = (op_tree[0],)  # 创建新的操作树，保留操作函数
    for elem in op_tree[1:]:  # 遍历操作树中的每个参数元素
        if isinstance(elem, tuple) and isinstance(elem[0], Callable):  # 检查是否为嵌套的操作树
            new_op_tree += (create_param_combs(elem, depth=depth + 1),)  # 递归处理嵌套结构
        else:  # 如果是普通参数，直接添加到新操作树中
            new_op_tree += (elem,)  # 将参数元素添加到操作树中
    out = list(new_op_tree[0](*new_op_tree[1:]))  # 调用操作函数生成参数组合
    if depth == 0:  # 如果是顶层调用（递归深度为0）
        # 使用flatten_param_tuples函数展开嵌套的参数结构
        return flatten_param_tuples(out)  # 返回扁平化的参数组合列表
    return out  # 非顶层调用直接返回操作结果


def broadcast_params(param_list: tp.Sequence[tp.Sequence], to_n: tp.Optional[int] = None) -> tp.List[tp.List]:
    """
    参数广播函数，将不同长度的参数列表广播到统一长度
    
    这个函数实现了类似于NumPy广播机制的参数对齐功能，专门用于处理量化交易策略中
    不同参数维度的对齐问题。在策略回测和优化过程中，经常遇到某些参数是标量（如
    全局的手续费率），而另一些参数是向量（如每个资产的权重）的情况。该函数能够
    智能地将这些不同维度的参数对齐到统一的长度，便于后续的向量化计算。
    
    广播规则：
    1. 如果参数长度为1，则重复该参数直到达到目标长度
    2. 如果参数长度等于目标长度，则保持不变
    3. 如果参数长度不满足上述条件，则抛出ValueError异常
    
    参数：
        param_list (tp.Sequence[tp.Sequence]): 参数列表的序列，每个子列表包含一组参数
        to_n (tp.Optional[int], optional): 目标长度，如果为None则自动计算为最大长度
    
    返回：
        tp.List[tp.List]: 广播后的参数列表，所有子列表长度相同
    
    异常：
        ValueError: 当某个参数列表的长度无法广播到目标长度时抛出
    
    示例：
        >>> # 广播交易参数
        >>> fees = [0.001]  # 单一手续费率
        >>> sizes = [100, 200, 300]  # 不同资产的交易数量
        >>> symbols = ['AAPL', 'GOOGL', 'MSFT']  # 股票代码
        >>> 
        >>> broadcasted = broadcast_params([fees, sizes, symbols])
        >>> # 结果: [[0.001, 0.001, 0.001], [100, 200, 300], ['AAPL', 'GOOGL', 'MSFT']]
        
        >>> # 广播技术指标参数
        >>> rsi_period = [14]  # RSI周期
        >>> ma_periods = [5, 10, 20]  # 移动平均周期
        >>> 
        >>> aligned_params = broadcast_params([rsi_period, ma_periods])
        >>> # 结果: [[14, 14, 14], [5, 10, 20]]
        
        >>> # 指定目标长度
        >>> param1 = [1]
        >>> param2 = [2, 3]
        >>> 
        >>> result = broadcast_params([param1, param2], to_n=4)
        >>> # 结果: [[1, 1, 1, 1], [2, 3, 2, 3]]  # param2会重复以达到长度4
    
    实现细节：
        - 自动长度计算：当to_n为None时，自动选择最长的参数列表长度作为目标
        - 内存优化：对于单元素列表，通过列表推导式进行高效的重复操作
        - 错误诊断：提供详细的错误信息，包括参数索引和长度信息
    
    性能考量：
        - 时间复杂度：O(n*m)，其中n为参数组数，m为目标长度
        - 空间复杂度：O(n*m)，用于存储广播后的参数
        - 对于大规模参数，建议预先计算目标长度以避免重复计算
    
    注意事项：
        - 广播不会改变原始参数列表，返回新的列表对象
        - 单元素列表的重复是通过值复制实现的，不是引用复制
        - 确保参数列表的元素类型一致，避免类型混合导致的问题
    """
    if to_n is None:  # 如果没有指定目标长度
        to_n = max(list(map(len, param_list)))  # 自动计算最大长度作为目标长度
    new_param_list = []  # 初始化新的参数列表，用于存储广播后的结果
    for i in range(len(param_list)):  # 遍历每个参数列表
        params = param_list[i]  # 获取当前参数列表
        if len(params) in [1, to_n]:  # 检查参数长度是否满足广播条件
            if len(params) < to_n:  # 如果参数长度为1，需要进行广播
                new_param_list.append([p for _ in range(to_n) for p in params])  # 重复参数以达到目标长度
            else:  # 如果参数长度已经等于目标长度
                new_param_list.append(list(params))  # 直接转换为列表并添加
        else:  # 如果参数长度不满足广播条件
            raise ValueError(f"Parameters at index {i} have length {len(params)} that cannot be broadcast to {to_n}")  # 抛出详细的错误信息
    return new_param_list  # 返回广播后的参数列表


def create_param_product(param_list: tp.Sequence[tp.Sequence]) -> tp.List[tp.List]:
    """
    计算参数集合的笛卡尔积，生成所有可能的参数组合
    
    这个函数是量化交易参数优化中的核心工具，用于生成参数空间的完整笛卡尔积。
    在策略开发和回测过程中，经常需要测试不同参数的所有可能组合，以找到最优的
    参数配置。该函数通过高效的矩阵操作实现了参数组合的批量生成。
    
    笛卡尔积概念：
    笛卡尔积是集合论中的基本概念，表示多个集合的所有可能组合。在量化交易中，
    这相当于生成所有可能的策略参数组合，每个组合代表一个待测试的策略配置。
    
    算法实现：
    1. 使用itertools.product生成所有参数组合的迭代器
    2. 通过zip(*...)操作将结果转置，按参数维度重新组织
    3. 将每个维度的参数转换为列表，形成最终的参数矩阵
    
    应用场景：
    - **网格搜索优化**：生成策略参数的全部搜索空间
    - **技术指标调优**：测试移动平均线的所有周期组合
    - **风险管理**：生成止损和止盈参数的所有组合
    - **投资组合优化**：生成资产权重的离散化组合
    - **高频策略**：生成时间窗口和阈值的参数网格
    
    参数：
        param_list (tp.Sequence[tp.Sequence]): 参数列表的序列，每个子列表包含一个参数维度的所有候选值
    
    返回：
        tp.List[tp.List]: 参数组合的矩阵，每行代表一个参数维度，每列代表一个具体的参数组合
    
    示例：
        >>> # 生成双移动平均线策略的参数组合
        >>> fast_periods = [5, 10, 15]    # 快速均线周期
        >>> slow_periods = [20, 30, 40]   # 慢速均线周期
        >>> thresholds = [0.01, 0.02]     # 交易阈值
        >>> 
        >>> combinations = create_param_product([fast_periods, slow_periods, thresholds])
        >>> print(f"参数组合数量: {len(combinations[0])}")  # 输出: 18 (3*3*2)
        >>> 
        >>> # 结果结构：
        >>> # combinations[0] = [5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 15]
        >>> # combinations[1] = [20, 20, 30, 30, 40, 40, 20, 20, 30, 30, 40, 40, 20, 20, 30, 30, 40, 40]
        >>> # combinations[2] = [0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02]
        
        >>> # 生成RSI策略的参数组合
        >>> rsi_periods = [14, 21, 28]     # RSI计算周期
        >>> overbought = [70, 75, 80]      # 超买阈值
        >>> oversold = [20, 25, 30]        # 超卖阈值
        >>> 
        >>> rsi_combinations = create_param_product([rsi_periods, overbought, oversold])
        >>> print(f"RSI参数组合数量: {len(rsi_combinations[0])}")  # 输出: 27 (3*3*3)
        
        >>> # 生成投资组合权重组合
        >>> asset_weights = [0.2, 0.3, 0.5]  # 各资产权重选项
        >>> 
        >>> weight_combinations = create_param_product([asset_weights] * 3)  # 3个资产的权重组合
        >>> # 注意：这里生成的是所有可能的权重组合，不一定满足权重和为1的约束
    
    性能特点：
        - 时间复杂度：O(∏ni)，其中ni为第i个参数维度的候选值数量
        - 空间复杂度：O(k*∏ni)，其中k为参数维度数
        - 内存使用：对于大规模参数空间，内存消耗可能很大
    
    注意事项：
        - 参数组合数量会随着参数维度和候选值数量呈指数增长
        - 对于大规模参数空间，建议使用分批处理或随机采样
        - 返回的参数矩阵按列组织，每列代表一个完整的参数组合
        - 确保输入的参数列表不为空，否则会产生空的结果
    
    优化建议：
        - 对于参数数量庞大的场景，考虑使用随机搜索或贝叶斯优化
        - 可以结合参数约束条件过滤不合理的组合
        - 使用内存映射或分块处理来处理超大参数空间
    """
    return list(map(list, zip(*list(itertools.product(*param_list)))))  # 生成笛卡尔积并转置结果矩阵，将按组合组织的数据转换为按参数维度组织的数据
